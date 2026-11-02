"""E9：同一协议下的方法银行 → 方法 × 57 被试矩阵。

目的：用户要求「每一种方法 × 每一个被试」的 n×57 效果表，并从中找规律。
历史 JSON 的列协议不完全一致（有的含历史/融合/时间表），不能直接横比。
因此在 realorder.npz 的 878 维特征上，用**完全相同**的协议重跑一批方法：

  - 单窗口 6 s，只用当前窗特征（无位置/次序/已到达窗数）
  - StratifiedGroupKFold(5, shuffle=True, random_state=0)，被试不跨折
  - 标准化只在训练折拟合
  - 记录逐被试 AUC / bACC / F1（单类被试 AUC 记 NaN）

除逐被试指标外，还输出：
  - 每折训练分数量纲（用于跨方法 z 化后做共识集成）
  - 共识集成（各方法 z 分数均值）、逐被试 oracle 选方法上界

运行：py -3.11 test/10_projects/eeg_only_20261102/method_bank.py
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
T0 = time.perf_counter()


def el():
    return f"elapsed={time.perf_counter() - T0:.1f}s"


def build_methods():
    """name -> (需要标准化?, pipeline)"""
    lr = lambda C: LogisticRegression(C=C, max_iter=3000)
    return {
        "lr_c001": (True, lr(0.01)),
        "lr_c1": (True, lr(1.0)),
        "lda_shrink": (True, LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        "nystroem_svc": (True, Pipeline([("ny", Nystroem(gamma=1.0 / 878, n_components=400, random_state=0)),
                                         ("lr", LogisticRegression(C=0.1, max_iter=2000))])),
        "ridge": (True, RidgeClassifier(alpha=1000.0)),
        "hgb": (False, HistGradientBoostingClassifier(max_iter=150, learning_rate=0.06,
                                                      max_leaf_nodes=31, min_samples_leaf=40,
                                                      l2_regularization=1.0, random_state=0)),
        "extratrees": (False, ExtraTreesClassifier(n_estimators=300, max_features="sqrt",
                                                   min_samples_leaf=5, n_jobs=-1, random_state=0)),
        "rf": (False, RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                             min_samples_leaf=5, n_jobs=-1, random_state=0)),
        "knn50_pca32": (True, Pipeline([("pca", PCA(32, random_state=0)),
                                        ("knn", KNeighborsClassifier(50, weights="distance", n_jobs=-1))])),
        "pca16_lr": (True, Pipeline([("pca", PCA(16, random_state=0)), ("lr", lr(0.1))])),
        "pca64_lr": (True, Pipeline([("pca", PCA(64, random_state=0)), ("lr", lr(0.1))])),
        "gnb_pca32": (True, Pipeline([("pca", PCA(32, random_state=0)), ("gnb", GaussianNB())])),
        "mlp256": (True, MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-3, batch_size=256,
                                       learning_rate_init=1e-3, max_iter=120, early_stopping=True,
                                       n_iter_no_change=8, validation_fraction=0.1, random_state=0)),
    }


SUBSETS = {
    "lr_emg": "emg",
    "lr_stft": "stft",
    "lr_dwt": "dwt",
    "lr_hilbert": "hilbert",
}


def subset_indices(names):
    emg = [i for i, n in enumerate(names) if n.startswith("emg_")]
    out = {"emg": emg, "stft": [i for i, n in enumerate(names) if n.startswith("stft_")],
           "dwt": [i for i, n in enumerate(names) if n.startswith("dwt_")],
           "hilbert": [i for i, n in enumerate(names) if n.startswith("hilbert_")]}
    return out


def per_subject_metrics(y, s, subs):
    out = {}
    for q in subs:
        sel = subs[q]
        yy, ss = y[sel], s[sel]
        if len(np.unique(yy)) < 2:
            out[q] = {"auc": None, "bacc": None, "f1": None, "n": int(len(yy))}
            continue
        yh = (ss >= 0).astype(int)
        out[q] = {"auc": float(roc_auc_score(yy, ss)),
                  "bacc": float(balanced_accuracy_score(yy, yh)),
                  "f1": float(f1_score(yy, yh, average="binary")),
                  "n": int(len(yy))}
    return out


def pooled(y, s, subs):
    yh = (s >= 0).astype(int)
    per_b = [v["bacc"] for v in per_subject_metrics(y, s, subs).values() if v["bacc"] is not None]
    return {"AUC": float(roc_auc_score(y, s)), "bACC": float(balanced_accuracy_score(y, yh)),
            "F1_macro": float(f1_score(y, yh, average="macro")),
            "subject_macro_bACC": float(np.mean(per_b))}


def main():
    D = np.load(ROOT / "test" / "10_projects" / "protocol_20261102" / "realorder.npz")
    F = np.nan_to_num(D["F"].astype(np.float32))
    y = D["y"].astype(int)
    g = D["g"]
    subs = {q: np.flatnonzero(g == q) for q in sorted(np.unique(g))}
    man = json.loads((ROOT / "database" / "manifest.json").read_text(encoding="utf-8"))
    names = man["feature_names"] + man["emg_feature_names"]
    assert F.shape[1] == len(names), (F.shape, len(names))
    sub_idx = subset_indices(names)
    print(f"[data] n={len(y)} subs={len(subs)} dim={F.shape[1]} {el()}", flush=True)
    for k, v in sub_idx.items():
        print(f"  subset {k}: {len(v)}", flush=True)

    folds = np.empty(len(y), int)
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0)
                                .split(np.zeros(len(y)), y, g)):
        folds[te] = k

    methods = build_methods()
    for nm, key in SUBSETS.items():
        methods[nm] = (True, LogisticRegression(C=0.01, max_iter=3000))

    results = {}
    zscores = {}
    part = HERE / "method_bank_partial.json"
    done = json.loads(part.read_text(encoding="utf-8")) if part.exists() else {}
    for nm, (scale, model) in methods.items():
        cache = HERE / f"mb_scores_{nm}.npz"
        if nm in done and cache.exists():
            c = np.load(cache)
            zscores[nm] = c["sz"]
            results[nm] = done[nm]
            p = results[nm]["pooled"]
            print(f"[{nm:14s}] (cached) AUC={p['AUC']:.4f} bACC={p['bACC']:.4f}", flush=True)
            continue
        t0 = time.perf_counter()
        cols = sub_idx.get(SUBSETS.get(nm, ""), None)
        s = np.full(len(y), np.nan)
        sz = np.full(len(y), np.nan)
        for k in range(5):
            tr, te = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
            Xtr, Xte = (F[tr], F[te]) if cols is None else (F[tr][:, cols], F[te][:, cols])
            if scale:
                sc = StandardScaler().fit(Xtr)
                Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
            mdl = model
            # 重新实例化，避免方法间共享已拟合状态
            mdl = type(model)(**model.get_params()) if not isinstance(model, Pipeline) else Pipeline(
                [(n_, type(est)(**est.get_params())) for n_, est in model.steps])
            mdl.fit(Xtr, y[tr])
            if hasattr(mdl, "decision_function"):
                dtr, dte = mdl.decision_function(Xtr), mdl.decision_function(Xte)
            else:
                # 概率模型用训练折中位数做零点，避免类别不平衡把阈值推歪
                ptr, pte = mdl.predict_proba(Xtr)[:, 1], mdl.predict_proba(Xte)[:, 1]
                c = np.median(ptr)
                dtr, dte = ptr - c, pte - c
            s[te] = dte
            sz[te] = (dte - dtr.mean()) / (dtr.std() + 1e-12)
        zscores[nm] = sz
        pm = per_subject_metrics(y, s, subs)
        results[nm] = {"pooled": pooled(y, s, subs), "per_subject": pm, "seconds": time.perf_counter() - t0}
        np.savez_compressed(cache, sz=sz)
        done[nm] = results[nm]
        part.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
        p = results[nm]["pooled"]
        print(f"[{nm:14s}] AUC={p['AUC']:.4f} bACC={p['bACC']:.4f} F1m={p['F1_macro']:.4f} "
              f"smACC={p['subject_macro_bACC']:.4f} ({results[nm]['seconds']:.1f}s) {el()}", flush=True)

    # 共识集成：各方法 z 分数均值
    Z = np.vstack([zscores[k] for k in zscores])
    cons = np.nanmean(Z, axis=0)
    results["consensus_all"] = {"pooled": pooled(y, cons, subs),
                                "per_subject": per_subject_metrics(y, cons, subs), "seconds": 0.0}
    p = results["consensus_all"]["pooled"]
    print(f"[consensus_all ] AUC={p['AUC']:.4f} bACC={p['bACC']:.4f} F1m={p['F1_macro']:.4f}", flush=True)

    # 逐被试 oracle 选方法上界（诊断，不可部署）
    names_m = list(zscores)
    oracle_best, oracle_choice = {}, {}
    for q, sel in subs.items():
        best, bn = -1.0, None
        for nm in names_m:
            v = results[nm]["per_subject"][q]["auc"]
            if v is not None and v > best:
                best, bn = v, nm
        oracle_best[q] = best
        oracle_choice[q] = bn
    print(f"[oracle] 逐被试选方法 AUC 均值={np.mean(list(oracle_best.values())):.4f} "
          f"中位={np.median(list(oracle_best.values())):.4f} {el()}", flush=True)

    # 逐方法在 57 人上的排名一致性
    mat = np.array([[results[nm]["per_subject"][q]["auc"] for q in sorted(subs)] for nm in names_m], float)
    ranks = np.apply_along_axis(lambda r: np.argsort(np.argsort(-np.where(np.isnan(r), -9, r))), 1, mat)
    subj_rank_std = np.nanstd(ranks, axis=0)

    out = {"protocol": {"window_seconds": 6, "features": 878, "cv": "SGKF(5,shuffle,rs0)",
                        "position_free": True, "note": "所有方法仅用当前窗特征；标准化只拟合训练折"},
           "methods": {nm: results[nm]["pooled"] for nm in results},
           "per_subject": {nm: results[nm]["per_subject"] for nm in results},
           "oracle_best_method": oracle_best, "oracle_choice": oracle_choice,
           "subject_rank_std": dict(zip(sorted(subs), subj_rank_std.tolist())),
           "seconds": time.perf_counter() - T0}
    (HERE / "method_bank.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez_compressed(HERE / "method_bank_scores.npz", y=y, g=g, folds=folds,
                        **{f"s__{nm}": np.nan_to_num(v, nan=0.0) for nm, v in zscores.items()})
    print(f"[done] {out['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
