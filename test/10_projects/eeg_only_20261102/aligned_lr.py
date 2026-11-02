"""E3：符号对齐（sign-aligned）特征子集——能否削弱"1/3 被试方向相反"的影响？

动机（E2 诊断）：878 维里没有任何单一特征同时具备"高效应"与"高符号一致率"，
平均符号一致率只有 0.609。但若把每个特征按训练折上的主流符号翻转，再只保留高一致率的特征，
模型就不必在噪声特征上浪费容量，可能比直接用全部 878 维更好。

做法（全部只用训练折被试的标签；测试被试不参与任何选择）：
  1. 训练折内逐被试算 Cohen's d，得到每特征的符号一致率 cons 与中位符号 s；
  2. 选中 cons ≥ τ 且 |池化 AUC − 0.5| ≥ e 的特征，并按 -s 翻转，使全体的主流方向一致；
  3. 在翻转后的子集上拟合 LR / HGB；
  4. τ、e 由训练折内的 4 折被试分组 CV 选择（不碰测试被试）。

输出：aligned_lr.json / .log
运行：py -3.11 test/10_projects/eeg_only_20261102/aligned_lr.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DB = ROOT / "database" / "windows"
T0 = time.perf_counter()
TAUS = (0.60, 0.70, 0.80)
EFFS = (0.05, 0.15, 0.30)      # |中位 Cohen's d| 下限（比逐特征算 AUC 便宜得多）


def el():
    return f"elapsed={time.perf_counter() - T0:.1f}s"


def load():
    F, E, Y, G = [], [], [], []
    for nm, lab in (("relax", 0), ("task", 1)):
        for p in sorted(DB.joinpath(nm).glob("sub-*.npz")):
            d = np.load(p)
            F.append(d["f"]); E.append(d["e"])
            Y.append(np.full(len(d["f"]), lab)); G.append(np.array([p.stem.split("-")[1]] * len(d["f"])))
    return np.hstack([np.concatenate(F), np.concatenate(E)]), np.concatenate(Y).astype(int), np.concatenate(G)


def cohen(Z, y, g, rows):
    """rows 内逐被试 Cohen's d 矩阵 (n_subj, dim)。"""
    subs = np.unique(g[rows])
    D = np.zeros((len(subs), Z.shape[1]))
    for i, s in enumerate(subs):
        q = rows[g[rows] == s]
        a, b = Z[q][y[q] == 1], Z[q][y[q] == 0]
        if len(a) < 2 or len(b) < 2:
            continue
        sd = np.sqrt((a.var(0) + b.var(0)) / 2) + 1e-12
        D[i] = (a.mean(0) - b.mean(0)) / sd
    return D, subs


def select(Z, y, g, rows, tau, eff):
    D, _ = cohen(Z, y, g, rows)
    med = np.median(D, 0)
    cons = (np.sign(D) == np.sign(med)).mean(0)
    keep = (cons >= tau) & (np.abs(med) >= eff)
    return keep, np.sign(med)


def fit_eval(Z, y, g, tr, te, tau, eff, model="lr", C=0.01):
    keep, sg = select(Z, y, g, tr, tau, eff)
    if keep.sum() < 3:
        return None
    sg = np.where(sg == 0, 1.0, sg)
    Zt = Z * sg                      # 翻转特征，使主流方向一致
    sc = StandardScaler().fit(Zt[tr][:, keep])
    A, B = sc.transform(Zt[tr][:, keep]), sc.transform(Zt[te][:, keep])
    A, B = np.nan_to_num(A), np.nan_to_num(B)
    if model == "lr":
        m = LogisticRegression(C=C, max_iter=2000).fit(A, y[tr])
        s = m.decision_function(B)
    else:
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=3,
                                           min_samples_leaf=40, l2_regularization=1.0,
                                           random_state=0).fit(A, y[tr])
        p = m.predict_proba(B)[:, 1]
        s = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))
    return s, int(keep.sum())


def metrics(y, s, g):
    yh = (s >= 0).astype(int)
    per = [balanced_accuracy_score(y[g == q], yh[g == q]) for q in np.unique(g)]
    return {"AUC": float(roc_auc_score(y, s)), "bACC": float(balanced_accuracy_score(y, yh)),
            "F1_macro": float(f1_score(y, yh, average="macro")),
            "F1_task": float(f1_score(y, yh, average="binary", pos_label=1, zero_division=0)),
            "subject_macro_bACC": float(np.mean(per)),
            "n_ge_0.9": int(np.sum(np.array(per) >= 0.9)), "n_subj": len(per)}


def main():
    Z, y, g = load()
    print(f"[data] windows={len(y)} subs={len(np.unique(g))} dim={Z.shape[1]}", flush=True)
    folds = np.empty(len(y), int)
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y, g)):
        folds[te] = k

    out = {"protocol": "cross-subject SGKF(5,rs0); sign-aligned feature subset; selection only on train folds",
           "grid": {"tau": list(TAUS), "eff": list(EFFS)}, "methods": {}, "chosen": {}, "seconds": 0.0}

    for model in ("lr",):
        s_all = np.full(len(y), np.nan)
        picks = {}
        for k in range(5):
            tr, te = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
            # 内层：在训练被试上 4 折选 tau/eff
            inner = np.empty(len(y), int)
            subs_tr = np.unique(g[tr])
            for j, (_, iv) in enumerate(StratifiedGroupKFold(4, shuffle=True, random_state=0).split(
                    np.zeros(len(tr)), y[tr], g[tr])):
                inner[tr[iv]] = -1 - j
            best, bv = (0.6, 0.03), -1.0
            for tau in TAUS:
                for eff in EFFS:
                    pred = np.full(len(y), np.nan)
                    for j in range(4):
                        itr = tr[inner[tr] == -1 - j]
                        iva = tr[inner[tr] != -1 - j]
                        r = fit_eval(Z, y, g, itr, iva, tau, eff, model)
                        if r is not None:
                            pred[iva] = r[0]
                    ok = np.isfinite(pred) & np.isin(np.arange(len(y)), tr)
                    if ok.sum() > 100:
                        v = balanced_accuracy_score(y[ok], (pred[ok] >= 0).astype(int))
                        if v > bv:
                            best, bv = (tau, eff), v
            picks[str(k)] = {"tau": best[0], "eff": best[1], "inner_bACC": float(bv)}
            r = fit_eval(Z, y, g, tr, te, best[0], best[1], model)
            s_all[te] = r[0]
            print(f"[{model} fold {k}] tau={best[0]} eff={best[1]} n_feat={r[1]} inner={bv:.4f} {el()}", flush=True)
        m = metrics(y, s_all, g)
        out["methods"][model] = m
        out["chosen"][model] = picks
        print(f"[result {model}] AUC={m['AUC']:.4f} bACC={m['bACC']:.4f} F1m={m['F1_macro']:.4f} "
              f"macro={m['subject_macro_bACC']:.4f} n>=.9={m['n_ge_0.9']}/{m['n_subj']}", flush=True)

    out["seconds"] = time.perf_counter() - T0
    (HERE / "aligned_lr.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[done] {out['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
