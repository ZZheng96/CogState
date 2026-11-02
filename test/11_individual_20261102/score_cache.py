"""E14：把候选被试的滑窗分数缓存下来，供各种「个体记忆」决策规则快速对比。

对每个候选被试 q：
  - 训练集 = 不含 q 所在折的其余被试（严格被试隔离）；
  - 模型：LR(C=.01) / LDA(shrinkage) / HGB / ExtraTrees（--models 选择）；
  - 对 q 的 2 s 步长滑窗逐帧打分（decision_function，概率模型用训练中位数定零）；
  - 同时在训练集内部做 3 折 OOF，估计两类分数的均值/标准差，供 HMM 发射模型使用。

产物：scores/sub-XXX.npz  {score__<model>: (n,), y, t, blk, mu0, mu1, sd}
运行：py -3.11 test/11_individual_20261102/score_cache.py --models lr,lda
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "scores"
CAND = ["018", "036", "048", "054", "025", "044", "057", "038", "030", "043", "035", "056", "001"]


def model_of(name):
    if name == "lr":
        return True, LogisticRegression(C=0.01, max_iter=3000)
    if name == "lda":
        return True, LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    if name == "hgb":
        return False, HistGradientBoostingClassifier(max_iter=150, learning_rate=0.06,
                                                     max_leaf_nodes=31, min_samples_leaf=40,
                                                     l2_regularization=1.0, random_state=0)
    if name == "et":
        return False, ExtraTreesClassifier(n_estimators=300, max_features="sqrt",
                                           min_samples_leaf=5, n_jobs=-1, random_state=0)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="lr,lda")
    ap.add_argument("--subjects", default=",".join(CAND))
    a = ap.parse_args()
    models = a.models.split(",")
    subs_target = a.subjects.split(",")
    if subs_target == ["ALL"]:
        subs_target = sorted(p.name for p in (ROOT / "EEG_dataset").iterdir() if p.is_dir())

    D = np.load(ROOT / "test" / "10_projects" / "protocol_20261102" / "realorder.npz")
    F = np.nan_to_num(D["F"].astype(np.float32))
    y, g = D["y"].astype(int), D["g"]
    subs = sorted(np.unique(g))
    folds = np.empty(len(y), int)
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0)
                                .split(np.zeros(len(y)), y, g)):
        folds[te] = k
    fold_of = {q: int(folds[g == q][0]) for q in subs}
    OUT.mkdir(exist_ok=True)

    for q in subs_target:
        p = HERE / "sliding" / f"sub-{q}.npz"
        if not p.exists():
            print(f"[skip] {q}: no sliding features")
            continue
        S = np.load(p)
        Fs, ys, ts, blk = (np.nan_to_num(S["F"].astype(np.float32)), S["y"].astype(int),
                           S["t"], S["blk"])
        o = np.argsort(ts)
        Fs, ys, ts, blk = Fs[o], ys[o], ts[o], blk[o]
        tr = np.flatnonzero(folds != fold_of[q])
        arr, extra = {}, {}
        for name in models:
            scale, mdl = model_of(name)
            sc = StandardScaler().fit(F[tr])
            Xtr = sc.transform(F[tr]) if scale else F[tr]
            Xte = sc.transform(Fs) if scale else Fs
            mdl.fit(Xtr, y[tr])
            if hasattr(mdl, "decision_function"):
                s = mdl.decision_function(Xte)
            else:
                pte = mdl.predict_proba(Xte)[:, 1]
                s = pte - np.median(mdl.predict_proba(Xtr)[:, 1])
            arr[f"score__{name}"] = s.astype(np.float32)
            # 训练集内部 3 折 OOF 估计两类分布（给 HMM 用）
            oof = np.full(len(tr), np.nan)
            for ti, vi in StratifiedGroupKFold(3, shuffle=True, random_state=7).split(
                    np.zeros(len(tr)), y[tr], g[tr]):
                sc2 = StandardScaler().fit(F[tr][ti])
                m2 = type(mdl)(**mdl.get_params())
                m2.fit(sc2.transform(F[tr][ti]), y[tr][ti])
                Xv = sc2.transform(F[tr][vi])
                oof[vi] = (m2.decision_function(Xv) if hasattr(m2, "decision_function")
                           else m2.predict_proba(Xv)[:, 1])
            m0, m1 = oof[y[tr] == 0], oof[y[tr] == 1]
            extra[f"mu0__{name}"] = float(np.nanmean(m0))
            extra[f"mu1__{name}"] = float(np.nanmean(m1))
            extra[f"sd__{name}"] = float(np.nanstd(np.r_[m0, m1]))
            print(f"[{q}] {name}: n={len(ys)} static-AUC={roc_auc_score(ys, s):.4f} "
                  f"mu=({extra[f'mu0__{name}']:.2f},{extra[f'mu1__{name}']:.2f}) "
                  f"sd={extra[f'sd__{name}']:.2f}", flush=True)
        np.savez_compressed(OUT / f"sub-{q}.npz", y=ys, t=ts, blk=blk, **arr, **extra)
    (OUT / "meta.json").write_text(json.dumps({"models": models, "subjects": subs_target},
                                              ensure_ascii=False, indent=1), encoding="utf-8")
    print("[done]")


if __name__ == "__main__":
    main()
