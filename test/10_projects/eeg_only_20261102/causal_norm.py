"""E4：因果被试标准化（只用本人已到达的无标签窗）能否把跨被试 0.645 拉近被试内 0.80？

规范：
  * 每个判断仍只用当前 6 s 窗的特征；不使用时间表、窗序号、块序号；
  * 标准化用的参考统计量**只来自本人更早到达的窗**（先预测、后更新），符合规则第 2 条；
  * 代价：有效感受野超过 6 s（虽然决策窗仍为 6 s），所以单独成篇、与 E1 的严格单窗结果并列。

对比四档（同一折、同一模型）：
  none      : 直接用 fold-train 的标准化后送分类器（E1 基线）
  cum       : 再减去本人过去的累计均值、除以累计标准差
  win64     : 再减去本人过去 64 窗的中位数、除以 MAD
  rank64    : 把当前值换成它在本人过去 64 窗中的经验分位（0–1，再用 .5 中心化）

输出：causal_norm.json / .log
运行：py -3.11 test/10_projects/eeg_only_20261102/causal_norm.py
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
T0 = time.perf_counter()
K = 64          # 过去窗数上限（384 s）
N0 = 20         # 预热窗数；不足时退化为不标准化


def el():
    return f"elapsed={time.perf_counter() - T0:.1f}s"


def causal_norm(Zs, mode):
    """Zs: (n, d) 已用 fold-train 统计量标准化。按真实顺序逐窗因果处理。"""
    n, d = Zs.shape
    out = np.empty_like(Zs)
    if mode == "none":
        return Zs
    if mode == "cum":
        s = np.zeros(d); ss = np.zeros(d)
        for t in range(n):
            if t < N0:
                out[t] = Zs[t]
            else:
                mu = s / t
                var = np.maximum(ss / t - mu ** 2, 1e-12)
                out[t] = (Zs[t] - mu) / np.sqrt(var)
            s += Zs[t]; ss += Zs[t] ** 2
        return out
    if mode in ("win64", "rank64"):
        buf = []
        for t in range(n):
            if t < N0:
                out[t] = Zs[t]
            else:
                B = np.asarray(buf[-K:])
                if mode == "win64":
                    med = np.median(B, 0)
                    mad = np.median(np.abs(B - med), 0) + 1e-6
                    out[t] = (Zs[t] - med) / (1.4826 * mad)
                else:
                    out[t] = (B < Zs[t][None, :]).mean(0) - 0.5
            buf.append(Zs[t])
        return out
    raise ValueError(mode)


def metrics(y, s, g):
    yh = (s >= 0).astype(int)
    per = [balanced_accuracy_score(y[g == q], yh[g == q]) for q in np.unique(g)]
    return {"AUC": float(roc_auc_score(y, s)), "bACC": float(balanced_accuracy_score(y, yh)),
            "F1_macro": float(f1_score(y, yh, average="macro")),
            "F1_task": float(f1_score(y, yh, average="binary", pos_label=1, zero_division=0)),
            "subject_macro_bACC": float(np.mean(per)),
            "n_ge_0.9": int(np.sum(np.array(per) >= 0.9)), "n_subj": len(per)}


def main():
    D = np.load(ROOT / "test" / "10_projects" / "protocol_20261102" / "realorder.npz")
    F, y, g, t = D["F"], D["y"].astype(int), D["g"], D["t"]
    Z0 = np.nan_to_num(F.astype(float))
    print(f"[data] windows={len(y)} subs={len(np.unique(g))} dim={Z0.shape[1]} {el()}", flush=True)
    folds = np.empty(len(y), int)
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y, g)):
        folds[te] = k

    MODES = ("none", "cum", "win64", "rank64")
    res = {"protocol": "real-order, cross-subject SGKF(5,rs0); causal per-subject normalisation "
                       "(past unlabeled windows only) -> effective receptive field > 6 s",
           "K": K, "N0": N0, "methods": {}, "seconds": 0.0}

    for model in ("lr", "hgb"):
        S = {m: np.full(len(y), np.nan) for m in MODES}
        for k in range(5):
            tr = np.flatnonzero(folds != k)
            sc = StandardScaler().fit(Z0[tr])
            A = np.nan_to_num(sc.transform(Z0[tr]))
            if model == "lr":
                clf = LogisticRegression(C=.01, max_iter=2000).fit(A, y[tr])
            else:
                clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=3,
                                                     min_samples_leaf=40, l2_regularization=1.0,
                                                     random_state=0).fit(A, y[tr])
            for s in np.unique(g[folds == k]):
                loc = np.flatnonzero((g == s) & (folds == k))
                loc = loc[np.argsort(t[loc], kind="stable")]        # 真实顺序
                Zs = np.nan_to_num(sc.transform(Z0[loc]))
                for m in MODES:
                    Zc = causal_norm(Zs, m)
                    p = clf.predict_proba(Zc)[:, 1]
                    S[m][loc] = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))
            print(f"[{model} fold {k}] {el()}", flush=True)
        for m in MODES:
            mm = metrics(y, S[m], g)
            res["methods"][f"{model}|{m}"] = mm
            print(f"[result {model}|{m:6s}] AUC={mm['AUC']:.4f} bACC={mm['bACC']:.4f} "
                  f"F1m={mm['F1_macro']:.4f} macro={mm['subject_macro_bACC']:.4f} "
                  f"n>=.9={mm['n_ge_0.9']}/{mm['n_subj']}", flush=True)

    res["seconds"] = time.perf_counter() - T0
    (HERE / "causal_norm.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[done] {res['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
