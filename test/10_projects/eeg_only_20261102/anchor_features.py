"""E5：生理锚点能否给"本人方向"定符号？

思路：跨被试失败的核心是约 1/3 被试的判别方向相反，而无标签数据里不能定符号（label switching）。
但如果存在一个**生理上方向固定**的量，它就能当锚。最自然的候选是"闭眼/睁眼"：
  * 放松段通常是闭眼静息 -> α（8–13 Hz）功率高、谱峰集中、眨眼少；
  * 任务段要操作机械臂/看屏幕 -> 睁眼 -> α 抑制、眨眼与眼动多。
若 α 峰聚集度、眨眼率这类**正值派生量**在 57 人上方向一致（不随被试翻转），就能绕开符号问题。

本脚本只做诊断与最小验证：
  1. 从当前 6 s 原始窗算 8 个派生量；
  2. 逐被试算 Cohen's d，报符号一致率与池化 AUC；
  3. 只用这些派生量做跨被试 5 折（无任何次序信息），看能否超过 0.645。

运行：py -3.11 test/10_projects/eeg_only_20261102/anchor_features.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DB = ROOT / "database" / "windows"
FS = 250.0
T0 = time.perf_counter()
NAMES = ["alpha_rel", "alpha_peakiness", "theta_rel", "alpha_theta", "blink_rate",
         "lowfreq_rel", "one_over_f", "spec_entropy"]


def el():
    return f"elapsed={time.perf_counter() - T0:.1f}s"


def anchors(x):
    x = np.asarray(x, dtype=np.float64)
    f, p = welch(x, fs=FS, nperseg=512, noverlap=256, detrend="linear", axis=1)
    def band(lo, hi):
        m = (f >= lo) & (f < hi)
        return np.trapezoid(p[:, m], f[m], axis=1)
    tot = band(1, 45) + 1e-30
    a, th, lf = band(8, 13), band(4, 8), band(0.5, 4)
    ma = (f >= 8) & (f < 13)
    peak = p[:, ma].max(1) + 1e-30
    mean = p[:, ma].mean(1) + 1e-30
    # 1/f 斜率（2–40 Hz，log-log 线性拟合）
    mf = (f >= 2) & (f < 40)
    lf_, lp_ = np.log(f[mf] + 1e-9), np.log(p[:, mf] + 1e-30)
    lf_c = lf_ - lf_.mean()
    slope = (lp_ - lp_.mean(1, keepdims=True)) @ lf_c / (lf_c @ lf_c)
    # 谱熵
    pn = p[:, (f >= 1) & (f < 45)]
    pn = pn / (pn.sum(1, keepdims=True) + 1e-30)
    ent = -(pn * np.log(np.maximum(pn, 1e-30))).sum(1) / np.log(pn.shape[1])
    # 眨眼：0.5–4 Hz 分量上的峰计数
    blinks = np.zeros(len(x))
    for i in range(len(x)):
        seg = x[i] - np.convolve(x[i], np.ones(25) / 25, mode="same")
        sp = np.abs(seg)
        thr = 3 * np.median(sp) + 1e-30
        pk, _ = find_peaks(sp, height=thr, distance=100)   # 至少 0.4 s 间隔
        blinks[i] = len(pk)
    return np.column_stack([a / tot, peak / mean, th / tot, a / (th + 1e-30),
                            blinks, lf / tot, slope, ent])


def load():
    X, Y, G = [], [], []
    for nm, lab in (("relax", 0), ("task", 1)):
        for p in sorted(DB.joinpath(nm).glob("sub-*.npz")):
            d = np.load(p)
            X.append(d["x"]); Y.append(np.full(len(d["x"]), lab))
            G.append(np.array([p.stem.split("-")[1]] * len(d["x"])))
    return np.concatenate(X), np.concatenate(Y).astype(int), np.concatenate(G)


def main():
    X, y, g = load()
    A = anchors(X)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    subs = np.unique(g)
    print(f"[data] windows={len(y)} subs={len(subs)} anchors={A.shape[1]} {el()}", flush=True)

    D = np.zeros((len(subs), len(NAMES)))
    for i, s in enumerate(subs):
        q = g == s
        a, b = A[q][y[q] == 1], A[q][y[q] == 0]
        sd = np.sqrt((a.var(0) + b.var(0)) / 2) + 1e-12
        D[i] = (a.mean(0) - b.mean(0)) / sd
    med = np.median(D, 0)
    cons = (np.sign(D) == np.sign(med)).mean(0)
    auc = [roc_auc_score(y, A[:, j]) for j in range(A.shape[1])]
    rows = []
    for j, nm in enumerate(NAMES):
        rows.append({"name": nm, "AUC": float(auc[j]), "consistency": float(cons[j]), "median_d": float(med[j])})
        print(f"[anchor] {nm:16s} AUC={auc[j]:.3f} cons={cons[j]:.2f} medD={med[j]:+.2f}", flush=True)

    folds = np.empty(len(y), int)
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y, g)):
        folds[te] = k
    s = np.full(len(y), np.nan)
    for k in range(5):
        tr, te = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        sc = StandardScaler().fit(A[tr])
        m = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(A[tr]), y[tr])
        s[te] = m.decision_function(sc.transform(A[te]))
    yh = (s >= 0).astype(int)
    per = [balanced_accuracy_score(y[g == q], yh[g == q]) for q in subs]
    res = {"n_windows": int(len(y)), "n_subjects": int(len(subs)), "anchors": rows,
           "cv": {"AUC": float(roc_auc_score(y, s)), "bACC": float(balanced_accuracy_score(y, yh)),
                  "F1_macro": float(f1_score(y, yh, average="macro")),
                  "subject_macro_bACC": float(np.mean(per)),
                  "n_ge_0.9": int(np.sum(np.array(per) >= 0.9)), "n_subj": len(per)},
           "seconds": 0.0}
    c = res["cv"]
    print(f"[cv anchors-only] AUC={c['AUC']:.4f} bACC={c['bACC']:.4f} F1m={c['F1_macro']:.4f} "
          f"macro={c['subject_macro_bACC']:.4f} n>=.9={c['n_ge_0.9']}/{c['n_subj']}", flush=True)
    res["seconds"] = time.perf_counter() - T0
    (HERE / "anchor_features.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[done] {res['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
