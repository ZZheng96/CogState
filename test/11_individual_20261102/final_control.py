"""E16：对最终方法做零假设控制——bACC 0.928 是 EEG 还是"块交替 + 持久先验"？

最终方法（018, D=60 s）：
    raw score → 因果中位数/MAD 标准化（过去 300 s）→ 两态 HMM 前向滤波（p_stay=1-stride/D）
控制（标签与时间都不动，只动分数序列）：
    shift    循环平移随机偏移（保留自相关与边缘分布，破坏与标签的对齐）
    shuffle  被试内随机打乱（破坏自相关）
    ar1      匹配 lag-1 自相关的 AR(1) 替代序列（只有"慢"、没有状态信息）
    label_shift 把标签序列循环平移（保留元数据，破坏与分数的对齐）

运行：py -3.11 test/11_individual_20261102/final_control.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SC = HERE / "scores"
STRIDE = 2.0


def hmm_forward(x, mu0, mu1, sd, p_stay):
    T = np.array([[p_stay, 1 - p_stay], [1 - p_stay, p_stay]])
    logT = np.log(T + 1e-12)
    la = np.log(np.array([0.5, 0.5]))
    out = np.empty(len(x))
    for t, v in enumerate(x):
        if t:
            a = la[:, None] + logT
            m = a.max()
            la = m + np.log(np.exp(a - m).sum(axis=0))
        la = la - 0.5 * ((v - np.array([mu0, mu1])) / sd) ** 2
        m = la.max()
        la = la - (m + np.log(np.exp(la - m).sum()))
        out[t] = np.exp(la[1])
    return out


def cstd(x, w=150):
    med = np.array([np.median(x[max(0, i - w):i + 1]) for i in range(len(x))])
    mad = np.array([np.median(np.abs(x[max(0, i - w):i + 1] - med[i])) + 1e-6
                    for i in range(len(x))])
    return (x - med) / (1.4826 * mad)


def main():
    rng = np.random.default_rng(0)
    rows = []
    subs = ["018", "054", "036", "044"]
    for q in subs:
        Z = np.load(SC / f"sub-{q}.npz")
        y = Z["y"].astype(int)
        x = Z["score__lr"].astype(float)
        mu0, mu1, sd = float(Z["mu0__lr"]), float(Z["mu1__lr"]), float(Z["sd__lr"])
        p = 1 - STRIDE / 60

        def run(xs):
            pr = hmm_forward(cstd(xs), mu0, mu1, sd, p)
            return (roc_auc_score(y, pr), balanced_accuracy_score(y, (pr >= 0.5).astype(int)),
                    f1_score(y, (pr >= 0.5).astype(int)))

        real = run(x)
        ctl = {k: [] for k in ("shift", "shuffle", "ar1", "label_shift")}
        rho = float(np.corrcoef(x[:-1], x[1:])[0, 1])
        n = len(x)
        for _ in range(200):
            off = int(rng.integers(20, n - 20))
            ctl["shift"].append(run(np.roll(x, off))[1])
            xs = x.copy()
            rng.shuffle(xs)
            ctl["shuffle"].append(run(xs)[1])
            e = rng.normal(size=n)
            ar = np.empty(n)
            ar[0] = e[0]
            for i in range(1, n):
                ar[i] = rho * ar[i - 1] + np.sqrt(max(1 - rho ** 2, 1e-6)) * e[i]
            ctl["ar1"].append(run(ar)[1])
            ctl["label_shift"].append(
                balanced_accuracy_score(np.roll(y, off), (hmm_forward(cstd(x), mu0, mu1, sd, p) >= .5).astype(int)))
        row = {"subject": q, "real_AUC": real[0], "real_bACC": real[1], "real_F1": real[2],
               "lag1": rho}
        for k, v in ctl.items():
            row[f"{k}_mean"] = float(np.mean(v))
            row[f"{k}_p95"] = float(np.percentile(v, 95))
            row[f"{k}_max"] = float(np.max(v))
        rows.append(row)
        print(f"{q}: real AUC={real[0]:.3f} bACC={real[1]:.3f} F1={real[2]:.3f} | "
              f"shift {np.mean(ctl['shift']):.3f} | shuffle {np.mean(ctl['shuffle']):.3f} | "
              f"ar1 {np.mean(ctl['ar1']):.3f} (p95 {np.percentile(ctl['ar1'],95):.3f}) | "
              f"label_shift {np.mean(ctl['label_shift']):.3f}", flush=True)
    (HERE / "final_control.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    print("[done] final_control.json")


if __name__ == "__main__":
    main()
