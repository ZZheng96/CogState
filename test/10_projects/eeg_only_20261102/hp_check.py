"""E5b：高通截止频率是否把 α/δ 的判别方向洗掉了？

动机：E5 里 α 相对功率的池化 AUC 只有 0.474（低于随机）、符号一致率 0.58，与"任务→α 抑制"的
经典结论相反；而历史 858 维里最好的单特征是 1–4 Hz（δ）的 q75。δ 更可能对应眼动/慢漂移。
本脚本在同一批窗上比较 0.5/1/2/4 Hz 因果高通后 α、δ 相对功率的 AUC 与跨被试符号一致率，
用来判断"信号本身没有经典 α 效应"还是"被直流/慢漂移污染"。

运行：py -3.11 test/10_projects/eeg_only_20261102/hp_check.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt, welch
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DB = ROOT / "database" / "windows"


def load():
    X, Y, G = [], [], []
    for nm, lab in (("relax", 0), ("task", 1)):
        for p in sorted(DB.joinpath(nm).glob("sub-*.npz")):
            d = np.load(p)
            X.append(d["x"]); Y.append(np.full(len(d["x"]), lab))
            G.append(np.array([p.stem.split("-")[1]] * len(d["x"])))
    return np.concatenate(X).astype(float), np.concatenate(Y), np.concatenate(G)


def bands(x, hp):
    sos = butter(4, hp / 125.0, btype="high", output="sos")
    y = sosfilt(sos, x, axis=1)
    f, p = welch(y, fs=250.0, nperseg=512, noverlap=256, detrend="linear", axis=1)
    def bp(lo, hi):
        m = (f >= lo) & (f < hi)
        return np.trapezoid(p[:, m], f[m], axis=1)
    tot = bp(1, 45) + 1e-30
    return bp(8, 13) / tot, bp(1, 4) / tot


def main():
    X, Y, G = load()
    out = {"windows": int(len(Y)), "rows": []}
    print(f"[data] windows={len(Y)}", flush=True)
    for hp in (0.5, 1.0, 2.0, 4.0):
        a, d = bands(X, hp)
        row = {"hp": hp}
        for nm, v in (("alpha_rel", a), ("delta_rel", d)):
            ds = []
            for s in np.unique(G):
                q = G == s
                x1, x0 = v[q][Y[q] == 1], v[q][Y[q] == 0]
                sd = np.sqrt((x1.var() + x0.var()) / 2) + 1e-12
                ds.append((x1.mean() - x0.mean()) / sd)
            ds = np.array(ds)
            row[nm] = {"AUC": float(roc_auc_score(Y, v)),
                       "consistency": float(np.mean(np.sign(ds) == np.sign(np.median(ds)))),
                       "median_d": float(np.median(ds))}
        out["rows"].append(row)
        print(f"[hp={hp}] alpha AUC={row['alpha_rel']['AUC']:.3f} cons={row['alpha_rel']['consistency']:.2f} "
              f"medD={row['alpha_rel']['median_d']:+.2f} | delta AUC={row['delta_rel']['AUC']:.3f} "
              f"cons={row['delta_rel']['consistency']:.2f} medD={row['delta_rel']['median_d']:+.2f}", flush=True)
    (HERE / "hp_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[done]")


if __name__ == "__main__":
    main()
