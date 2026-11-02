"""E16：为单个被试生成新候选（特征子集 × 标准化窗 × EMA h × 是否剔除训练块首段），LOBO、因果。
被评块只用于测试；"drop30" 只剔除训练块（已知标签）前 30 帧的过渡段。
输出 scores_ps/sub-XXX.npz（score__ps_*），wsearch/diag 会自动并入。
运行：.venv\\Scripts\\python.exe test/16_subject_push/ps_cands.py 040
"""
from __future__ import annotations

import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "15_nested_select"))
sys.path.insert(0, str(HERE.parent / "11_individual_20261102"))
from feat_cstd import running_stats  # noqa: E402
from smoothfeat import ema_mat, lr_fit, fscore  # noqa: E402

V2 = "--v2" in sys.argv  # 更短的标准化窗
V3 = "--v3" in sys.argv  # k / C / 剔除段长度
V4 = "--v4" in sys.argv  # 纯 EEG 子集、更长 h、更长剔除段（针对块后段漂移）
V5 = "--v5" in sys.argv  # 大 k + 强正则 + 更长标准化窗
OUT = HERE / ("scores_ps2" if V2 else "scores_ps3" if V3 else "scores_ps4" if V4 else
              "scores_ps5" if V5 else "scores_ps")
WS = (45, 60, 90) if V2 else (120, 300) if V3 else (60, 120, 300) if V4 else (120, 300, 900)
HS_ = (10, 20, 40) if V3 else (20, 40, 80) if V4 else (5, 10, 20)
KC = [(10, 0.01), (20, 0.1), (100, 0.001)] if V3 else [(20, 0.1), (50, 0.01)] if V4 else [(50, 0.01)]
DROPS = (0, 60) if V3 else (60, 120) if V4 else (0, 30)
if V5:
    WS, HS_, DROPS = (300, 600), (30, 40, 60), (30, 60)
    KC = [(100, 0.0003), (200, 0.001), (200, 0.0003), (400, 0.0003)]
V6 = "--v6" in sys.argv  # 特征层 DEMA（2·EMA−EMA∘EMA），同等平滑下滞后更小
if V6:
    OUT = HERE / "scores_ps6"
    WS, HS_, DROPS = (120, 300), (20, 40, 60, 90), (30, 60)
    KC = [(20, 0.1), (100, 0.001), (100, 0.0003)]
man = json.loads((ROOT / "database" / "manifest.json").read_text(encoding="utf-8"))
names = man["feature_names"] + man["emg_feature_names"]
idx = lambda pat: [i for i, n in enumerate(names) if re.search(pat, n)]  # noqa: E731
SUBSETS = {
    "emg": idx(r"^emg_"),
    "spm": idx(r"^stft_.*_(log|relative)_.*_mean$"),
    "spmemg": idx(r"^emg_|^stft_.*_(log|relative)_.*_mean$"),
    "all": list(range(len(names))),
}
if V5:
    SUBSETS = {"all": SUBSETS["all"]}
if V6:
    SUBSETS = {"all": SUBSETS["all"], "emg": SUBSETS["emg"], "spm": SUBSETS["spm"]}
if V4:
    SUBSETS = {"eeg": idx(r"^(?!emg_)"), "spmemg": SUBSETS["spmemg"], "all": SUBSETS["all"]}


def main(q):
    OUT.mkdir(exist_ok=True)
    S = np.load(ROOT / "test" / "11_individual_20261102" / "sliding1" / f"sub-{q}.npz")
    X = np.nan_to_num(S["F"].astype(np.float32))
    y, blk, t = S["y"].astype(int), S["blk"].astype(int), S["t"]
    o = np.argsort(t)
    X, y, blk, t = X[o], y[o], blk[o], t[o]
    pos = np.zeros(len(y), int)
    for b in np.unique(blk):
        pos[blk == b] = np.arange((blk == b).sum())
    out = {}
    for w in WS:
        m, sd = running_stats(X, w=w)
        Xc = np.nan_to_num((X - m) / sd).astype(np.float32)
        for h in HS_:
            Xh = ema_mat(Xc, h)
            if V6:
                Xh = 2 * Xh - ema_mat(Xh, h)
            for (sname, cols), (k0, C) in itertools.product(SUBSETS.items(), KC):
                Xs = Xh[:, cols]
                k = min(k0, len(cols))
                for drop in DROPS:
                    s = np.zeros(len(y))
                    for b in np.unique(blk):
                        te = blk == b
                        tr = (blk != b) & (pos >= drop)
                        mu, sg = Xs[tr].mean(0), Xs[tr].std(0) + 1e-6
                        A, B = (Xs[tr] - mu) / sg, (Xs[te] - mu) / sg
                        sel = np.argsort(-fscore(A, y[tr]))[:k]
                        dtr, dte = lr_fit(A[:, sel], y[tr], B[:, sel], C=C)
                        s[te] = (dte - dtr.mean()) / (dtr.std() + 1e-12)
                    tag = f"_k{k0}C{C}" if (V3 or V4 or V5) else f"_k{k0}C{C}_dema" if V6 else ""
                    out[f"score__ps_{sname}_w{w}_h{h}_d{drop}{tag}"] = s.astype(np.float32)
        print(f"[{q}] w={w} done", flush=True)
    np.savez_compressed(OUT / f"sub-{q}.npz", y=y.astype(np.int8), blk=blk.astype(np.int16), t=t, **out)


if __name__ == "__main__":
    main(sys.argv[1])
