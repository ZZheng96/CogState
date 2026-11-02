"""E15-3：固定配置（全体被试同一配置、无逐人选择）的 bACC/AUC/F1 排行。

固定配置不存在逐人选择偏差，是比"嵌套"更诚实的口径；配置本身在全体上挑一次（16–40 个里选 1 个）。
运行：.venv/Scripts/python.exe test/15_nested_select/eval_fixed.py scores_sf scores_sf2
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import lfilter
from sklearn.metrics import f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "11_individual_20261102" / "scores1"


def ema(x, h):
    if h <= 1:
        return x
    a = 0.5 ** (1.0 / h)
    return lfilter([1 - a], [1, -a], x)


def fb(y, p):
    return 0.5 * ((p[y == 1] == 1).mean() + (p[y == 0] == 0).mean())


def main():
    dirs = [HERE / d for d in sys.argv[1:]]
    subs = sorted(p.stem[4:] for p in BASE.glob("sub-*.npz"))
    R = {}
    for q in subs:
        Z = np.load(BASE / f"sub-{q}.npz")
        y = Z["y"].astype(int)
        S = {"own_k": Z["score__own_k"].astype(float)}
        for d in dirs:
            f = d / f"sub-{q}.npz"
            if f.exists():
                W = np.load(f)
                S.update({k[7:]: W[k].astype(float) for k in W.files if k.startswith("score__")})
        for c, v in S.items():
            for h in (1, 3, 5, 10):
                s = ema(v, h)
                p = (s >= 0).astype(int)
                R.setdefault(f"{c}|h{h}", {})[q] = (fb(y, p), roc_auc_score(y, s), f1_score(y, p))
    rows = []
    for k, d in R.items():
        if len(d) < len(subs):
            continue
        a = np.array(list(d.values()))
        rows.append((k, a[:, 0].mean(), int((a[:, 0] >= .9).sum()), int((a[:, 0] >= .85).sum()),
                     a[:, 1].mean(), a[:, 2].mean()))
    rows.sort(key=lambda r: (-r[2], -r[1]))
    for r in rows[:20]:
        print(f"{r[0]:28s} bACC={r[1]:.3f} ge90={r[2]:2d} ge85={r[3]:2d} AUC={r[4]:.3f} F1={r[5]:.3f}")
    tag = "_".join(sys.argv[1:]) or "base"
    (HERE / f"eval_fixed_{tag}.json").write_text(json.dumps(
        {"rank": [dict(cfg=r[0], bACC=r[1], ge90=r[2], ge85=r[3], AUC=r[4], F1=r[5]) for r in rows],
         "per_subject": {k: {q: v[0] for q, v in d.items()} for k, d in R.items()}},
        ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
