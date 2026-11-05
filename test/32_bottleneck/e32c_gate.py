"""E32c-gate：寻找合规门控统计量，把 14 人分到 RA@m300（G2）或 RB@m1200（G3）。
候选统计量（只用校准块标签或测试流无标签数据）：
  校准块 AUC / bACC（RA vs RB，块 0–3 全体、块 2–3）；校准块 RA 与 RB 的相关；
  测试流前 N 帧（无标签）RA 与 RB 的相关、RA/RB 的 std。
运行：.venv\\Scripts\\python.exe test/32_bottleneck/e32c_gate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
T = HERE.parent
for p in ("23_synthesis", "24_cluster", "26_merge2"):
    sys.path.insert(0, str(T / p))
from clib import load  # noqa: E402
import e26lib as E  # noqa: E402

dev = "cuda"
G2 = "016 013 006 058 014 037 036 008".split()
G3 = "018 048 056 021 038 059".split()
COMP = ["idxgdwt10", "sim4390_C0.001", "k4390_w0.6", "ghilbert_k200", "sim4390_C0.01", "b23r400_C0.1", "sim800_C0.001"]


def zc(s, cal):
    return (s - s[..., cal].mean(-1, keepdim=True)) / (s[..., cal].std(-1, keepdim=True) + 1e-6)


def auc(s, y):
    r = s.argsort().argsort().float() + 1
    npos, nneg = y.sum().float(), (~y).sum().float()
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-9))


def main():
    R = load()
    base = R["meta"]["base"]
    ci = [base.index(c) for c in COMP]
    F = {}
    for q in G2 + G3:
        y, blk = R["D"][q][1], R["D"][q][2]
        cal = E.calref(q, R["D"])
        C = torch.tensor(R["SC"][q], device=dev, dtype=torch.float32)[ci]
        RA = zc(zc(C[0] + C[1], cal) + zc(C[2] + C[3], cal), cal)
        RB = zc(zc(C[4] + C[5], cal) + C[6], cal)
        yt = torch.tensor(y, device=dev).bool()
        bk = torch.tensor(blk, device=dev)
        f = {}
        for nm, m in (("cal", cal), ("b23", (bk >= 2) & (bk <= 3) & cal), ("b3", (bk == 3))):
            if m.sum() == 0 or yt[m].all() or (~yt[m]).all():
                continue
            f[f"dAUC_{nm}"] = auc(RA[m], yt[m]) - auc(RB[m], yt[m])
        f["corr_cal"] = corr(RA[cal], RB[cal])
        ti = torch.where(bk >= 4)[0]
        for N in (300, 600, 1200):
            s = ti[:N]
            f[f"corr_t{N}"] = corr(RA[s], RB[s])
            f[f"sdA_t{N}"] = float(RA[s].std())
            f[f"sdB_t{N}"] = float(RB[s].std())
            f[f"dsd_t{N}"] = float(RA[s].std() - RB[s].std())
        F[q] = f
    keys = sorted(set.intersection(*(set(v) for v in F.values())))
    res = {}
    for k in keys:
        a = np.array([F[q][k] for q in G2])
        b = np.array([F[q][k] for q in G3])
        # 单阈值可分性：G2 全在一侧
        sep = max(a.min() - b.max(), b.min() - a.max())
        res[k] = dict(G2=[round(x, 4) for x in a], G3=[round(x, 4) for x in b], margin=round(float(sep), 4))
        print(f"{k:12s} margin={sep:+.4f}  G2[{a.min():+.3f},{a.max():+.3f}] G3[{b.min():+.3f},{b.max():+.3f}]")
    json.dump(dict(feat=F, sep=res), open(HERE / "e32c_gate.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
