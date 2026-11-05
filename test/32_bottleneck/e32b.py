"""E32b：E31b 第 2、3 组（G2: 016 013 006 058 014 037 036 008；G3: 018 048 056 021 038 059）方法叠加（oracle）。
候选：
  A 两组配方的 7 个基础组件的全部非空子集（127 个），z(子集和)；
  B 两组原配方 RA、RB 的加权混合 z(w·RA+(1-w)·RB)，w∈{0,.1,...,1}；
  C 在 A 最优子集上再加 1 个任意基础组件（cands_scores 全部 base）。
阈值：e26lib 85 种因果阈值。目标：并集 14 人 min bACC ≥ .9。
运行：.venv\\Scripts\\python.exe test/32_bottleneck/e32b.py
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
T = HERE.parent
sys.path.insert(0, str(T / "23_synthesis"))
sys.path.insert(0, str(T / "24_cluster"))
sys.path.insert(0, str(T / "26_merge2"))
from clib import load  # noqa: E402
import e26lib as E  # noqa: E402

dev = "cuda"
G2 = "016 013 006 058 014 037 036 008".split()
G3 = "018 048 056 021 038 059".split()
COMP = ["idxgdwt10", "sim4390_C0.001", "k4390_w0.6", "ghilbert_k200", "sim4390_C0.01", "b23r400_C0.1", "sim800_C0.001"]
THN = [t[0] for t in E.THR2]


def zc(s, cal):
    return (s - s[..., cal].mean(-1, keepdim=True)) / (s[..., cal].std(-1, keepdim=True) + 1e-6)


def main():
    R = load()
    base = R["meta"]["base"]
    ci = [base.index(c) for c in COMP]
    names, per = None, {}
    for q in G2 + G3:
        y, blk = R["D"][q][1], R["D"][q][2]
        cal = E.calref(q, R["D"])
        B = torch.tensor(R["SC"][q], device=dev, dtype=torch.float32)
        Cc = B[ci]
        rows, nm = [], []
        for r in range(1, 8):
            for s in itertools.combinations(range(7), r):
                rows.append(zc(Cc[list(s)].sum(0), cal))
                nm.append("A:" + "+".join(COMP[j] for j in s))
        RA = zc(zc(Cc[0] + Cc[1], cal) + zc(Cc[2] + Cc[3], cal), cal)
        RB = zc(zc(Cc[4] + Cc[5], cal) + Cc[6], cal)
        for w in np.linspace(0, 1, 11):
            rows.append(zc(w * RA + (1 - w) * RB, cal))
            nm.append(f"B:{w:.1f}RA+{1 - w:.1f}RB")
        for j in range(len(base)):
            for X, tag in ((RA, "RA"), (RB, "RB")):
                rows.append(zc(X + B[j] / 2, cal))
                nm.append(f"C:{tag}+½{base[j]}")
        M = torch.stack(rows)
        tst = torch.tensor(np.where(blk >= 4)[0], device=dev)
        yy = torch.tensor(y[blk >= 4], device=dev).bool()
        per[q] = E.metrics(M, tst, yy).cpu().numpy()  # (R,85,3)
        names = nm
    out = {}
    iA, iB = names.index("B:1.0RA+0.0RB"), names.index("B:0.0RA+1.0RB")
    for tag, mem in (("G2", G2), ("G3", G3), ("G2+G3", G2 + G3)):
        V = np.stack([per[q] for q in mem])  # (n,R,85,3)
        bb = np.nan_to_num(V[..., 0], nan=-1)
        mn = bb.min(0)
        k = np.unravel_index(np.argmax(mn + 1e-4 * bb.mean(0)), mn.shape)
        v = V[:, k[0], k[1]]
        out[tag] = dict(cfg=names[k[0]], thr=THN[k[1]], min=float(mn[k]), bacc=float(v[:, 0].mean()),
                        auc=float(np.nanmean(v[:, 1])), f1=float(v[:, 2].mean()),
                        per={q: round(float(v[i, 0]), 4) for i, q in enumerate(mem)})
        print(tag, out[tag])
    # 自检：原配方在原组
    out["selfcheck"] = dict(G2_RA_m300=float(np.stack([per[q] for q in G2])[:, iA, THN.index("m300"), 0].min()),
                            G3_RB_m1200=float(np.stack([per[q] for q in G3])[:, iB, THN.index("m1200"), 0].min()))
    print("自检（应 .9015 / .9002）", out["selfcheck"])
    V = np.stack([per[q] for q in G2 + G3])
    mn = np.nan_to_num(V[..., 0], nan=-1).min(0)
    top = np.argsort(-mn.ravel())[:10]
    out["top10_union"] = [dict(cfg=names[t // len(THN)], thr=THN[t % len(THN)], min=float(mn.ravel()[t])) for t in top]
    for r in out["top10_union"]:
        print(r)
    json.dump(out, open(HERE / "e32b.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
