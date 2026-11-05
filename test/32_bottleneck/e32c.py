"""E32c：G2∪G3 合并——margin 融合（各分数先减各自因果阈值，再加权求和，以 0 判决）。
dA = sA - thrA(a)，dB = sB - thrB(b)，D = w·dA/σA + (1-w)·dB/σB，σ 为 1（不缩放）或过去 S 帧因果 std。
sA ∈ {RA 原配方, RA'(E32b 简化)}，sB = RB；a,b 遍历 85 种因果阈值，w∈{0,.05,...,1}。
附：个体门控 oracle（每人取 RA@m300 / RB@m1200 较优者）作参照。
运行：.venv\\Scripts\\python.exe test/32_bottleneck/e32c.py
"""
from __future__ import annotations

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
MEM = G2 + G3
COMP = ["idxgdwt10", "sim4390_C0.001", "k4390_w0.6", "ghilbert_k200", "sim4390_C0.01", "b23r400_C0.1", "sim800_C0.001"]
THN = [t[0] for t in E.THR2]
WS = np.round(np.linspace(0, 1, 21), 2)
SIG = [0, 300, 1200]  # 0 = 不缩放


def zc(s, cal):
    return (s - s[..., cal].mean(-1, keepdim=True)) / (s[..., cal].std(-1, keepdim=True) + 1e-6)


def cstd(s, tst, w):
    """过去 w 帧（含当前）因果 std，取 tst 位置。s (T,)"""
    cs = torch.nn.functional.pad(s.cumsum(0), (1, 0))
    c2 = torch.nn.functional.pad((s * s).cumsum(0), (1, 0))
    hi = tst + 1
    lo = (hi - w).clamp(min=0)
    n = (hi - lo).float()
    m = (cs[hi] - cs[lo]) / n
    return ((c2[hi] - c2[lo]) / n - m * m).clamp(min=1e-6).sqrt()


def auc(d, yy):
    r = d.argsort().argsort().float() + 1
    npos, nneg = yy.sum().float(), (~yy).sum().float()
    return float((r[yy].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main():
    R = load()
    base = R["meta"]["base"]
    ci = [base.index(c) for c in COMP]
    nA, nT, nW = 2, len(THN), len(WS)
    BA = {}  # q -> (nA, nS, 85, 85, nW) bACC
    F1 = {}
    keep = {}
    for q in MEM:
        y, blk = R["D"][q][1], R["D"][q][2]
        cal = E.calref(q, R["D"])
        B = torch.tensor(R["SC"][q], device=dev, dtype=torch.float32)
        C = B[ci]
        RA = zc(zc(C[0] + C[1], cal) + zc(C[2] + C[3], cal), cal)
        RA2 = zc(C[0] + C[1] + C[3], cal)
        RB = zc(zc(C[4] + C[5], cal) + C[6], cal)
        tst = torch.tensor(np.where(blk >= 4)[0], device=dev)
        yy = torch.tensor(y[blk >= 4], device=dev).bool()
        npos, nneg = yy.sum().float(), (~yy).sum().float()
        thB = E.thr_bank(RB[None], tst)[0]  # (85,nt)
        dB0 = RB[tst][None] - thB
        wv = torch.tensor(WS, device=dev, dtype=torch.float32)[:, None, None]  # (nW,1,1)
        ba = torch.empty(nA, len(SIG), nT, nT, nW)
        f1 = torch.empty_like(ba)
        for ia, sA in enumerate((RA, RA2)):
            thA = E.thr_bank(sA[None], tst)[0]
            dA0 = sA[tst][None] - thA
            for js, S in enumerate(SIG):
                dA = dA0 / cstd(sA, tst, S) if S else dA0
                dB = dB0 / cstd(RB, tst, S) if S else dB0
                for a in range(nT):
                    D = wv * dA[a][None, None] + (1 - wv) * dB[None]  # (nW,85,nt)
                    P = D > 0
                    tp, fp = (P & yy).sum(-1).float(), (P & ~yy).sum(-1).float()
                    ba[ia, js, a] = (0.5 * (tp / npos + 1 - fp / nneg)).T.cpu()
                    f1[ia, js, a] = (2 * tp / (2 * tp + fp + (npos - tp))).T.cpu()
        BA[q], F1[q] = ba.numpy(), f1.numpy()
        keep[q] = (RA, RA2, RB, tst, yy)
        print(q, "done", flush=True)

    out = {}
    for tag, mem in (("G2", G2), ("G3", G3), ("G2+G3", MEM)):
        V = np.stack([BA[q] for q in mem])
        mn, mu = V.min(0), V.mean(0)
        k = np.unravel_index(np.argmax(mn + 1e-4 * mu), mn.shape)
        ia, js, a, b, iw = k
        RA_, RA2_, RB_, _, _ = keep[mem[0]]
        aucs = []
        for q in mem:
            RA, RA2, RB, tst, yy = keep[q]
            sA = (RA, RA2)[ia]
            dA = sA[tst] - E.thr_bank(sA[None], tst)[0][a]
            dB = RB[tst] - E.thr_bank(RB[None], tst)[0][b]
            if SIG[js]:
                dA, dB = dA / cstd(sA, tst, SIG[js]), dB / cstd(RB, tst, SIG[js])
            aucs.append(auc(WS[iw] * dA + (1 - WS[iw]) * dB, yy))
        out[tag] = dict(sA=("RA", "RA'")[ia], sigma=SIG[js], thrA=THN[a], thrB=THN[b], w=float(WS[iw]),
                        min=float(mn[k]), bacc=float(V[(slice(None),) + k].mean()), auc=float(np.mean(aucs)),
                        f1=float(np.mean([F1[q][k] for q in mem])),
                        per={q: round(float(BA[q][k]), 4) for q in mem})
        print(tag, out[tag], flush=True)
    # 自检：w=1 → RA@m300（G2 应 .9015）；w=0 → RB@m1200（G3 应 .9002）
    i1, i0 = len(WS) - 1, 0
    out["selfcheck"] = dict(
        G2_RA_m300=float(min(BA[q][0, 0, THN.index("m300"), 0, i1] for q in G2)),
        G3_RB_m1200=float(min(BA[q][0, 0, 0, THN.index("m1200"), i0] for q in G3)))
    print("自检", out["selfcheck"])
    # 门控 oracle：每人取 RA@m300 与 RB@m1200 的较优者
    gA = {q: float(BA[q][0, 0, THN.index("m300"), 0, i1]) for q in MEM}
    gB = {q: float(BA[q][0, 0, 0, THN.index("m1200"), i0]) for q in MEM}
    out["gate"] = dict(RA_m300=gA, RB_m1200=gB, oracle_min=float(min(max(gA[q], gB[q]) for q in MEM)))
    print("门控", out["gate"])
    V = np.stack([BA[q] for q in MEM]).min(0)
    top = np.argsort(-V.ravel())[:10]
    out["top10_union"] = []
    for t in top:
        ia, js, a, b, iw = np.unravel_index(t, V.shape)
        out["top10_union"].append(dict(sA=("RA", "RA'")[ia], sigma=SIG[js], thrA=THN[a], thrB=THN[b],
                                       w=float(WS[iw]), min=float(V.ravel()[t])))
        print(out["top10_union"][-1])
    json.dump(out, open(HERE / "e32c.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
