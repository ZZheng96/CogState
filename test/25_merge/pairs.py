"""E25 第 1 步扩展：在 cands 池（1653 配置）之上，生成"配置两两 z 平均"新配方并在 57 人上评估（GPU）。
池子：每人 bACC（任一阈值）≥0.85 的配置并集太大（=1653），故取 每人前 5 + 按 n(≥0.85) 排名的全局前列，合计 NP 个，
生成 C(NP,2) 对，每对 × 8 阈值 = 新配方。z 平均后按本人校准段统计量再标准化（与 E23/E24 一致）。
输出 pairs_eval.npz：ev (57, 3[bACC,AUC,F1], npair, 8)、pool (NP,) 配置索引、ia/ib。
运行：.venv\\Scripts\\python.exe test/25_merge/pairs.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "24_cluster"))
from clib import load, THR, dev, calref  # noqa: E402
from general import score_of  # noqa: E402

NP = 220
CH = 192


def pick_pool(ev):
    m = np.nan_to_num(ev[:, 0], nan=0).max(2)  # (57, ncfg)
    pool = []
    for i in range(m.shape[0]):
        pool += list(np.argsort(-m[i])[:5])
    cnt = (m >= .85).sum(0) + 1e-3 * m.mean(0)
    for c in np.argsort(-cnt):
        if len(set(pool)) >= NP:
            break
        pool.append(c)
    return np.array(sorted(set(int(c) for c in pool)))


def main():
    R = load()
    ev = R["ev"]
    pool = pick_pool(ev)
    ia, ib = map(np.array, zip(*itertools.combinations(range(len(pool)), 2)))
    out = np.full((R["nq"], 3, len(ia), len(THR)), np.nan, np.float32)
    for j, q in enumerate(R["subs"]):
        y, blk = R["D"][q][1], R["D"][q][2]
        cal_t = calref(q, R["D"])
        tst = torch.tensor(np.where(blk >= 4)[0], device=dev)
        yy = torch.tensor(y[blk >= 4], device=dev).bool()
        npos, nneg = yy.sum().float(), (~yy).sum().float()
        B = torch.stack([score_of(R["SC"][q], int(c), R["nb"], cal_t) for c in pool])  # (NP,T)
        ta, tb = torch.tensor(ia, device=dev), torch.tensor(ib, device=dev)
        for a0 in range(0, len(ia), CH):
            M = B[ta[a0:a0 + CH]] + B[tb[a0:a0 + CH]]
            M = (M - M[:, cal_t].mean(1, keepdim=True)) / (M[:, cal_t].std(1, keepdim=True) + 1e-6)
            St = M[:, tst]
            nanrow = torch.isnan(M).any(1)
            for ti, (_, w, qq) in enumerate(THR):
                if w == 0:
                    Th = torch.zeros_like(St)
                else:
                    pad = torch.cat([torch.full((M.shape[0], w - 1), float("nan"), device=dev), M], 1)
                    Th = pad.unfold(1, w, 1)[:, tst].nanquantile(qq, 2)
                P = St > Th
                tp, fp = (P & yy).sum(1).float(), (P & ~yy).sum(1).float()
                r = (St - Th).argsort(1).argsort(1).float() + 1
                res = torch.stack([0.5 * (tp / npos + 1 - fp / nneg),
                                   (r[:, yy].sum(1) - npos * (npos + 1) / 2) / (npos * nneg),
                                   2 * tp / (2 * tp + fp + (npos - tp))])
                res[:, nanrow] = float("nan")
                out[j, :, a0:a0 + CH, ti] = res.cpu().numpy()
        b = np.nan_to_num(out[j, 0], nan=0)
        print(q, f"pair-oracle {b.max():.3f} n>=.9 {(b.max(1) >= .9).sum()}", flush=True)
        del B
        torch.cuda.empty_cache()
    np.savez_compressed(HERE / "pairs_eval.npz", ev=out, pool=pool, ia=ia, ib=ib)


if __name__ == "__main__":
    main()
