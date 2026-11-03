"""E19b：联合训练（其他被试 + 本人校准）× 一致性特征数 k × 本人权重 wo 扫描。协议同 xblock.py。
运行：.venv\\Scripts\\python.exe test/19_xblock/xblock2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from xblock import L, WS, dev, fit_wlr, dval, z  # noqa: E402

KS = [400, 800, 1600, 4390]
WOS = [0.6, 0.85]
C = 1e-2


def main():
    subs, D = L.load()
    subs = [q for q in subs if len(np.unique(D[q][2])) == 6]
    F = None
    for w in WS:
        Fw = L.feats(D, subs, w)
        F = Fw if F is None else {q: torch.cat([F[q], Fw[q]], 1) for q in subs}
    Xall = torch.cat([F[q] for q in subs])
    yall = np.concatenate([D[q][1] for q in subs])
    sid = np.concatenate([[i] * len(D[q][1]) for i, q in enumerate(subs)])
    R, SC = {}, {}
    for i, q in enumerate(subs):
        y, blk = D[q][1], D[q][2]
        cal, tst = blk <= 3, blk >= 4
        cal_t = torch.tensor(cal, device=dev)
        X = F[q]
        p1, p2 = blk <= 1, (blk >= 2) & (blk <= 3)
        d1, d2 = dval(X[p1], y[p1]), dval(X[p2], y[p2])
        cons = torch.where(torch.sign(d1) == torch.sign(d2), torch.minimum(d1.abs(), d2.abs()), torch.zeros_like(d1))
        order = torch.argsort(cons, descending=True)
        oth = sid != i
        Xo, yo = Xall[torch.tensor(oth, device=dev)], yall[oth]
        yj = np.concatenate([yo, y[cal]])
        R[q], SC[q] = {}, {}
        for wo in WOS:
            sw = np.concatenate([np.full(len(yo), (1 - wo) / len(yo)), np.full(cal.sum(), wo / cal.sum())])
            for k in KS:
                c = order[:k]
                s = z(fit_wlr(torch.cat([Xo[:, c], X[cal][:, c]]), yj, sw, C)(X[:, c]), cal_t)
                key = f"k{k}_w{wo}"
                R[q][key] = L.subj_metrics(y[tst], s[tst], torch.tensor(0.0, device=dev))
                SC[q][key] = s.cpu().numpy().tolist()
        print(q, " ".join(f"{k}={v['auc']:.2f}/{v['bacc']:.2f}" for k, v in R[q].items()), flush=True)
    keys = list(R[subs[0]])
    tab = {k: {m: float(np.mean([R[q][k][m] for q in subs])) for m in ("bacc", "auc", "f1")} for k in keys}
    for k in keys:
        tab[k]["n90"] = int(sum(R[q][k]["bacc"] >= .9 for q in subs))
        print(f"  {k:14s} bACC {tab[k]['bacc']:.3f} AUC {tab[k]['auc']:.3f} F1 {tab[k]['f1']:.3f} n90 {tab[k]['n90']}")
    json.dump(dict(table=tab, per_subject=R), open(HERE / "xblock2.json", "w"), indent=1)
    np.savez_compressed(HERE / "xblock2_scores.npz", **{f"{q}|{k}": np.array(v, np.float32)
                                                       for q in SC for k, v in SC[q].items()})


if __name__ == "__main__":
    main()
