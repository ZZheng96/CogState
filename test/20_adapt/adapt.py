"""E20：E19 联合模型 + 推理期因果伪标签自训练。

学习阶段：其他 55 人全部帧 + 本人校准块 0–3（带标签），一致性特征前 k 维（同 E19）。
推理阶段：测试段（块 4–5）按时间每 STEP 帧重训一次。时刻 c 的模型只用 c 之前的测试帧，
          伪标签 = 这些帧当时（因果）得到的分数符号，只取 |z|>TAU 的帧，组权重 wp。
          frame t 的分数来自 t 之前最后一次重训的模型，每次判断仍只读当前 6 s 窗特征。
运行：.venv\\Scripts\\python.exe test/20_adapt/adapt.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "19_xblock"))
from xblock import L, WS, dev, fit_wlr, dval, z  # noqa: E402

KS = [800, 1600]
WO = 0.6
WPS = [0.0, 0.15, 0.3]
TAU = 0.5
STEP = 150
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
    R = {}
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
        ti = np.where(tst)[0]
        cps = list(range(0, len(ti), STEP))
        R[q] = {}
        S_ens = {wp: torch.zeros(len(y), device=dev) for wp in WPS}
        for k in KS:
            c = order[:k]
            Xb = torch.cat([Xo[:, c], X[cal][:, c]])
            yb = np.concatenate([yo, y[cal]])
            swb = np.concatenate([np.full(len(yo), (1 - WO) / len(yo)), np.full(cal.sum(), WO / cal.sum())])
            base = z(fit_wlr(Xb, yb, swb, C)(X[:, c]), cal_t)
            for wp in WPS:
                s = base.clone()
                if wp > 0:
                    for cp in cps[1:]:
                        past = ti[:cp]
                        sp = s[past]
                        m = (sp.abs() > TAU).cpu().numpy()
                        pi = past[m]
                        if len(pi) < 20:
                            continue
                        pl = (sp[torch.tensor(m, device=dev)] > 0).long().cpu().numpy()
                        f = fit_wlr(torch.cat([Xb, X[pi][:, c]]), np.concatenate([yb, pl]),
                                    np.concatenate([swb * (1 - wp), np.full(len(pi), wp / len(pi))]), C)
                        s_new = z(f(X[:, c]), cal_t)
                        fut = ti[cp:cp + STEP]
                        s[fut] = s_new[fut]
                key = f"k{k}_wp{wp}"
                R[q][key] = L.subj_metrics(y[tst], s[tst], torch.tensor(0.0, device=dev))
                S_ens[wp] += s / len(KS)
        for wp in WPS:
            R[q][f"ens_wp{wp}"] = L.subj_metrics(y[tst], S_ens[wp][tst], torch.tensor(0.0, device=dev))
        print(q, " ".join(f"{k}={v['bacc']:.2f}" for k, v in R[q].items()), flush=True)
    keys = list(R[subs[0]])
    tab = {k: {m: float(np.mean([R[q][k][m] for q in subs])) for m in ("bacc", "auc", "f1")} for k in keys}
    for k in keys:
        tab[k]["n90"] = int(sum(R[q][k]["bacc"] >= .9 for q in subs))
        print(f"  {k:14s} bACC {tab[k]['bacc']:.3f} AUC {tab[k]['auc']:.3f} F1 {tab[k]['f1']:.3f} n90 {tab[k]['n90']}")
    json.dump(dict(table=tab, per_subject=R), open(HERE / "adapt.json", "w"), indent=1)


if __name__ == "__main__":
    main()
