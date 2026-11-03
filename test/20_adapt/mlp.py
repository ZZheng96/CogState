"""E20b：联合训练小 MLP（GPU），与 E19 联合 LR 同协议。

学习阶段：其他 55 人全部帧（组权重 1−WO）+ 本人校准块 0–3（组权重 WO），组内类别平衡；
          特征 = 本人校准块一致性特征前 K 维（同 E19）。
推理阶段：单窗 6 s 因果，分数按校准段统计量 z 化，阈值 0。
输出 mlp / lr / mlp+lr 平均 三种分数的指标。
运行：.venv\\Scripts\\python.exe test/20_adapt/mlp.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "19_xblock"))
from xblock import L, WS, dev, fit_wlr, dval, z  # noqa: E402

K = 1600
WO = 0.6
H = 128
EPOCHS = 6
BS = 4096
WD = 1e-3
DROP = 0.3


def group_weights(y, own_mask):
    sw = np.zeros(len(y))
    for g, wg in ((own_mask, WO), (~own_mask, 1 - WO)):
        for c in (0, 1):
            m = g & (y == c)
            sw[m] = wg / 2 / m.sum()
    return sw / sw.sum()


def fit_mlp(X, y, sw, seed=0):
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    net = nn.Sequential(nn.Dropout(0.1), nn.Linear(X.shape[1], H), nn.GELU(), nn.Dropout(DROP), nn.Linear(H, 1)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=WD)
    yt = torch.tensor(y, device=dev, dtype=torch.float32)
    p = torch.tensor(sw, device=dev, dtype=torch.float32)
    n = len(y)
    steps = EPOCHS * n // BS
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    net.train()
    for _ in range(steps):
        idx = torch.multinomial(p, BS, replacement=True)  # 按样本权重采样
        l = nn.functional.binary_cross_entropy_with_logits(net(Z[idx]).squeeze(1), yt[idx])
        opt.zero_grad()
        l.backward()
        opt.step()
        sched.step()
    net.eval()

    def f(Xn):
        with torch.no_grad():
            return net((Xn - mu) / sd).squeeze(1)
    return f


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
    thr0 = torch.tensor(0.0, device=dev)
    for i, q in enumerate(subs):
        y, blk = D[q][1], D[q][2]
        cal, tst = blk <= 3, blk >= 4
        cal_t = torch.tensor(cal, device=dev)
        X = F[q]
        p1, p2 = blk <= 1, (blk >= 2) & (blk <= 3)
        d1, d2 = dval(X[p1], y[p1]), dval(X[p2], y[p2])
        cons = torch.where(torch.sign(d1) == torch.sign(d2), torch.minimum(d1.abs(), d2.abs()), torch.zeros_like(d1))
        c = torch.argsort(cons, descending=True)[:K]
        oth = sid != i
        Xb = torch.cat([Xall[torch.tensor(oth, device=dev)][:, c], X[cal][:, c]])
        yb = np.concatenate([yall[oth], y[cal]])
        own = np.r_[np.zeros(oth.sum(), bool), np.ones(cal.sum(), bool)]
        sw = group_weights(yb, own)
        s_lr = z(fit_wlr(Xb, yb, sw, 1e-2)(X[:, c]), cal_t)
        s_mlp = sum(z(fit_mlp(Xb, yb, sw, sd)(X[:, c]), cal_t) for sd in range(2)) / 2
        S = {"lr": s_lr, "mlp": z(s_mlp, cal_t), "avg": z(s_lr + s_mlp, cal_t)}
        R[q] = {k: L.subj_metrics(y[tst], s[tst], thr0) for k, s in S.items()}
        print(q, " ".join(f"{k}={v['auc']:.2f}/{v['bacc']:.2f}" for k, v in R[q].items()), flush=True)
    keys = list(R[subs[0]])
    tab = {k: {m: float(np.mean([R[q][k][m] for q in subs])) for m in ("bacc", "auc", "f1")} for k in keys}
    for k in keys:
        tab[k]["n90"] = int(sum(R[q][k]["bacc"] >= .9 for q in subs))
        print(f"  {k:6s} bACC {tab[k]['bacc']:.3f} AUC {tab[k]['auc']:.3f} F1 {tab[k]['f1']:.3f} n90 {tab[k]['n90']}")
    json.dump(dict(table=tab, per_subject=R), open(HERE / "mlp.json", "w"), indent=1)


if __name__ == "__main__":
    main()
