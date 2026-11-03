"""E19：提高校准→测试的跨块可分性（诊断各候选族的测试段 AUC/bACC）。

协议同 E18：块 0–3 校准（带标签，学习阶段），块 4–5 测试（单窗 6 s 因果推理，阈值 0 来自校准统计量）。
候选族：
  seed            跨被试 LR（5 折，不含本人）
  own             本人校准块 LR
  joint{w}        其他被试全部帧 + 本人校准帧（样本权重 w）联合 LR
  cons{k}         本人校准块内两对 (0,1)/(2,3) 效应量同号、min|d| 前 k 维特征上的 LR
  jcons{k}        joint 只在 cons 特征上
运行：.venv\\Scripts\\python.exe test/19_xblock/xblock.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "17_compliant"))
sys.path.insert(0, str(HERE.parent / "18_calib"))
import loso as L  # noqa: E402
from calib2 import WS  # noqa: E402

dev = L.dev


def fit_wlr(X, y, sw, C, iters=150):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    w = torch.zeros(X.shape[1], device=dev, requires_grad=True)
    b = torch.zeros(1, device=dev, requires_grad=True)
    yt = torch.tensor(y, device=dev, dtype=torch.float32)
    swt = torch.tensor(sw, device=dev, dtype=torch.float32)
    # 类别平衡
    for c in (0, 1):
        m = yt == c
        swt[m] = swt[m] / swt[m].sum()
    opt = torch.optim.LBFGS([w, b], max_iter=iters, line_search_fn="strong_wolfe")
    lam = 1.0 / (C * len(y))

    def closure():
        opt.zero_grad()
        l = (torch.nn.functional.binary_cross_entropy_with_logits(Z @ w + b, yt, reduction="none") * swt).sum() / 2
        l = l + 0.5 * lam * (w ** 2).sum()
        l.backward()
        return l
    opt.step(closure)
    w, b = w.detach(), b.detach()
    return lambda X: ((X - mu) / sd) @ w + b


def dval(X, y):
    a, b = X[y == 1], X[y == 0]
    return (a.mean(0) - b.mean(0)) / torch.sqrt(0.5 * (a.var(0) + b.var(0)) + 1e-8)


def z(s, ref):
    return (s - s[ref].mean()) / (s[ref].std() + 1e-6)


def main():
    subs, D = L.load()
    subs = [q for q in subs if len(np.unique(D[q][2])) == 6]
    F = None
    for w in WS:
        Fw = L.feats(D, subs, w)
        F = Fw if F is None else {q: torch.cat([F[q], Fw[q]], 1) for q in subs}
    perm = list(np.random.RandomState(0).permutation(subs))
    seed = {}
    for k in range(5):
        te = perm[k::5]
        tr = [q for q in subs if q not in te]
        f = L.fit_lr(torch.cat([F[q] for q in tr]), np.concatenate([D[q][1] for q in tr]), 1e-3)
        for q in te:
            seed[q] = f(F[q])
    Xall = torch.cat([F[q] for q in subs])
    yall = np.concatenate([D[q][1] for q in subs])
    sid = np.concatenate([[i] * len(D[q][1]) for i, q in enumerate(subs)])
    R = {}
    for i, q in enumerate(subs):
        y, blk = D[q][1], D[q][2]
        cal, tst = blk <= 1 + 2, blk >= 4
        cal_t = torch.tensor(cal, device=dev)
        X = F[q]
        S = {"seed": z(seed[q], cal_t)}
        for C in (1e-3, 1e-2):
            S[f"own_C{C}"] = z(L.fit_lr(X[cal], y[cal], C)(X), cal_t)
        p1, p2 = blk <= 1, (blk >= 2) & (blk <= 3)
        d1, d2 = dval(X[p1], y[p1]), dval(X[p2], y[p2])
        cons = torch.where(torch.sign(d1) == torch.sign(d2), torch.minimum(d1.abs(), d2.abs()), torch.zeros_like(d1))
        order = torch.argsort(cons, descending=True)
        for k in (50, 200, 800):
            c = order[:k]
            for C in (1e-3, 1e-2):
                S[f"cons{k}_C{C}"] = z(L.fit_lr(X[cal][:, c], y[cal], C)(X[:, c]), cal_t)
        oth = sid != i
        Xo, yo = Xall[torch.tensor(oth, device=dev)], yall[oth]
        for wo in (0.3, 0.6):
            Xj = torch.cat([Xo, X[cal]])
            yj = np.concatenate([yo, y[cal]])
            sw = np.concatenate([np.full(len(yo), 1 - wo), np.full(cal.sum(), wo)])
            # 分组权重：先在 fit 内按类平衡，再整体乘组权重比例
            sw = sw / np.where(np.arange(len(yj)) < len(yo), len(yo), cal.sum())
            S[f"joint{wo}"] = z(fit_wlr(Xj, yj, sw, 1e-3)(X), cal_t)
            c = order[:800]
            S[f"jcons800_{wo}"] = z(fit_wlr(Xj[:, c], yj, sw, 1e-2)(X[:, c]), cal_t)
        R[q] = {k: L.subj_metrics(y[tst], s[tst], torch.tensor(0.0, device=dev)) for k, s in S.items()}
        print(q, " ".join(f"{k}={v['auc']:.2f}/{v['bacc']:.2f}" for k, v in R[q].items()), flush=True)
    keys = list(R[subs[0]])
    tab = {k: {m: float(np.mean([R[q][k][m] for q in subs])) for m in ("bacc", "auc", "f1")} for k in keys}
    for k in keys:
        tab[k]["n90"] = int(sum(R[q][k]["bacc"] >= .9 for q in subs))
        print(f"  {k:18s} bACC {tab[k]['bacc']:.3f} AUC {tab[k]['auc']:.3f} F1 {tab[k]['f1']:.3f} n90 {tab[k]['n90']}")
    orc = {q: max(R[q][k]["bacc"] for k in keys) for q in subs}
    print("oracle over E19 families", np.mean(list(orc.values())), sum(v >= .9 for v in orc.values()))
    json.dump(dict(table=tab, per_subject=R), open(HERE / "xblock.json", "w"), indent=1)


if __name__ == "__main__":
    main()
