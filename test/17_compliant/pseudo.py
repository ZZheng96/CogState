"""E17c：测试被试无标签伪标签自适应（先预测后更新）。

1. 种子：跨被试 LR+MLP 集成（多尺度因果标准化特征，训练被试标签）。
2. 测试被试上，每 R 帧用"已到达帧"的种子分数做伪标签（取 |z| 最大的 top 比例两端），
   拟合本人岭回归 LR；之后 R 帧的分数 = 种子 z + α·本人 z。当前帧的分数只依赖已更新的模型和当前窗。
3. 无分数平滑，决策阈值 0。

运行：.venv\\Scripts\\python.exe test/17_compliant/pseudo.py [c60_120_300_600_all]
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

import loso as L
from multiscale import fit_mlp

HERE = Path(__file__).resolve().parent
dev = L.dev


def ridge_dir(X, p, lam):
    """带符号伪标签 p∈{-1,+1} 的岭回归方向（特征已 z-score）。"""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    A = Z.T @ Z + lam * len(p) * torch.eye(Z.shape[1], device=dev)
    w = torch.linalg.solve(A, Z.T @ (p - p.mean()))
    return mu, sd, w


def adapt(F, seed, R, top, lam, alphas, warm=120):
    """返回 {alpha: 分数}。seed 已按被试内 z 化前的原始分数。"""
    n = len(seed)
    out = {a: seed.clone() for a in alphas}
    for t0 in range(warm, n, R):
        s_past = seed[:t0]
        k = max(10, int(top * t0))
        hi = torch.topk(s_past, k).indices
        lo = torch.topk(-s_past, k).indices
        idx = torch.cat([hi, lo])
        p = torch.cat([torch.ones(k, device=dev), -torch.ones(k, device=dev)])
        mu, sd, w = ridge_dir(F[idx], p, lam)
        own_past = ((F[:t0] - mu) / sd) @ w
        own = ((F[t0:t0 + R] - mu) / sd) @ w
        own = (own - own_past.median()) / (own_past.std() + 1e-6)
        sp = seed[t0:t0 + R] / (s_past.std() + 1e-6)
        for a in alphas:
            out[a][t0:t0 + R] = sp + a * own
    return out


def main():
    combo = sys.argv[1] if len(sys.argv) > 1 else "c60_120_300_600_all"
    ws = tuple(99999 if t == "all" else int(t) for t in combo[1:].split("_"))
    subs, D = L.load()
    perm = list(np.random.RandomState(0).permutation(subs))
    outer = [perm[k::5] for k in range(5)]
    F = None
    for w in ws:
        Fw = L.feats(D, subs, w)
        F = Fw if F is None else {q: torch.cat([F[q], Fw[q]], 1) for q in subs}
        del Fw
        torch.cuda.empty_cache()
    seeds = {}
    for te in outer:
        tr = [q for q in subs if q not in te]
        X = torch.cat([F[q] for q in tr])
        y = np.concatenate([D[q][1] for q in tr])
        flr = L.fit_lr(X, y, 0.001)
        fm = [fit_mlp(X, y, seed=s) for s in range(3)]
        del X
        torch.cuda.empty_cache()
        for q in te:
            a, m = flr(F[q]), torch.stack([f(F[q]) for f in fm]).mean(0)
            seeds[q] = a / a.std() + m / m.std()
    # 适应用较低维特征（只用 w=300 一档，减少噪声维），也试全部
    d0 = 878
    pos300 = ws.index(300) if 300 in ws else 0
    sub_feat = {"c300": lambda q: F[q][:, pos300 * d0:(pos300 + 1) * d0], "all": lambda q: F[q]}
    ALPH = (0.0, 0.5, 1.0, 2.0)
    res = {}
    for (fk, ff), R, top, lam in itertools.product(sub_feat.items(), (30, 60), (0.2, 0.35), (1.0, 10.0)):
        for q in subs:
            out = adapt(ff(q), seeds[q], R, top, lam, ALPH)
            for a in ALPH:
                res.setdefault(f"{fk}_R{R}_top{top}_lam{lam}_a{a}", {})[q] = L.subj_metrics(
                    D[q][1], out[a], torch.zeros_like(out[a]))
        best = None
        for a in ALPH:
            r = res[f"{fk}_R{R}_top{top}_lam{lam}_a{a}"]
            m = {k: np.mean([v[k] for v in r.values()]) for k in ("bacc", "auc", "f1")}
            n90 = sum(v["bacc"] >= .9 for v in r.values())
            print(f"{fk:4s} R{R} top{top} lam{lam} a{a}: bACC {m['bacc']:.3f} AUC {m['auc']:.3f} "
                  f"F1 {m['f1']:.3f} n>=.9 {n90}", flush=True)
    # 嵌套选参：每个外层折的配置只在其余折（训练被试）上按宏 bACC 选
    nested = {}
    for te in outer:
        tr = [q for q in subs if q not in te]
        best = max(res, key=lambda k: np.mean([res[k][q]["bacc"] for q in tr]))
        for q in te:
            nested[q] = dict(res[best][q], cfg=best)
    m = {k: np.mean([v[k] for v in nested.values()]) for k in ("bacc", "auc", "f1")}
    print(f"NESTED: bACC {m['bacc']:.3f} AUC {m['auc']:.3f} F1 {m['f1']:.3f} "
          f"n>=.9 {sum(v['bacc'] >= .9 for v in nested.values())} "
          f"cfgs {sorted(set(v['cfg'] for v in nested.values()))}", flush=True)
    res["_nested"] = nested
    (HERE / f"pseudo_{combo}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
