"""E17b：多尺度因果标准化拼接 + 跨被试 LR / MLP 集成。单窗决策，无分数平滑，测试被试无标签。

特征 = [raw(训练集 z-score), cstd(w1), cstd(w2), ...]，每个 cstd 只用测试被试过去 w 帧的无标签特征。
模型在训练被试上训练（5 折 group，按被试分），被评被试不参与训练与选参；
本脚本只扫少量固定配置，结果需用 loso.py 的嵌套口径复核后才能作为主结果。

运行：.venv\\Scripts\\python.exe test/17_compliant/multiscale.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import loso as L

HERE = Path(__file__).resolve().parent
dev = L.dev


def fit_mlp(X, y, hid=128, ep=30, wd=1e-3, drop=0.3, seed=0):
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    net = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(X.shape[1], hid), torch.nn.GELU(),
                              torch.nn.Dropout(drop), torch.nn.Linear(hid, 1)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=wd)
    yt = torch.tensor(y, device=dev, dtype=torch.float32)
    pw = (yt == 0).sum() / (yt == 1).sum()
    n = len(y)
    for _ in range(ep):
        for idx in torch.randperm(n, device=dev).split(512):
            opt.zero_grad()
            torch.nn.functional.binary_cross_entropy_with_logits(
                net(Z[idx]).squeeze(1), yt[idx], pos_weight=pw).backward()
            opt.step()
    net.eval()

    def f(X):
        with torch.no_grad():
            return net((X - mu) / sd).squeeze(1)
    return f


def main():
    subs, D = L.load()
    rng = np.random.RandomState(0)
    perm = list(rng.permutation(subs))
    outer = [perm[k::5] for k in range(5)]
    import sys
    combos = {"c120_300_600_all": (120, 300, 600, 99999)}
    if len(sys.argv) > 1:  # 例：c30_60_120_300_600_all
        combos = {a: tuple(99999 if t == "all" else int(t) for t in a[1:].split("_")) for a in sys.argv[1:]}
    tag = "_".join(sys.argv[1:2]) or "default"
    FW = {w: L.feats(D, subs, w) for w in sorted({w for ws in combos.values() for w in ws})}
    res = {}
    for name, ws in combos.items():
        F = {q: torch.cat([FW[w][q] for w in ws], 1) for q in subs}
        for mname in ("lr", "mlp", "ens"):
            res[f"{name}_{mname}"] = {}
        for te in outer:
            tr = [q for q in subs if q not in te]
            X = torch.cat([F[q] for q in tr])
            y = np.concatenate([D[q][1] for q in tr])
            flr = L.fit_lr(X, y, 0.001)
            fm = [fit_mlp(X, y, seed=s) for s in range(3)]
            for q in te:
                yq = D[q][1]
                s_lr = flr(F[q])
                s_m = torch.stack([f(F[q]) for f in fm]).mean(0)
                s_e = s_lr / s_lr.std() + s_m / s_m.std()
                z = torch.zeros_like(s_lr)
                res[f"{name}_lr"][q] = L.subj_metrics(yq, s_lr, z)
                res[f"{name}_mlp"][q] = L.subj_metrics(yq, s_m, z)
                res[f"{name}_ens"][q] = L.subj_metrics(yq, s_e, z)
        for mname in ("lr", "mlp", "ens"):
            r = res[f"{name}_{mname}"]
            m = {k: np.mean([v[k] for v in r.values()]) for k in ("bacc", "auc", "f1")}
            n90 = sum(v["bacc"] >= .9 for v in r.values())
            print(f"{name:18s} {mname:4s} bACC {m['bacc']:.3f} AUC {m['auc']:.3f} F1 {m['f1']:.3f} n>=.9 {n90}",
                  flush=True)
    (HERE / f"multiscale_{tag}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
