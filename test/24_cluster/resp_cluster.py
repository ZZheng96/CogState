"""E24-1 上界（不可部署）：用每人测试 bACC 向量选 k 个共享配置。
U1 贪心集合覆盖：k 个 (配置,阈值)，每人按 oracle 分配到其中最好的一个；目标 = 平均 bACC 或 n≥0.9（平均 bACC 破平）。
U2 响应聚类：在测试 bACC 向量上 KMeans（GPU），每组用组内平均 bACC 最高的单配置，组分配即聚类标签。
输出 resp_cluster.json。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clib import HERE, load, agg, dev  # noqa: E402


def cover(TB, k, obj):
    X = torch.tensor(np.nan_to_num(TB, nan=0.0), device=dev)
    cur = torch.zeros(X.shape[0], device=dev)
    picks, curve = [], []
    for _ in range(k):
        nw = torch.maximum(cur[:, None], X)
        mean = nw.mean(0)
        sc = mean if obj == "mean" else (nw >= .9).float().sum(0) + mean
        c = int(sc.argmax())
        picks.append(c)
        cur = nw[:, c]
        curve.append(list(picks))
    return curve


def kmeans_t(X, k, seeds=20, it=100):
    best, bl = None, None
    for s in range(seeds):
        g = torch.Generator(device="cpu").manual_seed(s)
        C = X[torch.randperm(X.shape[0], generator=g)[:k]].clone()
        for _ in range(it):
            lab = torch.cdist(X, C).argmin(1)
            C = torch.stack([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
        inertia = float(((X - C[lab]) ** 2).sum())
        if bl is None or inertia < bl:
            bl, best = inertia, lab.cpu().numpy()
    return best


def main():
    R = load()
    ev, subs, nq = R["ev"], R["subs"], R["nq"]
    TB, TA, TF = (ev[:, m].reshape(nq, -1) for m in range(3))
    res = {}

    def vals_of(assign):
        return [(TB[i, c], TA[i, c], TF[i, c]) for i, c in enumerate(assign)]

    res["oracle"] = agg(vals_of([int(np.nanargmax(TB[i])) for i in range(nq)]))
    for obj in ("mean", "n90"):
        for k, picks in enumerate(cover(TB, 8, obj), 1):
            sub = np.nan_to_num(TB[:, picks], nan=-1)
            assign = [picks[int(sub[i].argmax())] for i in range(nq)]
            a = agg(vals_of(assign))
            a["picks"] = [f"{R['meta']['cands'][c // R['nt']]}|{R['meta']['thr'][c % R['nt']]}" for c in picks]
            a["sizes"] = [int(sum(x == p for x in assign)) for p in picks]
            res[f"U1_{obj}_k{k}"] = a
            print(f"U1 {obj:4s} k={k} bACC {a['bacc']:.3f} AUC {a['auc']:.3f} F1 {a['f1']:.3f} n90 {a['n90']}",
                  flush=True)
    X = torch.tensor(np.nan_to_num(TB, nan=0.5), device=dev)
    X = X - X.mean(1, keepdim=True)
    for k in range(1, 9):
        lab = kmeans_t(X, k)
        assign = []
        for i in range(nq):
            g = np.where(lab == lab[i])[0]
            mu = np.nanmean(TB[g], 0)
            mu = np.where(np.isnan(TB[i]), -1, mu)
            assign.append(int(np.nanargmax(mu)))
        a = agg(vals_of(assign))
        a["sizes"] = sorted([int((lab == j).sum()) for j in range(k)], reverse=True)
        res[f"U2_kmeans_k{k}"] = a
        print(f"U2 kmeans k={k} bACC {a['bacc']:.3f} AUC {a['auc']:.3f} F1 {a['f1']:.3f} n90 {a['n90']}"
              f" sizes {a['sizes']}", flush=True)
    json.dump(res, open(HERE / "resp_cluster.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
