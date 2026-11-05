"""E28 融合：E24（ax5 Ward k=3 top1，按 LOSO 重建逐人分数）+ E27（cat|kcv+own）+ E28 候选行 z 平均。
权重网格 w∈{0,.25,.5,.75,1}³（和>0，归一化），E28 行取「其余 56 人 LOSO bACC 前 K 名」之一，
(E28 行, 权重, 阈值) 由其余 56 人平均测试 bACC 选（严格 LOSO）；阈值为 e26lib 85 种因果阈值。
输出 fuse.json。运行：.venv\\Scripts\\python.exe test/28_block_invariant/fuse.py
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
T = HERE.parent
sys.path.insert(0, str(T / "24_cluster"))
from clib import load as load24, score_of, calref, dev  # noqa: E402


def _imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


E = _imp("e26lib", T / "26_merge2" / "e26lib.py")
KTOP = 5
GRID = [w for w in itertools.product((0, .25, .5, .75, 1), repeat=3) if sum(w) > 0]


def summ(a):
    return dict(bacc=float(a[:, 0].mean()), auc=float(a[:, 1].mean()), f1=float(a[:, 2].mean()),
                n90=int((a[:, 0] >= .9).sum()), min=float(a[:, 0].min()))


def main():
    R = load24()
    subs, nq, nt, nb = R["subs"], R["nq"], R["nt"], R["nb"]
    TB = R["ev"][:, 0].reshape(nq, -1)
    lab = np.array(json.load(open(T / "24_cluster" / "prof_groups.json"))["ax5_ward_k3"])
    S28 = np.load(HERE / "e28_scores.npz")
    names = list(S28["names"])
    E28 = np.load(HERE / "e28_ev.npz")
    e28_subs = E28["subs"].tolist()
    assert set(e28_subs) == set(subs)
    order = [e28_subs.index(q) for q in subs]
    EV28 = E28["EV"][:, order]  # (nr,nq,85,3)，统一为 E23/E24 的被试顺序
    i27 = names.index("e27|kcv+own")
    # E24 逐人分数（同 prof_cluster.top1）
    s24, chk = {}, []
    for i, q in enumerate(subs):
        g = [j for j in np.where(lab == lab[i])[0] if j != i] or [j for j in range(nq) if j != i]
        mu = np.where(np.isnan(TB[i]), -np.inf, np.nanmean(TB[g], 0))
        c = int(np.argmax(mu))
        s24[q] = score_of(R["SC"][q], c // nt, nb, calref(q, R["D"]))
        chk.append(TB[i, c])
    print("E24 重建 bACC(8 阈值) =", round(float(np.mean(chk)), 4), flush=True)
    # 每人：候选 E28 行 = 全部行（选择时只看其余人）；为省算力先用其余人 LOSO 均值取前 KTOP（逐人）
    A28 = np.nanmax(EV28[..., 0], 2)  # (nr,nq) 粗筛用，后面逐人排除本人
    # 逐人计算全部 (候选行∈并集, 权重, 阈值) 的指标
    cand_sets = []
    for i in range(nq):
        m = np.nanmean(np.delete(A28, i, 1), 1)
        m[i27] = -np.inf
        cand_sets.append(set(np.argsort(-m)[:KTOP].tolist()))
    U = sorted(set().union(*cand_sets))
    print("候选 E28 行:", [names[r] for r in U], flush=True)
    keys = [(r, w) for r in U for w in GRID]
    M_all = np.zeros((nq, len(keys), 85, 3), np.float32)
    for i, q in enumerate(subs):
        y, blk = R["D"][q][1], R["D"][q][2]
        cal = calref(q, R["D"])
        tst = torch.tensor(np.where(blk >= 4)[0], device=dev)
        yy = torch.tensor(y[blk >= 4], device=dev).bool()
        Sq = torch.tensor(S28[f"s{q}"].astype(np.float32), device=dev)
        z = lambda s: (s - s[cal].mean()) / (s[cal].std() + 1e-6)  # noqa: E731
        a24, a27 = z(s24[q].float()), z(Sq[i27])
        rows = []
        for r, (w1, w2, w3) in keys:
            rows.append(z(w1 * a24 + w2 * a27 + w3 * z(Sq[r])))
        M_all[i] = E.metrics(torch.stack(rows), tst, yy).cpu().numpy()
    # 单方法参照（85 阈值、LOSO 阈值）
    out = {}
    for nm, w in (("E24", (1, 0, 0)), ("E27", (0, 1, 0))):
        k = keys.index((U[0], w))
        V = M_all[:, k]
        res = np.zeros((nq, 3))
        for i in range(nq):
            res[i] = V[i, int(np.nanargmax(np.nanmean(np.delete(V[:, :, 0], i, 0), 0)))]
        out[nm + "@85thr"] = summ(res)
    # 融合：逐人在其余人上选 (行∈该人候选, 权重, 阈值)
    res, picks = np.zeros((nq, 3)), {}
    for i, q in enumerate(subs):
        ok = np.array([r in cand_sets[i] for r, _ in keys])
        mm = np.nanmean(np.delete(M_all[..., 0], i, 0), 0)  # (nk,85)
        mm[~ok] = -np.inf
        k, t = np.unravel_index(np.nanargmax(mm), mm.shape)
        res[i] = M_all[i, k, t]
        picks[q] = f"{names[keys[k][0]]}|w={keys[k][1]}|{E.THR2[t][0]}"
    out["fuse"] = summ(res)
    # 固定 E24+E27 (无 E28) 融合
    ok2 = np.array([w[2] == 0 for _, w in keys])
    res2 = np.zeros((nq, 3))
    for i in range(nq):
        mm = np.nanmean(np.delete(M_all[..., 0], i, 0), 0)
        mm[~ok2] = -np.inf
        k, t = np.unravel_index(np.nanargmax(mm), mm.shape)
        res2[i] = M_all[i, k, t]
    out["fuse_24+27"] = summ(res2)
    for k, v in out.items():
        print(k, {a: round(b, 4) for a, b in v.items()}, flush=True)
    from collections import Counter
    print(Counter(picks.values()).most_common(5))
    json.dump(dict(summary=out, picks=picks, per_subject={q: [round(float(x), 4) for x in res[i]]
                                                           for i, q in enumerate(subs)}),
              open(HERE / "fuse.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
