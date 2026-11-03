"""E25 第 2 步：自底向上两两合并被试，全程组内最小 bACC ≥ 0.9。
配方池 = cands（1653 配置 × 8 阈值）+ 两两 z 平均配方（pairs_eval.npz）+ 7 人专属配方（excl.json）。
主算法：逐轮最大权匹配（networkx max_weight_matching，maxcardinality），边权 = 合并后组内最小 bACC 的最优值（≥0.9 才有边）。
变体：HAC（每步只合并最优一对）。可部署性：LOSO，在其余 56 人上重跑合并，留出者按 ax5 画像分到最近组。
输出 merge.json / merge.log。运行：.venv\\Scripts\\python.exe test/25_merge/merge.py
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "24_cluster"))
from clib import load, prof_matrix, AX5, dev  # noqa: E402

TH = 0.9


def cfg_name(c, base):
    nb = len(base)
    if c < nb:
        return base[c]
    a, b = list(itertools.combinations(range(nb), 2))[c - nb]
    return f"{base[a]}+{base[b]}"


def build_pool(R):
    meta, ev = R["meta"], R["ev"]
    base, thr = meta["base"], meta["thr"]
    nq, nc, nt = R["nq"], ev.shape[2], ev.shape[3]
    M = [ev[:, :3].reshape(nq, 3, nc * nt)]
    names = [("cands", c, t) for c in range(nc) for t in range(nt)]
    P = np.load(HERE / "pairs_eval.npz")
    pe, pool, ia, ib = P["ev"], P["pool"], P["ia"], P["ib"]
    M.append(pe.reshape(nq, 3, -1))
    names += [("pair", (int(pool[a]), int(pool[b])), t) for a, b in zip(ia, ib) for t in range(nt)]
    ex = json.load(open(HERE / "excl.json"))["per"]
    E = np.full((nq, 3, len(ex)), np.nan, np.float32)
    for k, r in enumerate(ex):
        for j, q in enumerate(R["subs"]):
            if q in ex[r]:
                E[j, :, k] = [ex[r][q]["bacc"], ex[r][q]["auc"], ex[r][q]["f1"]]
        names.append(("excl", r, None))
    M.append(E)
    M = np.concatenate(M, 2)

    def label(k):
        kind, c, t = names[k]
        if kind == "cands":
            return f"{cfg_name(c, base)} | {thr[t]}"
        if kind == "pair":
            return f"z[{cfg_name(c[0], base)} ; {cfg_name(c[1], base)}] | {thr[t]}"
        return f"专属配方({c})"

    return M, names, label, dict(n_cands=nc * nt, n_pair=pe.shape[2] * nt, n_excl=len(ex), pool_np=len(pool))


def pair_scores(Gm):
    """Gm (ng, K) 组内最小 bACC → S (ng, ng) 合并后最优最小 bACC。"""
    ng = Gm.shape[0]
    S = torch.full((ng, ng), -1.0, device=dev)
    for a in range(ng - 1):
        S[a, a + 1:] = torch.minimum(Gm[a][None], Gm[a + 1:]).max(1).values
    return S


def merge_matching(B, members):
    groups = [[m] for m in members]
    Gm = B[members].clone()
    hist = []
    while True:
        S = pair_scores(Gm).cpu().numpy()
        Gr = nx.Graph()
        for a, b in zip(*np.where(S >= TH)):
            Gr.add_edge(int(a), int(b), weight=float(S[a, b]))
        mt = nx.max_weight_matching(Gr, maxcardinality=True) if Gr.number_of_edges() else set()
        if not mt:
            break
        used, ng, nm = set(), [], []
        for a, b in mt:
            used |= {a, b}
            ng.append(groups[a] + groups[b])
            nm.append(torch.minimum(Gm[a], Gm[b]))
        for g in range(len(groups)):
            if g not in used:
                ng.append(groups[g])
                nm.append(Gm[g])
        groups, Gm = ng, torch.stack(nm)
        hist.append(snapshot(groups, Gm))
    return groups, Gm, hist


def merge_hac(B, members):
    groups = [[m] for m in members]
    Gm = B[members].clone()
    hist = []
    while True:
        S = pair_scores(Gm)
        v, k = S.flatten().max(0)
        if v.item() < TH:
            break
        a, b = divmod(int(k), S.shape[0])
        nm = torch.minimum(Gm[a], Gm[b])
        keep = [g for g in range(len(groups)) if g not in (a, b)]
        groups = [groups[g] for g in keep] + [groups[a] + groups[b]]
        Gm = torch.cat([Gm[keep], nm[None]])
        hist.append(snapshot(groups, Gm))
    return groups, Gm, hist


def snapshot(groups, Gm):
    return dict(ng=len(groups), sizes=sorted((len(g) for g in groups), reverse=True),
                min_bacc=float(Gm.max(1).values.min()))


def choose(B, g, Gm_row):
    """组内最小 bACC ≥.9 的配方中取组内平均 bACC 最高者（不满足则取最小值最大者）。"""
    feas = Gm_row >= TH
    if feas.any():
        mean = torch.nan_to_num(B[g], nan=0).mean(0)
        return int(torch.where(feas, mean, torch.tensor(-1.0, device=dev)).argmax())
    return int(Gm_row.argmax())


def summarize(M, B, groups, Gm, subs, label):
    out, per = [], {}
    for g, row in zip(groups, Gm):
        k = choose(B, g, row)
        vals = M[g][:, :, k]
        out.append(dict(members=[subs[m] for m in g], recipe=label(k), k=k,
                        min_bacc=float(np.nanmin(vals[:, 0])), mean_bacc=float(np.nanmean(vals[:, 0]))))
        for m, v in zip(g, vals):
            per[subs[m]] = v.tolist()
    A = np.array(list(per.values()))
    return out, dict(bacc=float(A[:, 0].mean()), auc=float(A[:, 1].mean()), f1=float(A[:, 2].mean()),
                     n90=int((A[:, 0] >= TH).sum()))


def main():
    R = load()
    subs = R["subs"]
    M, names, label, sizes = build_pool(R)
    Bn = np.nan_to_num(M[:, 0], nan=-1.0)
    B = torch.tensor(Bn, device=dev)
    log = open(HERE / "merge.log", "w", encoding="utf-8")

    def pr(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.write(s + "\n")

    pr("pool sizes", sizes, "total", M.shape[2])
    orc = B.max(1).values.cpu().numpy()
    pr("每人 oracle≥.9:", int((orc >= TH).sum()), "/", len(subs), "未达:", [subs[i] for i in np.where(orc < TH)[0]])
    # 各子池单独 oracle 覆盖
    o1 = Bn[:, :sizes["n_cands"]].max(1)
    o2 = Bn[:, :sizes["n_cands"] + sizes["n_pair"]].max(1)
    pr(f"覆盖：cands {int((o1 >= TH).sum())}，+pairs {int((o2 >= TH).sum())}，+excl {int((orc >= TH).sum())}")

    P5 = prof_matrix(R, AX5)
    Praw = R["P"][:, [R["keys"].index(c) for c in AX5]]
    ie8 = R["meta"]["base"].index("ens8")
    ens8 = np.nanmax(R["ev"][:, 0, ie8, :], 1)
    res = {}
    allm = list(range(len(subs)))
    for name, fn in (("matching", merge_matching), ("hac", merge_hac)):
        groups, Gm, hist = fn(B, allm)
        gs, mt = summarize(M, B, groups, Gm, subs, label)
        pr(f"\n=== {name} ===")
        if name == "matching":
            for r, h in enumerate(hist, 1):
                pr(f"round {r}: ng={h['ng']} sizes={h['sizes']} min_bacc={h['min_bacc']:.3f}")
        else:
            seq = [57] + [h["ng"] for h in hist]
            pr(f"HAC 合并 {len(hist)} 步, 组数 {seq[0]}→{seq[-1]}; 最终 sizes={hist[-1]['sizes']}")
        pr("final:", mt)
        gs.sort(key=lambda d: -len(d["members"]))
        for d in gs:
            idx = [subs.index(q) for q in d["members"]]
            d["prof_ax5"] = dict(zip(AX5, np.nanmean(Praw[idx], 0).round(3).tolist()))
            d["ens8"] = float(np.nanmean(ens8[idx]))
            pr(f"[{len(idx)}] min {d['min_bacc']:.3f} mean {d['mean_bacc']:.3f} ens8 {d['ens8']:.3f} "
               f"{d['recipe']} | {d['prof_ax5']} | {' '.join(d['members'])}")
        res[name] = dict(hist=hist, groups=gs, metrics=mt)

    # 逐轮详细（重放 matching，记录每轮 57 人平均指标与配方）
    pr("\n=== matching 逐轮 57 人平均（每组取组内最小≥.9 下平均最高的配方）===")
    groups = [[m] for m in allm]
    Gm = B.clone()
    rd = []
    rnd = 0
    while True:
        gs, mt = summarize(M, B, groups, Gm, subs, label)
        nrec = len({d["k"] for d in gs})
        pr(f"round {rnd}: ng={len(groups)} 不同配方数={nrec} bACC {mt['bacc']:.3f} AUC {mt['auc']:.3f} "
           f"F1 {mt['f1']:.3f} n90 {mt['n90']} 组内最小 {min(d['min_bacc'] for d in gs):.3f}")
        rd.append(dict(round=rnd, ng=len(groups), sizes=sorted((len(g) for g in groups), reverse=True),
                       metrics=mt, n_recipes=nrec,
                       groups=[dict(members=d["members"], recipe=d["recipe"], min_bacc=d["min_bacc"])
                               for d in gs if len(d["members"]) > 1]))
        S = pair_scores(Gm).cpu().numpy()
        Gr = nx.Graph()
        for a, b in zip(*np.where(S >= TH)):
            Gr.add_edge(int(a), int(b), weight=float(S[a, b]))
        mtc = nx.max_weight_matching(Gr, maxcardinality=True) if Gr.number_of_edges() else set()
        if not mtc:
            break
        used, ng, nm = set(), [], []
        for a, b in mtc:
            used |= {a, b}
            ng.append(groups[a] + groups[b])
            nm.append(torch.minimum(Gm[a], Gm[b]))
        for g in range(len(groups)):
            if g not in used:
                ng.append(groups[g])
                nm.append(Gm[g])
        groups, Gm = ng, torch.stack(nm)
        rnd += 1
    res["matching"]["rounds_detail"] = rd

    # LOSO 可部署性
    pr("\n=== LOSO 画像分配（ax5 z 画像；其余 56 人重跑 matching 合并）===")
    lo = {"centroid": [], "1nn": []}
    ok = {"centroid": [], "1nn": []}
    anyok, ngs = [], []
    for i in allm:
        oth = [m for m in allm if m != i]
        groups, Gm, _ = merge_matching(B, oth)
        ngs.append(len(groups))
        ks = [choose(B, g, row) for g, row in zip(groups, Gm)]
        fits = [Bn[i, k] >= TH for k in ks]
        anyok.append(any(fits))
        cen = np.array([P5[g].mean(0) for g in groups])
        gi_c = int(np.argmin(((cen - P5[i]) ** 2).sum(1)))
        nn = oth[int(np.argmin(((P5[oth] - P5[i]) ** 2).sum(1)))]
        gi_n = next(gi for gi, g in enumerate(groups) if nn in g)
        for nm_, gi in (("centroid", gi_c), ("1nn", gi_n)):
            lo[nm_].append(M[i, :, ks[gi]])
            ok[nm_].append(bool(fits[gi]))
    for nm_ in lo:
        A = np.nan_to_num(np.array(lo[nm_]), nan=0.5)
        pr(f"{nm_}: bACC {A[:, 0].mean():.3f} AUC {A[:, 1].mean():.3f} F1 {A[:, 2].mean():.3f} "
           f"n90 {int((A[:, 0] >= TH).sum())} 分组正确率 {np.mean(ok[nm_]):.3f}")
        res.setdefault("loso", {})[nm_] = dict(bacc=float(A[:, 0].mean()), auc=float(A[:, 1].mean()),
                                               f1=float(A[:, 2].mean()), n90=int((A[:, 0] >= TH).sum()),
                                               acc_group=float(np.mean(ok[nm_])),
                                               per={subs[i]: lo[nm_][i].tolist() for i in allm})
    pr(f"留出者存在任一可用组（≥.9）比例 {np.mean(anyok):.3f}; 56 人合并后组数 中位 {int(np.median(ngs))} "
       f"范围 {min(ngs)}–{max(ngs)}")
    res["loso"]["any_fit"] = float(np.mean(anyok))
    res["loso"]["ngroups"] = ngs
    res["pool"] = sizes
    json.dump(res, open(HERE / "merge.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
