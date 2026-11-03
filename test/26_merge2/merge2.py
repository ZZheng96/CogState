"""E26：扩池 + 定向搜索，从 E25 的 21 组继续两两合并，全程组内最小 bACC ≥ 0.9。
静态池 = E25 池（cands×8 阈值 + 两两 z 平均×8 阈值 + 7 专属配方）+ cands×85 扩展因果阈值（e26lib.THR2）。
定向搜索（每对候选组）：在合并组成员上剪枝 cands（成员各自最优阈值 bACC 的最小值 ≥.7，取前 NC），
  生成全部两两 z 平均（按本人校准段再标准化）× 85 阈值，取组内最小 bACC 最大者。
逐轮：边 = max(静态, 定向) ≥.9；networkx max_weight_matching(maxcardinality) 合并；直到无边。
LOSO（简化版，不重跑合并）：留出者从各组中剔除，按 ax5 画像分到最近组心；
  若分回本组，则在其余成员上重选配方（不看留出者）。
运行：.venv\\Scripts\\python.exe test/26_merge2/merge2.py   → ev2.npz（缓存）/ merge2.json / merge2.log
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "25_merge"))
from e26lib import THR2, metrics, subj_data, zpair  # noqa: E402
from clib import load, prof_matrix, AX5, dev  # noqa: E402
from merge import build_pool, cfg_name  # noqa: E402

TH = .9
NC = 60          # 定向搜索剪枝后保留的 cfg 数（→ C(60,2)=1770 对 × 85 阈值）
NCAND = 60       # 每轮做定向搜索的候选组对上限
LOWER = .8       # 静态最优 ≥ LOWER 的组对才做定向搜索
NT2 = len(THR2)


def main():
    t00 = time.time()
    R = load()
    subs, nq = R["subs"], R["nq"]
    base = R["meta"]["base"]
    log = open(HERE / "merge2.log", "w", encoding="utf-8")

    def pr(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    SD = [subj_data(R, q) for q in subs]
    f2 = HERE / "ev2.npz"
    if f2.exists():
        ev2 = np.load(f2)["ev2"]
    else:
        ev2 = np.stack([metrics(C, tst, yy).cpu().numpy() for C, _, tst, yy in SD])  # (57,1653,85,3)
        np.savez_compressed(f2, ev2=ev2)
    pr(f"ev2 {ev2.shape}  {time.time() - t00:.0f}s")

    M0, names, label0, sizes = build_pool(R)
    P0 = M0.shape[2]
    Mx = np.concatenate([M0, ev2.transpose(0, 3, 1, 2).reshape(nq, 3, -1)], 2)  # (57,3,P)
    del M0
    Bn = np.nan_to_num(Mx[:, 0], nan=-1.0).astype(np.float32)
    B = torch.tensor(Bn, device=dev)
    ev2b = np.nan_to_num(ev2[..., 0], nan=-1.0)  # (57,1653,85)

    def slabel(k):
        if k < P0:
            return label0(k)
        c, t = divmod(k - P0, NT2)
        return f"{cfg_name(c, base)} | {THR2[t][0]}"

    orc0, orc = Bn[:, :P0].max(1), Bn.max(1)
    pr(f"静态池 {P0}+{Bn.shape[1] - P0}；每人 oracle≥.9：E25 池 {(orc0 >= TH).sum()}，扩池 {(orc >= TH).sum()}；"
       f"oracle 均值 {orc0.mean():.3f}→{orc.mean():.3f}")

    # ---------- 定向搜索 ----------
    dcache = {}

    def dyn(U):
        key = frozenset(U)
        if key in dcache:
            return dcache[key]
        bm = ev2b[U].max(2).min(0)
        cand = [int(c) for c in np.argsort(-bm)[:NC] if bm[c] >= .7]
        res = (-1.0, None, None)
        if len(cand) >= 2:
            ia, ib = map(list, zip(*itertools.combinations(cand, 2)))
            run, tot = None, None
            for j in U:
                C, cal_t, tst, yy = SD[j]
                b = torch.nan_to_num(metrics(zpair(C, cal_t, ia, ib), tst, yy, ch=128, auc=False)[..., 0], nan=-1.)
                run = b if run is None else torch.minimum(run, b)
                tot = b if tot is None else tot + b
                if run.max() < TH - .03 and j != U[-1] and run.max() < .85:
                    break
            feas = run >= TH
            k = int(torch.where(feas, tot, torch.tensor(-1e9, device=dev)).argmax()) if feas.any() \
                else int(run.argmax())
            p, t = divmod(k, NT2)
            res = (float(run.flatten()[k]), (ia[p], ib[p], t), None)
        dcache[key] = res
        return res

    def dyn_metrics(rec, j):
        a, b, t = rec
        C, cal_t, tst, yy = SD[j]
        return metrics(zpair(C, cal_t, [a], [b]), tst, yy)[0, t].cpu().numpy()

    def dlabel(rec):
        a, b, t = rec
        return f"z[{cfg_name(a, base)} ; {cfg_name(b, base)}] | {THR2[t][0]}"

    # ---------- 组与配方 ----------
    def static_choice(g):
        mn = B[g].min(0).values
        feas = mn >= TH
        if feas.any():
            mean = B[g].mean(0)
            k = int(torch.where(feas, mean, torch.tensor(-1.0, device=dev)).argmax())
        else:
            k = int(mn.argmax())
        return k, float(mn[k]), float(B[g, k].mean())

    def group_recipe(g, drec):
        """返回 (label, per-member metrics (n,3), min_bacc)。drec = 定向配方或 None。"""
        k, smin, smean = static_choice(g)
        opts = [(smin >= TH, smean, smin, "s", k)]
        if drec is not None:
            V = np.array([dyn_metrics(drec, j) for j in g])
            opts.append((V[:, 0].min() >= TH, V[:, 0].mean(), V[:, 0].min(), "d", (drec, V)))
        best = max(opts, key=lambda o: (o[0], o[1] if o[0] else o[2]))
        if best[3] == "s":
            return slabel(best[4]), Mx[g][:, :, best[4]], best[2], ("s", best[4])
        return dlabel(best[4][0]), best[4][1], best[2], ("d", best[4][0])

    def summary(groups, drecs):
        per, gs = {}, []
        for g, dr in zip(groups, drecs):
            lab, V, mn, rk = group_recipe(g, dr)
            gs.append(dict(members=[subs[m] for m in g], recipe=lab, kind=rk[0], min_bacc=float(np.nanmin(V[:, 0])),
                           mean_bacc=float(np.nanmean(V[:, 0])), rk=rk))
            for m, v in zip(g, V):
                per[m] = v
        A = np.nan_to_num(np.array([per[m] for m in range(nq)]), nan=.5)
        mt = dict(bacc=float(A[:, 0].mean()), auc=float(A[:, 1].mean()), f1=float(A[:, 2].mean()),
                  n90=int((A[:, 0] >= TH).sum()))
        return gs, mt

    # 参照：仅扩展静态池，从 57 个单例做 matching（不定向搜索）
    def static_matching(members):
        groups = [[m] for m in members]
        Gm = B[members].clone()
        while True:
            ng_ = len(groups)
            S = torch.full((ng_, ng_), -1.0, device=dev)
            for a in range(ng_ - 1):
                S[a, a + 1:] = torch.minimum(Gm[a][None], Gm[a + 1:]).max(1).values
            S = S.cpu().numpy()
            Gr = nx.Graph()
            for a, b in zip(*np.where(S >= TH)):
                Gr.add_edge(int(a), int(b), weight=float(S[a, b]))
            mt = nx.max_weight_matching(Gr, maxcardinality=True) if Gr.number_of_edges() else set()
            if not mt:
                return groups
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

    sg = static_matching(list(range(nq)))
    pr(f"参照（扩展静态池、单例起步 matching）：最终 {len(sg)} 组 sizes={sorted(map(len, sg), reverse=True)}")

    # ---------- 主流程：从 E25 21 组继续 ----------
    e25 = json.load(open(HERE.parent / "25_merge" / "merge.json", encoding="utf-8"))["matching"]["groups"]
    groups = [[subs.index(q) for q in d["members"]] for d in e25]
    drecs = [None] * len(groups)
    rounds = []
    rnd = 0
    while True:
        gs, mt = summary(groups, drecs)
        rd = dict(round=rnd, ng=len(groups), sizes=sorted(map(len, groups), reverse=True), metrics=mt,
                  min_bacc=min(d["min_bacc"] for d in gs), n_dyn=sum(d["kind"] == "d" for d in gs))
        rounds.append(rd)
        pr(f"round {rnd}: ng={rd['ng']} sizes={rd['sizes']} bACC {mt['bacc']:.3f} AUC {mt['auc']:.3f} "
           f"F1 {mt['f1']:.3f} n90 {mt['n90']} 组内最小 {rd['min_bacc']:.3f} 定向配方组 {rd['n_dyn']}  "
           f"[{time.time() - t00:.0f}s]")
        ng_ = len(groups)
        Gm = torch.stack([B[g].min(0).values for g in groups])
        S = np.full((ng_, ng_), -1.0)
        for a in range(ng_ - 1):
            S[a, a + 1:] = torch.minimum(Gm[a][None], Gm[a + 1:]).max(1).values.cpu().numpy()
        W = S.copy()
        DR = {}
        cand = [(S[a, b], a, b) for a, b in zip(*np.where((S >= LOWER) & (S < TH)))]
        cand.sort(reverse=True)
        nfeas = 0
        for _, a, b in cand[:NCAND]:
            v, rec, _ = dyn(groups[a] + groups[b])
            if v > W[a, b]:
                W[a, b] = v
                DR[(a, b)] = rec
                nfeas += v >= TH
        pr(f"  静态边 {(S >= TH).sum()}，定向搜索 {min(len(cand), NCAND)}/{len(cand)} 对，新增可行边 {nfeas}")
        Gr = nx.Graph()
        for a, b in zip(*np.where(W >= TH)):
            Gr.add_edge(int(a), int(b), weight=float(W[a, b]))
        mtc = nx.max_weight_matching(Gr, maxcardinality=True) if Gr.number_of_edges() else set()
        if not mtc:
            break
        used, ng, nd = set(), [], []
        for a, b in mtc:
            a, b = min(a, b), max(a, b)
            used |= {a, b}
            ng.append(groups[a] + groups[b])
            nd.append(DR.get((a, b)))
            pr(f"  合并 {[subs[m] for m in groups[a]]} + {[subs[m] for m in groups[b]]} → {W[a, b]:.3f} "
               f"({'定向' if (a, b) in DR else '静态'})")
        for g in range(ng_):
            if g not in used:
                ng.append(groups[g])
                nd.append(drecs[g])
        groups, drecs = ng, nd
        rnd += 1

    gs, mt = summary(groups, drecs)
    P5 = prof_matrix(R, AX5)
    order = sorted(range(len(gs)), key=lambda i: -len(gs[i]["members"]))
    pr("\n=== 最终分组 ===")
    for i in order:
        d = gs[i]
        pr(f"[{len(d['members'])}] min {d['min_bacc']:.3f} mean {d['mean_bacc']:.3f} {d['recipe']} | "
           f"{' '.join(d['members'])}")

    # ---------- LOSO（简化）----------
    lo, ok = [], []
    for i in range(nq):
        G2 = [([m for m in g if m != i], dr) for g, dr in zip(groups, drecs)]
        G2 = [(g, dr) for g, dr in G2 if g]
        cen = np.array([P5[g].mean(0) for g, _ in G2])
        gi = int(np.argmin(((cen - P5[i]) ** 2).sum(1)))
        g, dr = G2[gi]
        _, _, _, rk = group_recipe(g, dr)
        v = Mx[i, :, rk[1]] if rk[0] == "s" else dyn_metrics(rk[1], i)
        lo.append(v)
        ok.append(bool(np.nan_to_num(v[0]) >= TH))
    A = np.nan_to_num(np.array(lo), nan=.5)
    loso = dict(bacc=float(A[:, 0].mean()), auc=float(A[:, 1].mean()), f1=float(A[:, 2].mean()),
                n90=int((A[:, 0] >= TH).sum()), acc_group=float(np.mean(ok)))
    pr(f"\nLOSO（画像最近组心，简化不重跑合并）：bACC {loso['bacc']:.3f} AUC {loso['auc']:.3f} F1 {loso['f1']:.3f} "
       f"n90 {loso['n90']} 分组正确率 {loso['acc_group']:.3f}")
    for d in gs:
        d["rk"] = [d["rk"][0], d["rk"][1] if d["rk"][0] == "s" else list(map(int, d["rk"][1]))]
        d["rk"][1] = int(d["rk"][1]) if d["rk"][0] == "s" else d["rk"][1]
    json.dump(dict(rounds=rounds, groups=[gs[i] for i in order], metrics=mt, loso=loso,
                   static_ref=dict(ng=len(sg), sizes=sorted(map(len, sg), reverse=True)),
                   oracle=dict(e25=int((orc0 >= TH).sum()), ext=int((orc >= TH).sum())),
                   params=dict(NC=NC, NCAND=NCAND, LOWER=LOWER, n_thr=NT2)),
              open(HERE / "merge2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    pr(f"done {time.time() - t00:.0f}s")


if __name__ == "__main__":
    main()
