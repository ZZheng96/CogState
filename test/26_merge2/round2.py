"""E26 续：从 merge2.json 的 18 组继续合并，全程组内最小 bACC ≥ 0.9。
每个候选组对（静态最优 ≥ LOWER）依次尝试，前一级可行即停：
  d2 = 两两 z 平均（剪枝前 NC 个 cfg）× 85 阈值（同 merge2，全部候选都搜，不再截断）
  d3 = 三配置 z 平均（剪枝前 NC3 个 cfg）× 85 阈值      （仅 d2 最优 ≥ EXT 时）
  lr = 组联合 LR：剪枝前 NLR 个 cfg 的 z 分数为特征，只用组成员校准块（块 0–3）及标签训练，
       输出按本人校准段再标准化 × 85 阈值               （仅 d2 最优 ≥ EXT 时）
有可行边 → max_weight_matching 合并，循环至无可行合并。
LOSO（简化）：同 merge2；lr 配方在其余成员上重训。
运行：.venv\\Scripts\\python.exe test/26_merge2/round2.py → round2.json / round2.log
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

TH, LOWER, EXT = .9, .8, .85
NC, NC3, NLR = 60, 30, 40
L2S = (1e-3, 1e-2, 1e-1)
NT2 = len(THR2)


def zrows(M, cal_t):
    return (M - M[:, cal_t].mean(1, keepdim=True)) / (M[:, cal_t].std(1, keepdim=True) + 1e-6)


def main():
    t00 = time.time()
    R = load()
    subs, nq, base = R["subs"], R["nq"], R["meta"]["base"]
    log = open(HERE / "round2.log", "w", encoding="utf-8")

    def pr(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    SD = [subj_data(R, q) for q in subs]
    YC = [torch.tensor(R["D"][q][1], device=dev).float() for q in subs]  # 全时间轴标签（只取校准段用）
    ev2 = np.load(HERE / "ev2.npz")["ev2"]
    M0, _, label0, _ = build_pool(R)
    P0 = M0.shape[2]
    Mx = np.concatenate([M0, ev2.transpose(0, 3, 1, 2).reshape(nq, 3, -1)], 2)
    del M0
    B = torch.tensor(np.nan_to_num(Mx[:, 0], nan=-1.0).astype(np.float32), device=dev)
    ev2b = np.nan_to_num(ev2[..., 0], nan=-1.0)
    pr(f"载入 {time.time() - t00:.0f}s")

    def slabel(k):
        if k < P0:
            return label0(k)
        c, t = divmod(k - P0, NT2)
        return f"{cfg_name(c, base)} | {THR2[t][0]}"

    def prune(U, n):
        bm = ev2b[U].max(2).min(0)
        return [int(c) for c in np.argsort(-bm)[:n] if bm[c] >= .7]

    # ---------- 配方评估 ----------
    def lr_fit(U, cfgs, l2):
        X, Y, W = [], [], []
        for j in U:
            C, cal_t, _, _ = SD[j]
            Z = torch.nan_to_num(C[cfgs][:, cal_t].T)
            y = YC[j][cal_t]
            w = torch.where(y > 0, .5 / y.sum(), .5 / (1 - y).sum()) / len(U)
            X.append(Z), Y.append(y), W.append(w)
        X, Y, W = torch.cat(X), torch.cat(Y), torch.cat(W)
        w = torch.zeros(len(cfgs), device=dev, requires_grad=True)
        opt = torch.optim.LBFGS([w], max_iter=100, line_search_fn="strong_wolfe")

        def clo():
            opt.zero_grad()
            z = X @ w
            loss = (W * torch.nn.functional.softplus(-(2 * Y - 1) * z)).sum() + l2 * (w ** 2).sum()
            loss.backward()
            return loss

        opt.step(clo)
        return w.detach()

    def rec_rows(rec, j):
        """配方 → 本人 z 分数行 (R,T)，阈值索引 t。"""
        C, cal_t, tst, yy = SD[j]
        kind = rec[0]
        if kind == "d":
            _, a, b, t = rec
            return zpair(C, cal_t, [a], [b]), t
        if kind == "d3":
            _, a, b, c, t = rec
            return zrows(C[[a]] + C[[b]] + C[[c]], cal_t), t
        _, cfgs, w, t = rec[:4]
        return zrows((torch.nan_to_num(C[cfgs]) * w[:, None]).sum(0, keepdim=True), cal_t), t

    def rec_metrics(rec, j):
        if rec[0] == "s":
            return Mx[j, :, rec[1]]
        M, t = rec_rows(rec, j)
        C, cal_t, tst, yy = SD[j]
        return metrics(M, tst, yy)[0, t].cpu().numpy()

    def rec_label(rec):
        k = rec[0]
        if k == "s":
            return slabel(rec[1])
        if k == "d":
            return f"z[{cfg_name(rec[1], base)} ; {cfg_name(rec[2], base)}] | {THR2[rec[3]][0]}"
        if k == "d3":
            return "z[" + " ; ".join(cfg_name(c, base) for c in rec[1:4]) + f"] | {THR2[rec[4]][0]}"
        return f"组联合LR({len(rec[1])} cfg, l2={rec[4]}) | {THR2[rec[3]][0]}"

    def pick(run, tot):
        feas = run >= TH
        k = int(torch.where(feas, tot, torch.tensor(-1e9, device=dev)).argmax()) if feas.any() else int(run.argmax())
        return float(run.flatten()[k]), k

    def search_rows(U, rows_fn, nrow):
        """rows_fn(j) → (nrow,T) 行；返回 (组内 min, 平坦索引)。"""
        run = tot = None
        for j in U:
            C, cal_t, tst, yy = SD[j]
            b = torch.nan_to_num(metrics(rows_fn(j), tst, yy, ch=128, auc=False)[..., 0], nan=-1.)
            run = b if run is None else torch.minimum(run, b)
            tot = b if tot is None else tot + b
            if run.max() < .85:
                break
        return pick(run, tot)

    dcache = {}

    def search(U):
        key = frozenset(U)
        if key in dcache:
            return dcache[key]
        best = (-1.0, None)
        cand = prune(U, NC)
        if len(cand) >= 2:
            ia, ib = map(list, zip(*itertools.combinations(cand, 2)))
            v, k = search_rows(U, lambda j: zpair(SD[j][0], SD[j][1], ia, ib), len(ia))
            p, t = divmod(k, NT2)
            best = (v, ("d", ia[p], ib[p], t))
        if EXT <= best[0] < TH:
            c3 = cand[:NC3]
            tri = list(itertools.combinations(c3, 3))
            ta, tb, tc = map(list, zip(*tri))
            v, k = search_rows(U, lambda j: zrows(SD[j][0][ta] + SD[j][0][tb] + SD[j][0][tc], SD[j][1]), len(tri))
            p, t = divmod(k, NT2)
            if v > best[0]:
                best = (v, ("d3", ta[p], tb[p], tc[p], t))
        if EXT <= best[0] < TH:
            cl = cand[:NLR]
            for l2 in L2S:
                w = lr_fit(U, cl, l2)
                run = tot = None
                for j in U:
                    M, _ = rec_rows(("lr", cl, w, 0), j)
                    b = torch.nan_to_num(metrics(M, SD[j][2], SD[j][3], auc=False)[0, :, 0], nan=-1.)
                    run = b if run is None else torch.minimum(run, b)
                    tot = b if tot is None else tot + b
                v, t = pick(run, tot)
                if v > best[0]:
                    best = (v, ("lr", cl, w, t, l2))
        dcache[key] = best
        return best

    def static_choice(g):
        mn = B[g].min(0).values
        feas = mn >= TH
        k = int(torch.where(feas, B[g].mean(0), torch.tensor(-1.0, device=dev)).argmax()) if feas.any() \
            else int(mn.argmax())
        return ("s", k)

    def group_best(g, rec):
        """组配方：静态最优 vs 已有动态配方，取 (可行, mean) 最大。"""
        opts = []
        for r in [static_choice(g)] + ([rec] if rec is not None and rec[0] != "s" else []):
            V = np.array([rec_metrics(r, j) for j in g])
            mn = np.nanmin(V[:, 0])
            opts.append(((mn >= TH, np.nanmean(V[:, 0]) if mn >= TH else mn), r, V))
        return max(opts, key=lambda o: o[0])[1:]

    def summary(groups, recs):
        per, gs = {}, []
        for g, r in zip(groups, recs):
            r, V = group_best(g, r)
            gs.append(dict(members=[subs[m] for m in g], recipe=rec_label(r), kind=r[0],
                           min_bacc=float(np.nanmin(V[:, 0])), mean_bacc=float(np.nanmean(V[:, 0])), rec=r))
            for m, v in zip(g, V):
                per[m] = v
        A = np.nan_to_num(np.array([per[m] for m in range(nq)]), nan=.5)
        return gs, dict(bacc=float(A[:, 0].mean()), auc=float(A[:, 1].mean()), f1=float(A[:, 2].mean()),
                        n90=int((A[:, 0] >= TH).sum()))

    # ---------- 起点 ----------
    e26 = json.load(open(HERE / "merge2.json", encoding="utf-8"))["groups"]
    groups = [[subs.index(q) for q in d["members"]] for d in e26]
    recs = [("s", d["rk"][1]) if d["rk"][0] == "s" else ("d", *d["rk"][1]) for d in e26]
    rounds, rnd = [], 0
    while True:
        gs, mt = summary(groups, recs)
        recs = [d["rec"] for d in gs]
        rd = dict(round=rnd, ng=len(groups), sizes=sorted(map(len, groups), reverse=True), metrics=mt,
                  min_bacc=min(d["min_bacc"] for d in gs), kinds=[d["kind"] for d in gs])
        rounds.append(rd)
        pr(f"round {rnd}: ng={rd['ng']} sizes={rd['sizes']} bACC {mt['bacc']:.3f} AUC {mt['auc']:.3f} "
           f"F1 {mt['f1']:.3f} n90 {mt['n90']} 组内最小 {rd['min_bacc']:.3f}  [{time.time() - t00:.0f}s]")
        ng_ = len(groups)
        Gm = torch.stack([B[g].min(0).values for g in groups])
        S = np.full((ng_, ng_), -1.0)
        for a in range(ng_ - 1):
            S[a, a + 1:] = torch.minimum(Gm[a][None], Gm[a + 1:]).max(1).values.cpu().numpy()
        W, DR = S.copy(), {}
        cand = sorted([(S[a, b], a, b) for a, b in zip(*np.where((S >= LOWER) & (S < TH)))], reverse=True)
        stat = {"d": 0, "d3": 0, "lr": 0}
        best_inf = []
        for i, (_, a, b) in enumerate(cand):
            v, rec = search(groups[a] + groups[b])
            if v > W[a, b]:
                W[a, b], DR[(a, b)] = v, rec
            if v >= TH:
                stat[rec[0]] += 1
            elif v >= EXT:
                best_inf.append((v, a, b, rec[0]))
            if (i + 1) % 20 == 0:
                pr(f"    {i + 1}/{len(cand)} 可行 {stat} [{time.time() - t00:.0f}s]")
        best_inf.sort(reverse=True)
        pr(f"  静态边 {(S >= TH).sum()}，搜索 {len(cand)} 对，可行 {stat}；"
           f"不可行最佳 {[(round(v, 3), k) for v, _, _, k in best_inf[:5]]}")
        Gr = nx.Graph()
        for a, b in zip(*np.where(W >= TH)):
            Gr.add_edge(int(a), int(b), weight=float(W[a, b]))
        mtc = nx.max_weight_matching(Gr, maxcardinality=True) if Gr.number_of_edges() else set()
        if not mtc:
            break
        used, ng, nr = set(), [], []
        for a, b in mtc:
            a, b = min(a, b), max(a, b)
            used |= {a, b}
            ng.append(groups[a] + groups[b])
            nr.append(DR.get((a, b)))
            pr(f"  合并 {[subs[m] for m in groups[a]]} + {[subs[m] for m in groups[b]]} → {W[a, b]:.3f} "
               f"({DR[(a, b)][0] if (a, b) in DR else '静态'})")
        for g in range(ng_):
            if g not in used:
                ng.append(groups[g])
                nr.append(recs[g])
        groups, recs = ng, nr
        rnd += 1

    gs, mt = summary(groups, recs)
    order = sorted(range(len(gs)), key=lambda i: -len(gs[i]["members"]))
    pr("\n=== 最终分组 ===")
    for i in order:
        d = gs[i]
        pr(f"[{len(d['members'])}] min {d['min_bacc']:.3f} mean {d['mean_bacc']:.3f} {d['recipe']} | "
           f"{' '.join(d['members'])}")

    # ---------- LOSO（简化）----------
    P5 = prof_matrix(R, AX5)
    lo, ok = [], []
    for i in range(nq):
        G2 = [([m for m in g if m != i], r) for g, r in zip(groups, [d["rec"] for d in gs])]
        G2 = [(g, r) for g, r in G2 if g]
        cen = np.array([P5[g].mean(0) for g, _ in G2])
        g, r = G2[int(np.argmin(((cen - P5[i]) ** 2).sum(1)))]
        if r[0] == "lr":
            r = ("lr", r[1], lr_fit(g, r[1], r[4]), r[3], r[4])
        r, _ = group_best(g, r)
        v = rec_metrics(r, i)
        lo.append(v)
        ok.append(bool(np.nan_to_num(v[0]) >= TH))
    A = np.nan_to_num(np.array(lo), nan=.5)
    loso = dict(bacc=float(A[:, 0].mean()), auc=float(A[:, 1].mean()), f1=float(A[:, 2].mean()),
                n90=int((A[:, 0] >= TH).sum()), acc_group=float(np.mean(ok)))
    pr(f"\nLOSO（简化）：bACC {loso['bacc']:.3f} AUC {loso['auc']:.3f} F1 {loso['f1']:.3f} "
       f"n90 {loso['n90']} 分组正确率 {loso['acc_group']:.3f}")
    for d in gs:
        r = d.pop("rec")
        d["rec"] = [r[0]] + [x.cpu().tolist() if torch.is_tensor(x) else x for x in r[1:]]
    json.dump(dict(rounds=rounds, groups=[gs[i] for i in order], metrics=mt, loso=loso,
                   params=dict(NC=NC, NC3=NC3, NLR=NLR, L2S=L2S, LOWER=LOWER, EXT=EXT)),
              open(HERE / "round2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=int)
    pr(f"done {time.time() - t00:.0f}s")


if __name__ == "__main__":
    main()
