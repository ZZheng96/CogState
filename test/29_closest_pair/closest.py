"""E29：在 E26 的 18 组上找最接近可行的组对，用更大的配方搜索尝试合并（组内最小 bACC ≥ 0.9，oracle 口径）。
步骤 1：复现 round2 的候选组对搜索（d2 NC=60 → d3 NC3=30），列出最接近的 5 对及其成员。
步骤 2：对最接近的 TOPK 对做扩展搜索：
  d2x  两两 z 平均，剪枝前 NCX 个 cfg
  d3x  三配置 z 平均，剪枝前 NC3X 个 cfg
  gr   从 d3x 最优出发前向贪心加 cfg（等权 z 平均，至多 GMAX 个），每步全部 85 阈值
  wg   d3x 最优三元组的权重网格 {0.5,1,1.5,2}^3
可行则报告合并后 17 组的 57 人 bACC/AUC/F1。
依赖：test/26_merge2/ev2.npz、test/25_merge/pairs_eval.npz、excl.json、E23 缓存。
运行：python -u test/29_closest_pair/closest.py → closest.json / closest.log
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
T = HERE.parent
for p in ("26_merge2", "25_merge", "24_cluster"):
    sys.path.insert(0, str(T / p))
from e26lib import THR2, metrics, subj_data, zpair  # noqa: E402
from clib import load, dev  # noqa: E402
from merge import build_pool, cfg_name  # noqa: E402

TH, LOWER, EXT = .9, .8, .85
NC, NC3 = 60, 30
NCX, NC3X, GMAX, TOPK = 150, 50, 6, 3
NT2 = len(THR2)
t00 = time.time()
LOG = open(HERE / "closest.log", "w", encoding="utf-8")


def pr(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def zrows(M, cal_t):
    return (M - M[:, cal_t].mean(1, keepdim=True)) / (M[:, cal_t].std(1, keepdim=True) + 1e-6)


R = load()
subs, nq, base = R["subs"], R["nq"], R["meta"]["base"]
SD = [subj_data(R, q) for q in subs]
ev2 = np.load(T / "26_merge2" / "ev2.npz")["ev2"]
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


def search_rows(U, rows_fn, early=.85):
    run = tot = None
    for j in U:
        C, cal_t, tst, yy = SD[j]
        b = torch.nan_to_num(metrics(rows_fn(j), tst, yy, ch=128, auc=False)[..., 0], nan=-1.)
        run = b if run is None else torch.minimum(run, b)
        tot = b if tot is None else tot + b
        if run.max() < early:
            break
    feas = run >= TH
    k = int(torch.where(feas, tot, torch.tensor(-1e9, device=dev)).argmax()) if feas.any() else int(run.argmax())
    return float(run.flatten()[k]), k


def combo_rows(j, combos, w=None):
    """combos: list of tuples of cfg → (len, T) 等权（或 w 加权）z 平均行。"""
    C, cal_t = SD[j][0], SD[j][1]
    out = []
    for a in range(0, len(combos), 2048):
        cs = combos[a:a + 2048]
        k = len(cs[0])
        S = sum(C[[c[i] for c in cs]] * (1 if w is None else w[i]) for i in range(k))
        out.append(zrows(S, cal_t))
    return torch.cat(out)


def search_combos(U, combos, early=.85):
    v, k = search_rows(U, lambda j: combo_rows(j, combos), early)
    p, t = divmod(k, NT2)
    return v, tuple(combos[p]), t


def label(cfgs, t, w=None):
    ws = "" if w is None else f" w={list(w)}"
    return "z[" + " ; ".join(cfg_name(c, base) for c in cfgs) + f"]{ws} | {THR2[t][0]}"


def member_metrics(U, cfgs, t, w=None):
    out = []
    for j in U:
        C, cal_t, tst, yy = SD[j]
        M = combo_rows(j, [tuple(cfgs)], w)
        out.append(metrics(M, tst, yy)[0, t].cpu().numpy())
    return np.array(out)


# ---------- 起点：E26 的 18 组 ----------
e26 = json.load(open(T / "26_merge2" / "merge2.json", encoding="utf-8"))["groups"]
groups = [[subs.index(q) for q in d["members"]] for d in e26]
ng = len(groups)
Gm = torch.stack([B[g].min(0).values for g in groups])
S = np.full((ng, ng), -1.0)
for a in range(ng - 1):
    S[a, a + 1:] = torch.minimum(Gm[a][None], Gm[a + 1:]).max(1).values.cpu().numpy()
cand = sorted([(S[a, b], a, b) for a, b in zip(*np.where((S >= LOWER) & (S < TH)))], reverse=True)
pr(f"18 组，静态边 {(S >= TH).sum()}，候选组对 {len(cand)}")

# ---------- 步骤 1：复现 round2 的 d2/d3 搜索，找最接近的组对 ----------
res1 = []
for i, (s0, a, b) in enumerate(cand):
    U = groups[a] + groups[b]
    c = prune(U, NC)
    best = (-1.0, None, None)
    if len(c) >= 2:
        best = search_combos(U, list(itertools.combinations(c, 2)))
    if EXT <= best[0] < TH:
        r3 = search_combos(U, list(itertools.combinations(c[:NC3], 3)))
        if r3[0] > best[0]:
            best = r3
    res1.append(dict(a=int(a), b=int(b), static=float(s0), v=best[0], cfgs=best[1], t=best[2]))
    if (i + 1) % 10 == 0:
        pr(f"  {i + 1}/{len(cand)}  当前最好 {max(r['v'] for r in res1):.3f} [{time.time() - t00:.0f}s]")
res1.sort(key=lambda r: -r["v"])
pr("\n最接近的 5 对：")
for r in res1[:5]:
    ga, gb = [subs[m] for m in groups[r["a"]]], [subs[m] for m in groups[r["b"]]]
    pr(f"  {r['v']:.3f}  静态 {r['static']:.3f}  {ga} + {gb}  {label(r['cfgs'], r['t'])}")

# ---------- 步骤 2：扩展搜索 ----------
out2 = []
for r in res1[:TOPK]:
    a, b = r["a"], r["b"]
    U = groups[a] + groups[b]
    names = [subs[m] for m in U]
    pr(f"\n=== 扩展搜索 {[subs[m] for m in groups[a]]} + {[subs[m] for m in groups[b]]}（基线 {r['v']:.3f}）===")
    c = prune(U, NCX)
    trials = [("base", r["v"], r["cfgs"], r["t"], None)]
    v, cf, t = search_combos(U, list(itertools.combinations(c, 2)), early=0)
    trials.append(("d2x", v, cf, t, None))
    pr(f"  d2x NC={len(c)}: {v:.3f}  {label(cf, t)} [{time.time() - t00:.0f}s]")
    v, cf, t = search_combos(U, list(itertools.combinations(c[:NC3X], 3)), early=0)
    trials.append(("d3x", v, cf, t, None))
    pr(f"  d3x NC3={min(NC3X, len(c))}: {v:.3f}  {label(cf, t)} [{time.time() - t00:.0f}s]")
    cur, cv = list(cf), v
    while len(cur) < GMAX:
        combos = [tuple(cur) + (x,) for x in c if x not in cur]
        v2, cf2, t2 = search_combos(U, combos, early=0)
        pr(f"  gr m={len(cur) + 1}: {v2:.3f}  {label(cf2, t2)} [{time.time() - t00:.0f}s]")
        trials.append((f"gr{len(cur) + 1}", v2, cf2, t2, None))
        if v2 <= cv:
            break
        cur, cv = list(cf2), v2
    tri = [x for x in trials if x[0] == "d3x"][0][2]
    best_w = (-1.0, None, None)
    for w in itertools.product((.5, 1., 1.5, 2.), repeat=3):
        if len(set(w)) == 1 and w != (1., 1., 1.):
            continue
        v, k = search_rows(U, lambda j: combo_rows(j, [tri], w), early=0)
        if v > best_w[0]:
            best_w = (v, w, k)
    trials.append(("wg", best_w[0], tri, best_w[2], best_w[1]))
    pr(f"  wg: {best_w[0]:.3f}  {label(tri, best_w[2], best_w[1])} [{time.time() - t00:.0f}s]")
    kind, v, cf, t, w = max(trials, key=lambda x: x[1])
    V = member_metrics(U, cf, t, w)
    pr(f"  最优 {kind}: 组内最小 bACC {V[:, 0].min():.3f}，组均值 bACC {V[:, 0].mean():.3f} "
       f"AUC {V[:, 1].mean():.3f} F1 {V[:, 2].mean():.3f}  {'可行' if V[:, 0].min() >= TH else '不可行'}")
    for q, x in zip(names, V):
        pr(f"    {q}: bACC {x[0]:.3f} AUC {x[1]:.3f} F1 {x[2]:.3f}")
    out2.append(dict(groups=[[subs[m] for m in groups[a]], [subs[m] for m in groups[b]]],
                     trials=[dict(kind=k_, v=float(v_), recipe=label(c_, t_, w_)) for k_, v_, c_, t_, w_ in trials],
                     best=dict(kind=kind, recipe=label(cf, t, w), min_bacc=float(V[:, 0].min()),
                               per={q: dict(bacc=float(x[0]), auc=float(x[1]), f1=float(x[2])) for q, x in zip(names, V)}),
                     feasible=bool(V[:, 0].min() >= TH)))

json.dump(dict(top5=[dict(groups=[[subs[m] for m in groups[r["a"]]], [subs[m] for m in groups[r["b"]]]],
                          v=r["v"], static=r["static"], recipe=label(r["cfgs"], r["t"])) for r in res1[:5]],
               extended=out2, params=dict(NC=NC, NC3=NC3, NCX=NCX, NC3X=NC3X, GMAX=GMAX, TOPK=TOPK)),
          open(HERE / "closest.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
pr(f"done {time.time() - t00:.0f}s")
