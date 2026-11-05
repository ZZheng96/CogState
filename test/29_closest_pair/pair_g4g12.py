"""E29 小搜索：只对最接近的 G4+G12（6 人）找共用配方，目标组内最小 bACC ≥ .9。
依次：静态池（含 ev2 单 cfg × 85 阈值）→ d2 两 cfg z 平均 × 85 阈值（剪枝前 NC 个）→ d3 三 cfg（前 NC3 个）。
运行：python -u test/29_closest_pair/pair_g4g12.py → pair_g4g12.log / pair_g4g12.json
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
sys.path.insert(0, str(T / "26_merge2"))
sys.path.insert(0, str(T / "25_merge"))
from e26lib import THR2, metrics, subj_data, zpair  # noqa: E402
from clib import load, dev  # noqa: E402
from merge import build_pool, cfg_name  # noqa: E402

TH, NC, NC3, NT2 = .9, 80, 40, len(THR2)
GA, GB = 4, 12
log = open(HERE / "pair_g4g12.log", "w", encoding="utf-8")
t0 = time.time()


def pr(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.write(s + "\n")
    log.flush()


R = load()
subs, nq, base = R["subs"], R["nq"], R["meta"]["base"]
ev2 = np.load(T / "26_merge2" / "ev2.npz")["ev2"]
M0, _, label0, _ = build_pool(R)
P0 = M0.shape[2]
Mx = np.concatenate([M0, ev2.transpose(0, 3, 1, 2).reshape(nq, 3, -1)], 2)
del M0
G = json.load(open(T / "26_merge2" / "merge2.json", encoding="utf-8"))["groups"]
U = [subs.index(q) for q in G[GA]["members"] + G[GB]["members"]]
pr(f"G{GA}+G{GB} 成员 {[subs[j] for j in U]}  载入 {time.time() - t0:.0f}s")


def slabel(k):
    if k < P0:
        return label0(k)
    c, t = divmod(k - P0, NT2)
    return f"{cfg_name(c, base)} | {THR2[t][0]}"


def report(name, V, lab):
    pr(f"[{name}] 组内最小 {V[:, 0].min():.3f} 均值 {V[:, 0].mean():.3f}  {lab}")
    for j, x in zip(U, V):
        pr(f"    {subs[j]}: bACC {x[0]:.3f} AUC {x[1]:.3f} F1 {x[2]:.3f}{'  <.9' if x[0] < TH else ''}")


def pick(run, tot):
    feas = run >= TH
    k = int(torch.where(feas, tot, torch.tensor(-1e9, device=dev)).argmax()) if feas.any() else int(run.argmax())
    return float(run.flatten()[k]), k


# ---- 1. 静态池 ----
Bs = np.nan_to_num(Mx[U, 0], nan=0.)
mn = Bs.min(0)
k = int(np.argmax(np.where(mn >= TH, Bs.mean(0), mn - 1)))
best = (float(mn[k]), "static", slabel(k), Mx[U, :, k].copy())
report("静态", best[3], best[2])

SD = [subj_data(R, subs[j]) for j in U]
bm = np.nan_to_num(ev2[U][..., 0], nan=0.).max(2).min(0)
cand = [int(c) for c in np.argsort(-bm)[:NC]]
pr(f"候选 cfg {len(cand)}（单 cfg 最优组内最小 {bm[cand[0]]:.3f}）")


def search(rows_fn):
    run = tot = None
    for C, cal_t, tst, yy in SD:
        b = torch.nan_to_num(metrics(rows_fn(C, cal_t), tst, yy, ch=128, auc=False)[..., 0], nan=-1.)
        run = b if run is None else torch.minimum(run, b)
        tot = b if tot is None else tot + b
    return pick(run, tot)


def full(rows_fn, p, t):
    return np.array([metrics(rows_fn(C, cal_t)[p:p + 1], tst, yy)[0, t].cpu().numpy()
                     for C, cal_t, tst, yy in SD])


def zrows(M, cal_t):
    return (M - M[:, cal_t].mean(1, keepdim=True)) / (M[:, cal_t].std(1, keepdim=True) + 1e-6)


# ---- 2. d2 ----
if best[0] < TH:
    ia, ib = map(list, zip(*itertools.combinations(cand, 2)))
    fn = lambda C, cal_t: zpair(C, cal_t, ia, ib)  # noqa: E731
    v, kk = search(fn)
    p, t = divmod(kk, NT2)
    lab = f"z[{cfg_name(ia[p], base)} ; {cfg_name(ib[p], base)}] | {THR2[t][0]}"
    V = full(fn, p, t)
    report(f"d2 {len(ia)} 对 {time.time() - t0:.0f}s", V, lab)
    if v > best[0]:
        best = (v, "d2", lab, V)

# ---- 3. d3 ----
if best[0] < TH:
    c3 = cand[:NC3]
    tri = list(itertools.combinations(c3, 3))
    ta, tb, tc = map(list, zip(*tri))
    fn = lambda C, cal_t: zrows(C[ta] + C[tb] + C[tc], cal_t)  # noqa: E731
    v, kk = search(fn)
    p, t = divmod(kk, NT2)
    lab = "z[" + " ; ".join(cfg_name(c, base) for c in (ta[p], tb[p], tc[p])) + f"] | {THR2[t][0]}"
    V = full(fn, p, t)
    report(f"d3 {len(tri)} 组 {time.time() - t0:.0f}s", V, lab)
    if v > best[0]:
        best = (v, "d3", lab, V)

pr(f"\n最终：{'可行' if best[0] >= TH else '不可行'} 组内最小 {best[0]:.3f} 方式 {best[1]}  {best[2]}  [{time.time() - t0:.0f}s]")
json.dump(dict(members=[subs[j] for j in U], feasible=best[0] >= TH, min_bacc=best[0], kind=best[1],
               recipe=best[2], per=dict(zip([subs[j] for j in U], best[3].tolist()))),
          open(HERE / "pair_g4g12.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
