"""E29 公共：载入 18 组与配置池，try_merge(U) 对任意成员集合找共用配方（静态 → d2 → d3），目标组内最小 bACC ≥ .9。"""
from __future__ import annotations

import itertools
import json
import sys
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

TH, NT2 = .9, len(THR2)
R = load()
subs, nq, base = R["subs"], R["nq"], R["meta"]["base"]
ev2 = np.load(T / "26_merge2" / "ev2.npz")["ev2"]
M0, _, label0, _ = build_pool(R)
P0 = M0.shape[2]
Mx = np.concatenate([M0, ev2.transpose(0, 3, 1, 2).reshape(nq, 3, -1)], 2)
del M0
Bs_all = np.nan_to_num(Mx[:, 0], nan=0.)
ev2b = np.nan_to_num(ev2[..., 0], nan=0.)
GROUPS = [[subs.index(q) for q in g["members"]] for g in
          json.load(open(T / "26_merge2" / "merge2.json", encoding="utf-8"))["groups"]]
_SD = {}


def sd(j):
    if j not in _SD:
        _SD[j] = subj_data(R, subs[j])
    return _SD[j]


def slabel(k):
    if k < P0:
        return label0(k)
    c, t = divmod(k - P0, NT2)
    return f"{cfg_name(c, base)} | {THR2[t][0]}"


def _pick(run, tot):
    feas = run >= TH
    k = int(torch.where(feas, tot, torch.tensor(-1e9, device=dev)).argmax()) if feas.any() else int(run.argmax())
    return float(run.flatten()[k]), k


def _zrows(M, cal_t):
    return (M - M[:, cal_t].mean(1, keepdim=True)) / (M[:, cal_t].std(1, keepdim=True) + 1e-6)


def _search(U, fn, stop):
    run = tot = None
    for j in U:
        C, cal_t, tst, yy = sd(j)
        b = torch.nan_to_num(metrics(fn(C, cal_t), tst, yy, ch=128, auc=False)[..., 0], nan=-1.)
        run = b if run is None else torch.minimum(run, b)
        tot = b if tot is None else tot + b
        if run.max() < stop:
            break
    return _pick(run, tot)


def _full(U, fn, p, t):
    return np.array([metrics(fn(*sd(j)[:2])[p:p + 1], sd(j)[2], sd(j)[3])[0, t].cpu().numpy() for j in U])


def try_merge(U, nc=80, nc3=40, d2_from=.8, d3_from=.85, stop=.8):
    """返回 dict(v=组内最小 bACC, kind, recipe, per=(n,3))。"""
    Bs = Bs_all[U]
    mn = Bs.min(0)
    k = int(np.argmax(np.where(mn >= TH, Bs.mean(0), mn - 1)))
    best = dict(v=float(mn[k]), kind="static", recipe=slabel(k), per=Mx[U, :, k])
    if best["v"] >= TH or best["v"] < d2_from - .05:
        return best
    bm = ev2b[U].max(2).min(0)
    cand = [int(c) for c in np.argsort(-bm)[:nc]]
    ia, ib = map(list, zip(*itertools.combinations(cand, 2)))
    fn = lambda C, cal_t: zpair(C, cal_t, ia, ib)  # noqa: E731
    v, kk = _search(U, fn, stop)
    if v > best["v"]:
        p, t = divmod(kk, NT2)
        best = dict(v=v, kind="d2", recipe=f"z[{cfg_name(ia[p], base)} ; {cfg_name(ib[p], base)}] | {THR2[t][0]}",
                    per=_full(U, fn, p, t))
    if best["v"] >= TH or best["v"] < d3_from:
        return best
    c3 = cand[:nc3]
    ta, tb, tc = map(list, zip(*itertools.combinations(c3, 3)))
    fn = lambda C, cal_t: _zrows(C[ta] + C[tb] + C[tc], cal_t)  # noqa: E731
    v, kk = _search(U, fn, stop)
    if v > best["v"]:
        p, t = divmod(kk, NT2)
        best = dict(v=v, kind="d3", recipe="z[" + " ; ".join(cfg_name(c, base) for c in (ta[p], tb[p], tc[p]))
                    + f"] | {THR2[t][0]}", per=_full(U, fn, p, t))
    return best
