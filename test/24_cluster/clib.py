"""E24 公共：读取 E23 候选缓存、画像；z 平均叠加的评估（GPU，带缓存）；组内贪心叠加。
配置索引 c 指 cands（base + 两两对，共 1653）；阈值索引 t 指 THR（8 种因果阈值）。
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
E23 = HERE.parent / "23_synthesis"
sys.path.insert(0, str(E23))
from common import L, THR, dev, calref  # noqa: E402
from general import load as load23, score_of, PROF_DROP  # noqa: E402,F401

AX3 = ["sim_top10", "drift_sign", "d_emg"]
AX5 = AX3 + ["drift_cos", "d_top20"]


def load():
    meta, ev, SC, subs, P, keys = load23()
    base = meta["base"]
    nb, nt, nq = len(base), len(meta["thr"]), len(subs)
    ev = ev.copy()
    for i, q in enumerate(subs):  # 与 general.py 相同：分数含 NaN 的配置不可用
        bad = np.isnan(SC[q]).any(1)
        badc = np.concatenate([bad, [bad[a] or bad[b] for a, b in itertools.combinations(range(nb), 2)]])
        ev[i, :, badc, :] = np.nan
    _, D = L.load()
    return dict(meta=meta, ev=ev, SC=SC, subs=subs, P=P, keys=keys, D=D, nb=nb, nt=nt, nq=nq)


def prof_matrix(R, cols=None):
    P, keys = R["P"], R["keys"]
    if cols is not None:
        P = P[:, [keys.index(c) for c in cols]]
    med = np.nanmedian(P, 0)
    P = np.where(np.isnan(P), med, P)
    return (P - P.mean(0)) / (P.std(0) + 1e-9)


def agg(vals):
    a = np.array(vals, float)
    return dict(bacc=float(np.nanmean(a[:, 0])), auc=float(np.nanmean(a[:, 1])), f1=float(np.nanmean(a[:, 2])),
                n90=int((a[:, 0] >= .9).sum()))


class StackEval:
    """eval(j, key) → (8 阈值, 3 指标[bACC,AUC,F1])；key = tuple((c, w), ...)，z 平均后按校准段再标准化。"""

    def __init__(self, R):
        self.R, self.cache = R, {}
        self.dat = []
        for q in R["subs"]:
            y, blk = R["D"][q][1], R["D"][q][2]
            cal_t = calref(q, R["D"])
            tst = np.where(blk >= 4)[0]
            self.dat.append((R["SC"][q], cal_t, torch.tensor(tst, device=dev),
                             torch.tensor(y[tst], device=dev).bool()))

    def _compute(self, j, keys):
        B, cal_t, tst, yy = self.dat[j]
        nb = self.R["nb"]
        rows = []
        for key in keys:
            s = sum(w * score_of(B, c, nb, cal_t) for c, w in key)
            rows.append((s - s[cal_t].mean()) / (s[cal_t].std() + 1e-6))
        M = torch.stack(rows)
        out = torch.full((len(keys), len(THR), 3), float("nan"), device=dev)
        St = M[:, tst]
        npos, nneg = yy.sum().float(), (~yy).sum().float()
        for ti, (_, w, qq) in enumerate(THR):
            if w == 0:
                Th = torch.zeros_like(St)
            else:
                Th = torch.empty_like(St)
                pad = torch.cat([torch.full((M.shape[0], w - 1), float("nan"), device=dev), M], 1)
                for a in range(0, M.shape[0], 32):
                    U = pad[a:a + 32].unfold(1, w, 1)[:, tst]
                    Th[a:a + 32] = U.nanquantile(qq, 2)
            P = St > Th
            tp, fp = (P & yy).sum(1).float(), (P & ~yy).sum(1).float()
            out[:, ti, 0] = 0.5 * (tp / npos + 1 - fp / nneg)
            r = (St - Th).argsort(1).argsort(1).float() + 1  # 与 E23 一致：AUC 按减去因果阈值后的分数
            out[:, ti, 1] = (r[:, yy].sum(1) - npos * (npos + 1) / 2) / (npos * nneg)
            out[:, ti, 2] = 2 * tp / (2 * tp + fp + (npos - tp))
        nanrow = torch.isnan(M).any(1)
        out[nanrow] = float("nan")
        return out.cpu().numpy()

    def get_many(self, pairs):
        need = {}
        for j, key in pairs:
            if (j, key) not in self.cache:
                need.setdefault(j, set()).add(key)
        for j, ks in need.items():
            ks = list(ks)
            for a in range(0, len(ks), 256):
                res = self._compute(j, ks[a:a + 256])
                for k, r in zip(ks[a:a + 256], res):
                    self.cache[(j, k)] = r
        return [self.cache[p] for p in pairs]

    def get(self, j, key):
        return self.get_many([(j, key)])[0]


def ekey(cfgs):
    cfgs = sorted(set(cfgs))
    return tuple((c, 1.0) for c in cfgs)


def wmean(V, w):
    """V (n, ...) 含 NaN，w (n,) → 加权 nanmean。"""
    m = ~np.isnan(V)
    ww = w.reshape((-1,) + (1,) * (V.ndim - 1)) * m
    return np.where(m, V, 0).__mul__(ww).sum(0) / np.maximum(ww.sum(0), 1e-12)


def cfg_rank(TB3, members, w, availc):
    """TB3 (nq, ncfg, nt) → 按成员加权平均 bACC 的 (cfg 排序, 每 cfg 最优阈值, 分数矩阵)。"""
    M = wmean(TB3[members], w)  # (ncfg, nt)
    M = np.where(availc[:, None], M, -np.inf)
    best = M.max(1)
    return np.argsort(-best), M.argmax(1), M


def greedy(ev, TB3, members, w, availc, K=10, mmax=5):
    """组内贪心前向叠加：返回 [(cfgs tuple, thr), ...] m=1..mmax。目标=成员加权平均测试 bACC（阈值共享）。"""
    members = list(members)
    w = np.asarray(w, float)
    order, bt, _ = cfg_rank(TB3, members, w, availc)
    pool = [int(c) for c in order[:K]]
    S, out = [], [((pool[0],), int(bt[pool[0]]))]
    S = [pool[0]]
    for _ in range(1, mmax):
        cands = [c for c in pool if c not in S]
        keys = [ekey(S + [c]) for c in cands]
        R = ev.get_many([(j, k) for j in members for k in keys])
        V = np.array([r[:, 0] for r in R]).reshape(len(members), len(cands), len(THR))
        obj = wmean(V, w)
        ci, ti = np.unravel_index(np.nanargmax(obj), obj.shape)
        S.append(cands[ci])
        out.append((tuple(sorted(S)), int(ti)))
    return out
