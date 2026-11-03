"""C. 通用方案（不看本人测试标签做选择）。需先运行 common.py、subj_profile.py。
C1  固定大集成：E22 胜出频率高的方法族各取一名代表，z 平均（z 参考=校准段），因果阈值。
C1L 同上，但每族代表由其他 56 人测试段平均 bACC 选（LOSO）。
C2  画像驱动元选择器（LOSO，内层 LOSO 选超参）：
    a 全局最优配置（其他 56 人平均 bACC 最高）
    b kNN：画像空间近邻被试的 bACC 加权平均 → argmax（top1 / top3 集成）
    c 岭回归：画像 → 每配置 bACC → argmax
    d 全局 + 本人校准段 bACC（in-sample）加权
C3  阈值：t0 / 因果中位数 / 因果分位数（已含于配置空间与 C1 评估）。
输出 general.json、general_per_subject.csv。
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import L, THR, dev, thresholds, metrics_batch, calref  # noqa: E402

PROF_DROP = ("subject", "special", "nan_frac")


def load():
    meta = json.load(open(HERE / "cands_meta.json"))
    E = np.load(HERE / "cands_eval.npz")
    S = np.load(HERE / "cands_scores.npz")
    subs = list(E["subs"])
    rows = {r["subject"]: r for r in csv.DictReader(open(HERE / "profile.csv", encoding="utf-8"))}
    keys = [k for k in next(iter(rows.values())) if k not in PROF_DROP]
    P = np.array([[float(rows[q][k] or "nan") for k in keys] for q in subs])
    return meta, E["ev"], {q: S[q] for q in subs}, subs, P, keys


def score_of(B, ci, base_n, cal_t):
    """配置索引 ci（base 或 pair）→ z 分数（numpy B: nb×T）。"""
    nb = base_n
    if ci < nb:
        return torch.tensor(B[ci], device=dev)
    k = ci - nb
    a = 0
    while k >= nb - 1 - a:
        k -= nb - 1 - a
        a += 1
    b = a + 1 + k
    s = torch.tensor(B[a] + B[b], device=dev)
    return (s - s[cal_t].mean()) / (s[cal_t].std() + 1e-6)


def eval_ens(q, D, B, cis, nb, thr_list):
    y, blk = D[q][1], D[q][2]
    cal_t = calref(q, D)
    ss = [score_of(B, c, nb, cal_t) for c in cis]
    ss = [s for s in ss if not torch.isnan(s).any()]
    s = torch.stack(ss).mean(0)
    s = (s - s[cal_t].mean()) / (s[cal_t].std() + 1e-6)
    tst = torch.tensor(blk >= 4, device=dev)
    out = {}
    for t in thr_list:
        tt = next(x for x in THR if x[0] == t)
        b, a, f = metrics_batch(s[None], thresholds(s[None], tt), y, tst)
        out[t] = (float(b[0]), float(a[0]), float(f[0]))
    return out


def agg(per):
    a = np.array(per)
    return dict(bacc=float(a[:, 0].mean()), auc=float(a[:, 1].mean()), f1=float(a[:, 2].mean()),
                n90=int((a[:, 0] >= .9).sum()))


def zfit(Ptr, Pte):
    med = np.nanmedian(Ptr, 0)
    Ptr = np.where(np.isnan(Ptr), med, Ptr)
    Pte = np.where(np.isnan(Pte), med, Pte)
    mu, sd = Ptr.mean(0), Ptr.std(0) + 1e-9
    return (Ptr - mu) / sd, (Pte - mu) / sd


def knn_pred(Ptr, Ytr, pte, k):
    d = np.sqrt(((Ptr - pte) ** 2).sum(1))
    nn = np.argsort(d)[:k]
    w = 1 / (d[nn] + 1e-3)
    Y = Ytr[nn]
    m = ~np.isnan(Y)
    return (np.where(m, Y, 0) * w[:, None]).sum(0) / ((m * w[:, None]).sum(0) + 1e-9)


def ridge_pred(Ptr, Ytr, pte, lam):
    mu = np.nanmean(Ytr, 0)
    Y = np.where(np.isnan(Ytr), mu, Ytr) - mu
    A = Ptr.T @ Ptr + lam * np.eye(Ptr.shape[1])
    W = np.linalg.solve(A, Ptr.T @ Y)
    return mu + pte @ W


def pick(pred, avail, m=1):
    p = np.where(avail, pred, -np.inf)
    return np.argsort(-p)[:m]


def main():
    meta, ev, SC, subs, P, pkeys = load()
    _, D = L.load()
    base, thr = meta["base"], meta["thr"]
    nb, nt = len(base), len(thr)
    nq = len(subs)
    TB = ev[:, 0].reshape(nq, -1)  # 测试 bACC (57, ncfg*nthr)
    TA, TF = ev[:, 1].reshape(nq, -1), ev[:, 2].reshape(nq, -1)
    CB = ev[:, 3].reshape(nq, -1)
    import itertools
    for i, q in enumerate(subs):  # 分数含 NaN 的配置（034 的 own/sim/b23 及其组合）置为不可用
        bad = np.isnan(SC[q]).any(1)
        badc = np.concatenate([bad, [bad[a] or bad[b] for a, b in itertools.combinations(range(nb), 2)]])
        badc = np.repeat(badc, nt)
        for M_ in (TB, TA, TF, CB):
            M_[i, badc] = np.nan
    avail = ~np.isnan(TB)
    res, per = {}, {}

    def rec(name, vals):
        per[name] = vals
        res[name] = agg(vals)
        res[name + "@56"] = agg([v for q, v in zip(subs, vals) if q != "034"])
        print(f"{name:28s} bACC {res[name]['bacc']:.3f} AUC {res[name]['auc']:.3f} F1 {res[name]['f1']:.3f}"
              f" n90 {res[name]['n90']}", flush=True)

    def from_cfg(i, j):
        return (float(TB[i, j]), float(TA[i, j]), float(TF[i, j]))

    # 参照
    sj = json.load(open(HERE.parent / "22_subject" / "subject.json"))["per_subject"]
    best = json.load(open(HERE.parent / "22_subject" / "best.json"))
    rec("E22_oracle", [(best[q]["bacc"], best[q]["auc"], best[q]["f1"]) for q in subs])
    ie8 = base.index("ens8") * nt
    rec("ens8|t0(本候选集)", [from_cfg(i, ie8) for i in range(nq)])
    rec("oracle_in_set", [from_cfg(i, int(np.nanargmax(TB[i]))) for i in range(nq)])
    calv = np.where(np.isnan(CB), -1, CB)
    rec("calsel_in_set", [from_cfg(i, int(np.argmax(np.where(avail[i], calv[i], -9)))) if subs[i] != "034"
                          else from_cfg(i, ie8) for i in range(nq)])

    # C1 固定集成
    fixed = {"ens8": ["ens8"],
             "ens8+own": ["ens8", "own800_C0.01"],
             "ens8+own+idx": ["ens8", "own800_C0.01", "idx10"],
             "ens8+own+idx+g": ["ens8", "own800_C0.01", "idx10", "goth_k200", "ghilbert_k200"],
             "ens8+own+idx+g+sim": ["ens8", "own800_C0.01", "idx10", "goth_k200", "ghilbert_k200",
                                    "sim4390_C0.01"],
             "ens8+own+idx+g+sim+b23": ["ens8", "own800_C0.01", "idx10", "goth_k200", "ghilbert_k200",
                                        "sim4390_C0.01", "b23n400_C0.1"]}
    TL = [t[0] for t in THR]
    c1 = {}
    for nm, mem in fixed.items():
        cis = [base.index(x) for x in mem]
        c1[nm] = [eval_ens(q, D, SC[q], cis, nb, TL) for q in subs]
        for t in ("t0", "m300", "q300_.4", "q300_.6"):
            rec(f"C1[{nm}]|{t}", [c1[nm][i][t] for i in range(nq)])

    # C1L：每族代表由其他人 m300 均值选（LOSO）
    from common import family
    fams = np.array([family(n) for n in base])
    jm3 = thr.index("m300")
    Bm3 = TB[:, np.arange(nb) * nt + jm3]
    for famset in (("ens8", "own", "idx"), ("ens8", "own", "idx", "g", "sim"),
                   ("ens8", "joint", "own", "idx", "idxg", "g", "sim", "b23", "f")):
        vals = []
        for i, q in enumerate(subs):
            o = [j for j in range(nq) if j != i]
            mu = np.nanmean(Bm3[o], 0)
            cis = [int(np.where(fams == f)[0][np.argmax(mu[fams == f])]) for f in famset]
            vals.append(eval_ens(q, D, SC[q], cis, nb, ["m300"])["m300"])
        rec(f"C1L[{'+'.join(famset)}]|m300", vals)

    # C2 元选择器
    TS, TN = np.nansum(TB, 0), (~np.isnan(TB)).sum(0)

    def gmean(excl):
        s, n = TS.copy(), TN.copy()
        for j in excl:
            m = ~np.isnan(TB[j])
            s[m] -= TB[j][m]
            n[m] -= 1
        return s / np.maximum(n, 1)

    def select(i_out, method, hp, excl=()):
        o = np.array([j for j in range(nq) if j != i_out and j not in excl])
        if method in ("global", "glob+cal"):
            g = gmean((i_out, *excl))
            if method == "global":
                return g
            c = CB[i_out]
            return g if np.all(np.isnan(c)) else (1 - hp) * g + hp * np.where(np.isnan(c), 0, c)
        Ptr, pte = zfit(P[o], P[i_out:i_out + 1])
        return knn_pred(Ptr, TB[o], pte[0], hp) if method == "knn" else ridge_pred(Ptr, TB[o], pte[0], hp)

    def inner_score(o_idx, method, hp):  # noqa: F811
        out_i = [j for j in range(nq) if j not in o_idx][0]
        return float(np.mean([TB[i, pick(select(i, method, hp, (out_i,)), avail[i])[0]] for i in o_idx]))

    grids = {"global": [None], "knn": [3, 5, 10, 20, 55], "ridge": [1, 10, 100, 1000],
             "glob+cal": [0, .1, .25, .5, .75, 1.0]}
    chosen = {}
    for method, grid in grids.items():
        v1, v3, hps = [], [], []
        for i, q in enumerate(subs):
            o = [j for j in range(nq) if j != i]
            hp = grid[0] if len(grid) == 1 else max(grid, key=lambda h: inner_score(o, method, h))
            hps.append(hp)
            pred = select(i, method, hp)
            top = pick(pred, avail[i], 3)
            v1.append(from_cfg(i, int(top[0])))
            cis = sorted({int(c) // nt for c in top})
            v3.append(eval_ens(q, D, SC[q], cis, nb, [thr[int(top[0]) % nt]])[thr[int(top[0]) % nt]])
            if method != "global":
                print(" ", method, q, "hp", hp, f"{v1[-1][0]:.3f}", flush=True)
        chosen[method] = hps
        rec(f"C2{method}_top1", v1)
        rec(f"C2{method}_top3ens", v3)

    json.dump(dict(results=res, hp=chosen, profile_keys=pkeys), open(HERE / "general.json", "w",
                                                                    encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(HERE / "general_per_subject.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        names = list(per)
        w.writerow(["subject"] + [f"{n}:{m}" for n in names for m in ("bacc", "auc", "f1")])
        for i, q in enumerate(subs):
            w.writerow([q] + [round(x, 4) for n in names for x in per[n][i]])


if __name__ == "__main__":
    main()
