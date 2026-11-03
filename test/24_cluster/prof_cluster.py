"""E24-2/3 可部署分组：只用校准画像聚类，组内方法由同组其他人测试 bACC 选（严格 LOSO）。
特征集：cal=全部校准画像列；ax3=sim_top10/drift_sign/d_emg；ax5=ax3+drift_cos/d_top20（标准化，NaN→中位数）。
聚类：KMeans（GPU，20 次随机初始化）/ Ward 层次聚类，k=1..8。画像不含测试信息，聚类可包含留出者。
组内方法：
  top1  同组其他人平均测试 bACC 最高的 (配置, 阈值)
  greedy 同组其他人上贪心前向 z 平均叠加 m=1..5（池=前 10 配置，阈值共享），m 由组内留二 LOSO 定
         （耗时，只在 Ward k∈{8,6,4,3,2,1} 上跑）
  blend  Ward 分组下：本组 top1 与最近邻组（画像质心）top1 的 z 加权叠加，权重 a∈{0,.25,.5} 由其他人 LOSO 定
单人组（留出后组内无人）回退到其他全部人的 top1。
输出 prof_cluster.json、prof_cluster_per_subject.csv、prof_groups.json。
用法：python prof_cluster.py [--nogreedy]
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, fcluster

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clib import HERE, AX3, AX5, load, prof_matrix, agg, StackEval, ekey, greedy, dev  # noqa: E402
from resp_cluster import kmeans_t  # noqa: E402

R = load()
EV = StackEval(R)
nq, nt = R["nq"], R["nt"]
TB3 = R["ev"][:, 0]  # (nq, ncfg, nt)
TB = TB3.reshape(nq, -1)
TA, TF = R["ev"][:, 1].reshape(nq, -1), R["ev"][:, 2].reshape(nq, -1)
AVC = ~np.all(np.isnan(TB3), 2)  # (nq, ncfg)
CANDS, THRN = R["meta"]["cands"], R["meta"]["thr"]


def top1(members, i):
    mu = np.nanmean(TB[members], 0)
    mu = np.where(np.isnan(TB[i]), -np.inf, mu)
    return int(np.argmax(mu))


def from_flat(i, c):
    return (float(TB[i, c]), float(TA[i, c]), float(TF[i, c]))


def others(lab, i, excl=()):
    g = [j for j in np.where(lab == lab[i])[0] if j != i and j not in excl]
    return g if g else [j for j in range(nq) if j != i and j not in excl]


def run_top1(lab):
    return [from_flat(i, top1(others(lab, i), i)) for i in range(nq)]


def eval_path(j, step):
    cfgs, t = step
    return EV.get(j, ekey(cfgs))[t]


def run_greedy(lab, mmax=5, K=10):
    vals, ms = [], []
    for i in range(nq):
        o = others(lab, i)
        avi = AVC[i]
        if len(o) >= 2:
            sc = np.zeros(mmax)
            cnt = np.zeros(mmax)
            for j in o:
                oj = [x for x in o if x != j]
                path = greedy(EV, TB3, oj, np.ones(len(oj)), avi & AVC[j], K, mmax)
                for m, st in enumerate(path):
                    v = eval_path(j, st)[0]
                    if not np.isnan(v):
                        sc[m] += v
                        cnt[m] += 1
            mbest = int(np.argmax(np.round(sc / np.maximum(cnt, 1), 6)))
        else:
            mbest = 0
        path = greedy(EV, TB3, o, np.ones(len(o)), avi, K, mbest + 1)
        vals.append(tuple(float(x) for x in eval_path(i, path[mbest])))
        ms.append(mbest + 1)
    return vals, ms


def run_blend(lab, Pz, grid=(0., .25, .5)):
    ks = np.unique(lab)
    if len(ks) < 2:
        return None, None
    cent = {g: Pz[lab == g].mean(0) for g in ks}
    nbr = {g: min((h for h in ks if h != g), key=lambda h: np.linalg.norm(cent[g] - cent[h])) for g in ks}

    def kt(j, excl, a):
        mo = others(lab, j, excl)
        mn = [x for x in np.where(lab == nbr[lab[j]])[0] if x != j and x not in excl]
        co = top1(mo, j)
        if a == 0 or not mn:
            return ekey([co // nt]), co % nt
        cn = top1(mn, j)
        a_, b_ = co // nt, cn // nt
        key = ((a_, 1.0),) if a_ == b_ else tuple(sorted(((a_, 1 - a), (b_, a))))
        return key, co % nt

    def score(j, excl, a):
        key, t = kt(j, excl, a)
        return EV.get(j, key)[t]

    jobs = {(j, i, a): kt(j, (i,), a) for i in range(nq) for j in range(nq) if j != i for a in grid}
    EV.get_many([(j, v[0]) for (j, _, _), v in jobs.items()])
    vals, al = [], []
    for i in range(nq):
        inner = [np.nanmean([score(j, (i,), a)[0] for j in range(nq) if j != i]) for a in grid]
        a = grid[int(np.argmax(np.round(inner, 6)))]
        vals.append(tuple(float(x) for x in score(i, (), a)))
        al.append(a)
    return vals, al


def main():
    do_greedy = "--nogreedy" not in sys.argv
    feats = {"cal": prof_matrix(R), "ax3": prof_matrix(R, AX3), "ax5": prof_matrix(R, AX5)}
    res, per, groups = {}, {}, {}

    def rec(name, vals, extra=None):
        per[name] = vals
        res[name] = agg(vals)
        if extra:
            res[name].update(extra)
        a = res[name]
        print(f"{name:26s} bACC {a['bacc']:.3f} AUC {a['auc']:.3f} F1 {a['f1']:.3f} n90 {a['n90']}", flush=True)

    for fs, Pz in feats.items():
        Z = linkage(Pz, "ward")
        Xt = torch.tensor(Pz, device=dev, dtype=torch.float32)
        for algo in ("ward", "km"):
            for k in range(8, 0, -1):
                if algo == "km" and k == 1:
                    continue
                lab = fcluster(Z, k, "maxclust") - 1 if algo == "ward" else kmeans_t(Xt, k)
                nm = f"{fs}_{algo}_k{k}"
                groups[nm] = lab.tolist()
                sizes = sorted(np.bincount(lab).tolist(), reverse=True)
                rec(f"{nm}_top1", run_top1(lab), dict(sizes=sizes))
                if do_greedy and algo == "ward" and k in (8, 6, 4, 3, 2, 1):  # 贪心耗时（~3.5 min/次），只跑合并曲线节点
                    v, ms = run_greedy(lab)
                    rec(f"{nm}_greedy", v, dict(m_hist=np.bincount(ms, minlength=6)[1:].tolist()))
                if algo == "ward" and k >= 2:
                    v, al = run_blend(lab, Pz)
                    rec(f"{nm}_blend", v, dict(a_hist={str(a): al.count(a) for a in sorted(set(al))}))
                json.dump(dict(results=res), open(HERE / "prof_cluster.json", "w", encoding="utf-8"),
                          ensure_ascii=False, indent=1)
    json.dump(dict(results=res), open(HERE / "prof_cluster.json", "w", encoding="utf-8"), ensure_ascii=False,
              indent=1)
    json.dump(groups, open(HERE / "prof_groups.json", "w"), indent=0)
    with open(HERE / "prof_cluster_per_subject.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        names = list(per)
        w.writerow(["subject"] + [f"{n}:{m}" for n in names for m in ("bacc", "auc", "f1")])
        for i, q in enumerate(R["subs"]):
            w.writerow([q] + [round(x, 4) for n in names for x in per[n][i]])


if __name__ == "__main__":
    main()
