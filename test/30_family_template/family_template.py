"""E30 家族模板：同一原理族的各组配方合成一个候选集，每人只用本人校准块（0–3）挑配方，测试块 4–5 评估。
候选 = 族内各组的原配方（cfg 或 z 双 cfg + 因果阈值），原配方在候选集内，成败取决于校准选择。
选择规则（对所有人相同）：
  mean  = 校准块 0–3 整体 bACC 最大
  minb  = 块对 (0,1)(2,3)(0,3)(2,1) bACC 的最小值最大（稳定性；块按类交替，单块只有一类）
  last  = 块对 (2,3) bACC 最大（离测试最近）
只算族内成员 × 族内配方，不做全量搜索。
运行：python -u family_template.py [A|B|C] → family_<F>.log / family_<F>.json
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "29_closest_pair"))
from e29lib import GROUPS, R, THR2, TH, base, cfg_name, dev, ev2b, metrics, sd, slabel, subs  # noqa: E402

FAM = {"A": [0, 2, 3, 6, 7, 9, 10, 11, 13, 16], "B": [1, 4, 8, 12, 14, 17], "C": [5, 15]}
F = sys.argv[1] if len(sys.argv) > 1 else "A"
G = json.load(open(HERE.parent / "26_merge2" / "merge2.json", encoding="utf-8"))["groups"]
NAME2C = {}
for c in range(ev2b.shape[1]):
    n = cfg_name(c, base)
    NAME2C[n] = c
    if "+" in n:
        a, b = n.split("+")
        NAME2C[f"{b}+{a}"] = c
TNAME = [t[0] for t in THR2]
log = open(HERE / f"family_{F}.log", "w", encoding="utf-8")


def pr(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.write(s + "\n")
    log.flush()


def recipe_of(g):
    lab = g["recipe"]
    if lab.startswith("专属"):  # excl.json 只有测试指标、无分数序列，无法在校准块上评分 → 不进候选
        return None
    body, thr = [x.strip() for x in lab.rsplit("|", 1)]
    body = body[2:-1] if body.startswith("z[") else body
    cs = [NAME2C[p.strip()] for p in body.split(";")]
    return lab, cs, TNAME.index(thr)


def score(C, cal_t, cs):
    M = C[cs].sum(0, keepdim=True)
    if len(cs) == 1:
        return M
    return (M - M[:, cal_t].mean(1, keepdim=True)) / (M[:, cal_t].std(1, keepdim=True) + 1e-6)


gi = FAM[F]
REC, RG = [], []
for i in gi:
    r = recipe_of(G[i])
    if r:
        REC.append(r)
        RG.append(i)
U = [j for i in gi for j in GROUPS[i]]
home = {j: (RG.index(i) if i in RG else -1) for i in gi for j in GROUPS[i]}
pr(f"家族 {F}：组 {gi}，成员 {len(U)} 人，候选配方 {len(REC)}（专属配方组 {[i for i in gi if i not in RG]} 无分数序列，"
   f"其成员只用族内其他配方，home=-1）")
for k, (lab, cs, t) in enumerate(REC):
    pr(f"  R{k} (G{RG[k]}) {lab}")

t0 = time.time()
out = []
RULES = ("mean", "minb", "last")
for n, j in enumerate(U):
    q = subs[j]
    C, cal_t, tst, yy = sd(j)
    y, blk = R["D"][q][1], R["D"][q][2]
    tb = []  # (候选, 3) 测试 bACC/AUC/F1
    cb = []  # (候选, 5) 校准：整体 + 块 0..3
    for lab, cs, t in REC:
        M = score(C, cal_t, cs)
        tb.append(metrics(M, tst, yy)[0, t].cpu().numpy())
        row = []
        for sel in ([0, 1, 2, 3], [0, 1], [2, 3], [0, 3], [2, 1]):  # 块交替单类：0/2 负、1/3 正
            m = np.isin(blk, sel)
            if m.sum() == 0 or len(np.unique(y[m])) < 2:
                row.append(np.nan)
                continue
            ti = torch.tensor(np.where(m)[0], device=dev)
            row.append(float(metrics(M, ti, torch.tensor(y[m], device=dev).bool(), auc=False)[0, t, 0]))
        cb.append(row)
    tb, cb = np.array(tb), np.array(cb)
    allnan = np.isnan(cb[:, 1:]).all()
    crit = {"mean": cb[:, 0], "minb": cb[:, 0] if allnan else np.nanmin(cb[:, 1:], 1),
            "last": cb[:, 0] if np.isnan(cb[:, 2]).all() else cb[:, 2]}
    sel = {r: int(np.nanargmax(np.nan_to_num(v, nan=-1))) for r, v in crit.items()}
    pr(f"\n[{n + 1}/{len(U)}] {q} 原组 R{home[j]}  测试最优 R{int(tb[:, 0].argmax())} {tb[:, 0].max():.3f}  [{time.time() - t0:.0f}s]")
    for k in range(len(REC)):
        flag = "".join(r[0] for r in RULES if sel[r] == k)
        pr(f"   R{k}: test {tb[k, 0]:.3f}  cal {cb[k, 0]:.3f}  blk " +
           " ".join("  nan" if np.isnan(x) else f"{x:.3f}" for x in cb[k, 1:]) +
           f"{'  *原' if k == home[j] else ''}{'  <选:' + flag if flag else ''}")
    out.append(dict(subject=q, home=home[j], test=tb.tolist(), cal=cb.tolist(), sel=sel))

json.dump(dict(family=F, groups=gi, recipes=[r[0] for r in REC], subjects=out),
          open(HERE / f"family_{F}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
pr("\n汇总（达标 = 测试 bACC ≥ .9）：")
for r in RULES + ("home", "oracle"):
    v = []
    for o in out:
        tb = np.array(o["test"])
        if r == "home" and o["home"] < 0:
            continue
        k = o["home"] if r == "home" else int(tb[:, 0].argmax()) if r == "oracle" else o["sel"][r]
        v.append(tb[k])
    v = np.array(v)
    pr(f"  {r:6s} 达标 {int((v[:, 0] >= TH).sum())}/{len(v)}  bACC {v[:, 0].mean():.3f}  AUC {np.nanmean(v[:, 1]):.3f}"
       f"  F1 {v[:, 2].mean():.3f}")
pr("done")
