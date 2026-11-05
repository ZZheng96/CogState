"""E30b 家族 A：把"阈值"从个人选择里拿出来，全族共用一个漂移自适应阈值；个人只在校准块上选特征 cfg。
E30 诊断：选错主要是校准块偏爱静态阈值、测试块上自适应阈值才对。所以：
  (a) 阈值 T 全族统一（85 种中逐一列出），每人用校准块 0–3 bACC（同一 T）选 cfg 组合（族内 6 个 + 族均值）；
  (b) 全族同一配方：族内 6 个 cfg 组合的 z 均值 + 统一 T，不做个人选择。
只算族内 33 人 × 7 个候选 × 85 阈值。
运行：python -u fam_thr.py [A] → fam_thr_<F>.log / .json
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch

from ft_common import F, HERE, R, REC, THR2, TH, U, dev, metrics, sd, subs

log = open(HERE / f"fam_thr_{F}.log", "w", encoding="utf-8")


def pr(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.write(s + "\n")
    log.flush()


def z(M, cal_t):
    return (M - M[:, cal_t].mean(1, keepdim=True)) / (M[:, cal_t].std(1, keepdim=True) + 1e-6)


NR = len(REC)
TE = np.zeros((len(U), NR + 1, len(THR2), 3))  # 测试 bACC/AUC/F1
CA = np.zeros((len(U), NR + 1, len(THR2)))  # 校准 bACC
t0 = time.time()
for n, j in enumerate(U):
    q = subs[j]
    C, cal_t, tst, yy = sd(j)
    y, blk = R["D"][q][1], R["D"][q][2]
    rows = [z(C[cs].sum(0, keepdim=True), cal_t) for _, cs, _ in REC]
    M = torch.cat(rows)
    M = torch.cat([M, z(M.mean(0, keepdim=True), cal_t)])
    TE[n] = metrics(M, tst, yy).cpu().numpy()
    m = blk <= 3
    ci = torch.tensor(np.where(m)[0], device=dev)
    CA[n] = metrics(M, ci, torch.tensor(y[m], device=dev).bool(), auc=False)[..., 0].cpu().numpy()
    print(f"[{n + 1}/{len(U)}] {q} [{time.time() - t0:.0f}s]", flush=True)

TE, CA = np.nan_to_num(TE, nan=0.), np.nan_to_num(CA, nan=-1.)
pr(f"家族 {F}：{len(U)} 人，候选 R0..R{NR - 1} + 族均值 R{NR}")
res = []
for t, (tn, *_) in enumerate(THR2):
    k = CA[:, :, t].argmax(1)  # (a) 校准选 cfg
    a = TE[np.arange(len(U)), k, t]
    b = TE[:, NR, t]  # (b) 族均值
    o = TE[:, :, t, 0].max(1)  # 统一 T 下 cfg oracle
    res.append((tn, int((a[:, 0] >= TH).sum()), a.mean(0), int((b[:, 0] >= TH).sum()), b.mean(0), int((o >= TH).sum())))
res.sort(key=lambda r: -max(r[1], r[3]))
pr("\n阈值      (a)校准选cfg 达标/bACC/AUC/F1        (b)族均值 达标/bACC/AUC/F1        统一T的cfg-oracle")
for tn, na, ma, nb, mb, no in res[:15]:
    pr(f"  {tn:9s} {na:2d}/{len(U)} {ma[0]:.3f} {ma[1]:.3f} {ma[2]:.3f}    {nb:2d}/{len(U)} {mb[0]:.3f} {mb[1]:.3f} {mb[2]:.3f}    {no:2d}")
best = res[0]
t = [x[0] for x in THR2].index(best[0])
k = CA[:, :, t].argmax(1)
pr(f"\n最优统一阈值 {best[0]} 下逐人（(a) 校准选 cfg）：")
for n, j in enumerate(U):
    x = TE[n, k[n], t]
    pr(f"  {subs[j]}: 选 R{k[n]}  bACC {x[0]:.3f} AUC {x[1]:.3f} F1 {x[2]:.3f}  | 族均值 {TE[n, NR, t, 0]:.3f}"
       f"  | 该T最优 R{TE[n, :, t, 0].argmax()} {TE[n, :, t, 0].max():.3f}{'  <.9' if x[0] < TH else ''}")
json.dump(dict(top=[dict(thr=r[0], a=r[1], a_mean=list(r[2]), b=r[3], b_mean=list(r[4]), oracle=r[5]) for r in res]),
          open(HERE / f"fam_thr_{F}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
pr("done")
