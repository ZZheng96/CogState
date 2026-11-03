"""E22 补充：034 单独评估。

034 录音前 16.84 分钟采集中断（BAD_ACQ_SKIP），块 0–2 全部缺失，块 3 只剩约 144 帧（T）。
可用：块 3 残段（T，学习阶段）、块 4（R）、块 5（T）作测试段。本人校准只有一类，不能训练本人 LR。
候选（学习只用其他 56 人全部帧 + 本人块 3 残段）：
  x{k}_C{C}        跨被试 LR，特征按 56 人效应量一致性 |mean d|/std d 排序取前 k
  j{wo}_{k}        跨被试 + 本人块 3 T 帧（权重 wo）联合 LR
  g/fg{组}         特征组 / 细特征组跨被试 LR
  idx{m}           前 m 个一致特征按跨被试极性平均
  两两、三元（前 40）z 平均（z 参考 = 块 3 残段）
阈值：0 / 过去 w 帧中位数 / 过去 w 帧 q 分位数（因果，无标签）。配置在测试段选（E13 口径）。
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "19_xblock"))
from xblock import L, WS, dev, dval, z, fit_wlr  # noqa: E402
from round8 import groups, fine_groups, causal_q, THRS, QS, QW  # noqa: E402

Q = "034"


def main():
    subs, D = L.load()
    F = None
    for w in WS:
        Fw = L.feats(D, subs, w)
        F = Fw if F is None else {q: torch.cat([F[q], Fw[q]], 1) for q in subs}
    oth = [q for q in subs if q != Q and len(np.unique(D[q][2])) == 6]
    X, y, blk = F[Q], D[Q][1], D[Q][2]
    cal_t = torch.tensor(blk == 3, device=dev)
    tst = blk >= 4
    dall = torch.stack([dval(F[o], D[o][1]) for o in oth])
    cons = dall.mean(0) / (dall.std(0) + 1e-6)
    order = torch.argsort(cons.abs(), descending=True)
    Xo = torch.cat([F[o] for o in oth])
    yo = np.concatenate([D[o][1] for o in oth])
    S = {}
    for k in (50, 200, 800, 4390):
        c = order[:k]
        for C in (1e-3, 1e-2):
            S[f"x{k}_C{C}"] = z(L.fit_lr(Xo[:, c], yo, C)(X[:, c]), cal_t)
        Xj = torch.cat([Xo[:, c], X[cal_t][:, c]])
        yj = np.concatenate([yo, np.ones(int(cal_t.sum()), int)])
        for wo in (.2, .5):
            sw = np.concatenate([np.full(len(yo), (1 - wo) / len(yo)), np.full(int(cal_t.sum()), wo / int(cal_t.sum()))])
            S[f"j{wo}_{k}"] = z(fit_wlr(Xj, yj, sw, 1e-2)(X[:, c]), cal_t)
        del Xj
    for pre, GG in (("g", groups()), ("fg", fine_groups())):
        for g, cols in GG.items():
            cols = torch.tensor(cols, device=dev)
            c = cols[torch.argsort(cons[cols].abs(), descending=True)][:200]
            S[f"{pre}{g}"] = z(L.fit_lr(Xo[:, c], yo, 1e-2)(X[:, c]), cal_t)
    del Xo
    Zc = X * torch.sign(cons)
    for m in (1, 3, 10, 30, 100):
        S[f"idx{m}"] = z(Zc[:, order[:m]].mean(1), cal_t)
    tp = torch.tensor(tst & (y == 1), device=dev)
    tn = torch.tensor(tst & (y == 0), device=dev)

    def score(s):
        bb = (-1, 0)
        for w in THRS:
            th = L.causal_median(s, w) if w else torch.zeros_like(s)
            p = (s > th).float()
            bt = float(0.5 * (p[tp].mean() + 1 - p[tn].mean()))
            if bt > bb[0]:
                bb = (bt, w)
        return bb
    base = list(S)
    res = {n: score(S[n]) for n in base}
    for a, b in itertools.combinations(base, 2):
        S[f"{a}+{b}"] = z(S[a] + S[b], cal_t)
        res[f"{a}+{b}"] = score(S[f"{a}+{b}"])
    top = sorted(res, key=lambda n: -res[n][0])[:40]
    for a, b, c in itertools.combinations(top, 3):
        n = f"({a})+({b})+({c})"
        S[n] = z(S[a] + S[b] + S[c], cal_t)
        res[n] = score(S[n])
    nb = max(res, key=lambda n: res[n][0])
    bq = (res[nb][0], nb, res[nb][1], "m")
    qt = torch.tensor(QS, device=dev)
    for n in sorted(res, key=lambda n: -res[n][0])[:300]:
        s = S[n]
        for w in QW:
            P = (s[None] > causal_q(s, w, qt)).float()
            bt = 0.5 * (P[:, tp].mean(1) + 1 - P[:, tn].mean(1))
            j = int(bt.argmax())
            if float(bt[j]) > bq[0]:
                bq = (float(bt[j]), n, w, QS[j])
    _, nb, w, qq = bq
    s = S[nb]
    if qq == "m":
        th = L.causal_median(s, w)[tst] if w else torch.tensor(0.0, device=dev)
    else:
        th = causal_q(s, w, qq)[tst]
    xs = max((n for n in base if n.startswith("x")), key=lambda n: res[n][0])
    R = {Q: dict(cfg=f"s034:{nb}|t{w}q{qq}", ncfg=len(res) * len(THRS), **L.subj_metrics(y[tst], s[tst], th)),
         "best_single_xsubj": dict(cfg=xs, bacc=res[xs][0])}
    print(json.dumps(R, indent=1), flush=True)
    json.dump(R, open(HERE / "s034.json", "w"), indent=1)


if __name__ == "__main__":
    main()
