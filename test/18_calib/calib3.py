"""E18c：稳健选择 + 推理阶段无标签阈值自适应。

协议同 calib2：块 0–3 为校准（带标签，学习阶段），块 4–5 为测试（推理阶段，单窗 6 s 因果）。
选择策略（每人只用校准块内层两折结果）：
  seed        跨被试 LR
  pick        内层 bACC 最优配置（= calib2）
  marg{m}     内层最优比 seed 高出 m 才换，否则 seed
  ens{k}      内层前 k 个配置 + seed 的 z 分数平均
阈值（推理阶段，无标签、因果）：
  t0          阈值 0（校准块统计量）
  med{w}      过去 w 帧（含当前，含校准尾部）分数的中位数；只更新阈值参数，不平滑分数
全局策略×阈值组合用留一被试选择（只看其他被试结果），被评被试不参与。
运行：.venv\\Scripts\\python.exe test/18_calib/calib3.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "17_compliant"))
sys.path.insert(0, str(HERE))
import loso as L  # noqa: E402
from calib2 import WS, scores, bacc  # noqa: E402

TWS = [0, 300, 600, 900]
MARG = [0.02, 0.05, 0.1]
ENS = [3, 5, 10]


def strategies(inner, fin):
    order = sorted(inner, key=inner.get, reverse=True)
    best = order[0]
    out = {"seed": fin["seed"], "pick": fin[best]}
    for m in MARG:
        out[f"marg{m}"] = fin[best] if inner[best] - inner["seed"] > m else fin["seed"]
    for k in ENS:
        top = [c for c in order if c != "seed"][:k]
        out[f"ens{k}"] = torch.stack([fin[c] for c in top + ["seed"]]).mean(0)
    return out


def main():
    subs, D = L.load()
    subs = [q for q in subs if len(np.unique(D[q][2])) == 6]
    F = None
    for w in WS:
        Fw = L.feats(D, subs, w)
        F = Fw if F is None else {q: torch.cat([F[q], Fw[q]], 1) for q in subs}
    perm = list(np.random.RandomState(0).permutation(subs))
    seed = {}
    for k in range(5):
        te = perm[k::5]
        tr = [q for q in subs if q not in te]
        f = L.fit_lr(torch.cat([F[q] for q in tr]), np.concatenate([D[q][1] for q in tr]), 1e-3)
        for q in te:
            seed[q] = f(F[q])
    R = {}
    for q in subs:
        y, blk = D[q][1], D[q][2]
        a, b_, cal, tst = blk <= 1, (blk >= 2) & (blk <= 3), blk <= 3, blk >= 4
        inner = {}
        for tr, va in ((a, b_), (b_, a)):
            for k, s in scores(F[q], y, tr, seed[q]).items():
                inner.setdefault(k, []).append(bacc(y[va], s[va]))
        inner = {k: float(np.mean(v)) for k, v in inner.items()}
        fin = scores(F[q], y, cal, seed[q])
        R[q] = {}
        for sn, s in strategies(inner, fin).items():
            for tw in TWS:
                thr = L.causal_median(s, tw)
                R[q][f"{sn}|t{tw}"] = L.subj_metrics(y[tst], s[tst], thr[tst])
        print(q, " ".join(f"{k}={R[q][k]['bacc']:.3f}" for k in ("seed|t0", "pick|t0", "ens5|t0", "ens5|t600")), flush=True)
    combos = list(R[subs[0]])
    tab = {c: {m: float(np.mean([R[q][c][m] for q in subs])) for m in ("bacc", "auc", "f1")} for c in combos}
    for c in combos:
        tab[c]["n90"] = int(sum(R[q][c]["bacc"] >= .9 for q in subs))
    for c, v in sorted(tab.items(), key=lambda kv: -kv[1]["bacc"])[:12]:
        print(f"  {c:16s} bACC {v['bacc']:.3f} AUC {v['auc']:.3f} F1 {v['f1']:.3f} n90 {v['n90']}")
    # 留一被试选全局组合
    final, chosen = {}, {}
    for q in subs:
        others = [p for p in subs if p != q]
        c = max(combos, key=lambda c: np.mean([R[p][c]["bacc"] for p in others]))
        chosen[q], final[q] = c, R[q][c]
    m = {k: float(np.mean([final[q][k] for q in subs])) for k in ("bacc", "auc", "f1")}
    m["n90"] = int(sum(final[q]["bacc"] >= .9 for q in subs))
    print("LOSO-selected", m, "combos", sorted(set(chosen.values())), flush=True)
    json.dump(dict(summary=m, chosen=chosen, table=tab, per_subject=R), open(HERE / "calib3.json", "w"), indent=1)


if __name__ == "__main__":
    main()
