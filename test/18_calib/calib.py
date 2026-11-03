"""E18：决赛口径——学习阶段用本人带标签校准块，推理阶段单窗 6 s 因果 + 无标签在线自适应。

每人：前 K 个块（时间顺序）= 校准段（有标签，模拟决赛调试期采集）；其余块 = 流式测试段。
特征：sliding1 878 维，多尺度因果 running 标准化（过去帧的无标签统计量，细则（2）在线自适应）。
模型：seed = 跨被试 LR（不含本人，5 折）；own = 本人校准段 LR；score = seed_z + beta*own_z（z 用校准段统计量）。
超参 (C_own, beta, K) 只在其他被试上按宏 bACC 选（留一被试嵌套），被评被试测试段不参与选择。
运行：.venv\\Scripts\\python.exe test/18_calib/calib.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "17_compliant"))
import loso as L  # noqa: E402

WS = [60, 120, 300, 600, 99999]
KS = [2, 4]
COWN = [1e-4, 1e-3, 1e-2]
BETAS = [0, 0.5, 1, 2, 4, 1e3]


def main():
    subs, D = L.load()
    subs = [q for q in subs if len(np.unique(D[q][2])) == 6]  # 034 仅 3 块，无法校准→测试
    F = None
    for w in WS:
        Fw = L.feats(D, subs, w)
        F = Fw if F is None else {q: torch.cat([F[q], Fw[q]], 1) for q in subs}
    perm = list(np.random.RandomState(0).permutation(subs))
    outer = [perm[k::5] for k in range(5)]
    seed = {}
    for te in outer:
        tr = [q for q in subs if q not in te]
        f = L.fit_lr(torch.cat([F[q] for q in tr]), np.concatenate([D[q][1] for q in tr]), 1e-3)
        for q in te:
            seed[q] = f(F[q])
        print("seed fold done", flush=True)
    res = {}  # res[(K,C,beta)][q] = metrics
    for q in subs:
        y, blk = D[q][1], D[q][2]
        _, first = np.unique(blk, return_index=True)
        order = blk[np.sort(first)]
        blk = np.searchsorted(order, blk) if (np.diff(order) > 0).all() else np.array([list(order).index(b) for b in blk])
        for K in KS:
            cal = blk < K
            tst = ~cal
            s0 = seed[q]
            z0 = (s0 - s0[cal].mean()) / (s0[cal].std() + 1e-6)
            for C in COWN:
                fo = L.fit_lr(F[q][cal], y[cal], C)
                so = fo(F[q])
                zo = (so - so[cal].mean()) / (so[cal].std() + 1e-6)
                for b in BETAS:
                    s = zo if b >= 1e3 else z0 + b * zo
                    m = L.subj_metrics(y[tst], s[tst], torch.tensor(0.0, device=L.dev))
                    res.setdefault(f"K{K}_C{C}_b{b}", {})[q] = m
        print(q, " ".join(f"{k}:{res[k][q]['bacc']:.3f}" for k in res if k.startswith("K2_C0.001")), flush=True)
    keys = list(res)
    summ = {k: {m: float(np.mean([res[k][q][m] for q in subs])) for m in ("bacc", "auc", "f1")} |
            {"n90": int(sum(res[k][q]["bacc"] >= .9 for q in subs))} for k in keys}
    for k in sorted(summ, key=lambda k: -summ[k]["bacc"])[:12]:
        print("CFG", k, summ[k], flush=True)
    nested = {}
    for q in subs:
        others = [p for p in subs if p != q]
        best = max(keys, key=lambda k: np.mean([res[k][p]["bacc"] for p in others]))
        nested[q] = dict(cfg=best, **res[best][q])
    ns = {m: float(np.mean([nested[q][m] for q in subs])) for m in ("bacc", "auc", "f1")}
    ns["n90"] = int(sum(nested[q]["bacc"] >= .9 for q in subs))
    print("NESTED", ns, flush=True)
    for q in sorted(subs, key=lambda q: -nested[q]["bacc"]):
        print(q, nested[q]["cfg"], f"{nested[q]['bacc']:.3f} {nested[q]['auc']:.3f}")
    json.dump(dict(res=res, summary=summ, nested=nested, nested_summary=ns), open(HERE / "calib.json", "w"), indent=1)


if __name__ == "__main__":
    main()
