"""E18b：逐人在校准段内选方法（学习阶段），测试段单窗 6 s 因果推理。

校准段 = 块 0–3（R,T,R,T，带标签）；测试段 = 块 4–5。
内层：块(0,1)→验证(2,3) 与 (2,3)→(0,1) 两折，按 bACC 选该人配置；再用全部校准块重训，测试段评估。
候选：seed（跨被试 LR）/ own LR（尺度子集 × C）/ seed+β·own。z 标准化与阈值 0 均只用训练块统计量。
另报 oracle（测试段最优配置）作上限诊断，不作为成绩。
运行：.venv\\Scripts\\python.exe test/18_calib/calib2.py
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
NF = 878
SUBSETS = {"all": list(range(5)), "w60": [0], "w300": [2], "wInf": [4], "w60_300": [0, 2], "w300_inf": [2, 3, 4]}
COWN = [1e-4, 1e-3, 1e-2, 1e-1]
BETAS = [0.5, 1, 2, 1e3]


def cols(sub):
    return torch.cat([torch.arange(i * NF, (i + 1) * NF) for i in SUBSETS[sub]]).to(L.dev)


def z(s, ref):
    return (s - s[ref].mean()) / (s[ref].std() + 1e-6)


def scores(Fq, y, tr, s0):
    """训练块 tr 上拟合所有候选，返回 {cfg: 全帧分数(z 用 tr 统计量)}。"""
    out = {"seed": z(s0, tr)}
    z0 = out["seed"]
    for sub in SUBSETS:
        X = Fq[:, cols(sub)]
        for C in COWN:
            zo = z(L.fit_lr(X[tr], y[tr], C)(X), tr)
            for b in BETAS:
                out[f"{sub}_C{C}_b{b}"] = zo if b >= 1e3 else z0 + b * zo
    return out


def bacc(y, s):
    p = (s > 0).cpu().numpy()
    return 0.5 * (p[y == 1].mean() + 1 - p[y == 0].mean())


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
        test = {k: L.subj_metrics(y[tst], s[tst], torch.tensor(0.0, device=L.dev)) for k, s in fin.items()}
        pick = max(inner, key=inner.get)
        orc = max(test, key=lambda k: test[k]["bacc"])
        R[q] = dict(pick=pick, inner=inner[pick], **test[pick], seed=test["seed"]["bacc"],
                    oracle=orc, oracle_bacc=test[orc]["bacc"], test=test, inner_all=inner)
        print(f"{q} pick={pick} inner={inner[pick]:.3f} test bACC={test[pick]['bacc']:.3f} AUC={test[pick]['auc']:.3f} "
              f"| seed {test['seed']['bacc']:.3f} | oracle {orc} {test[orc]['bacc']:.3f}", flush=True)
    m = {k: float(np.mean([R[q][k] for q in subs])) for k in ("bacc", "auc", "f1", "seed", "oracle_bacc")}
    m["n90"] = int(sum(R[q]["bacc"] >= .9 for q in subs))
    m["n90_oracle"] = int(sum(R[q]["oracle_bacc"] >= .9 for q in subs))
    print("SUMMARY", m, flush=True)
    json.dump(dict(summary=m, per_subject=R), open(HERE / "calib2.json", "w"), indent=1)


if __name__ == "__main__":
    main()
