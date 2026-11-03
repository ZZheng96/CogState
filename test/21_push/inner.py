"""E21：逐人在校准块内层选参（学习阶段），测试段只报告。

内层两折：本人块 (0,1) 训练 → (2,3) 验证，及反向；训练集 = 其他被试全部帧 + 本人内层训练块。
内层特征排序只有一对块可用，按单对 |d| 排序；最终模型（全部校准块、一致性排序）的分数复用
E19 xblock2_scores.npz（同一配置族 k×wo，C=1e-2）。
选择方式：best1（内层 bACC 最高的单配置）、top3 集成、全 8 集成（=E19 基线）。
另记录内层 AUC，用于检测 042 类方向翻转。
运行：.venv\\Scripts\\python.exe test/21_push/inner.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "19_xblock"))
from xblock import L, WS, dev, fit_wlr, dval, z  # noqa: E402
from xblock2 import KS, WOS, C  # noqa: E402

T0 = torch.tensor(0.0, device=dev)


def main():
    subs, D = L.load()
    subs = [q for q in subs if len(np.unique(D[q][2])) == 6]
    F = None
    for w in WS:
        Fw = L.feats(D, subs, w)
        F = Fw if F is None else {q: torch.cat([F[q], Fw[q]], 1) for q in subs}
    Xall = torch.cat([F[q] for q in subs])
    yall = np.concatenate([D[q][1] for q in subs])
    sid = np.concatenate([[i] * len(D[q][1]) for i, q in enumerate(subs)])
    SC = np.load(HERE.parent / "19_xblock" / "xblock2_scores.npz")
    keys = [f"k{k}_w{wo}" for wo in WOS for k in KS]
    R = {}
    for i, q in enumerate(subs):
        y, blk = D[q][1], D[q][2]
        X = F[q]
        oth = torch.tensor(sid != i, device=dev)
        Xo, yo = Xall[oth], yall[sid != i]
        inner = {k: [] for k in keys}
        for trb, vab in (((0, 1), (2, 3)), ((2, 3), (0, 1))):
            tr = np.isin(blk, trb)
            va = np.isin(blk, vab)
            tr_t = torch.tensor(tr, device=dev)
            order = torch.argsort(dval(X[tr], y[tr]).abs(), descending=True)
            yj = np.concatenate([yo, y[tr]])
            for wo in WOS:
                sw = np.concatenate([np.full(len(yo), (1 - wo) / len(yo)), np.full(tr.sum(), wo / tr.sum())])
                for k in KS:
                    c = order[:k]
                    s = z(fit_wlr(torch.cat([Xo[:, c], X[tr][:, c]]), yj, sw, C)(X[:, c]), tr_t)
                    inner[f"k{k}_w{wo}"].append(L.subj_metrics(y[va], s[va], T0))
        ib = {k: float(np.mean([m["bacc"] for m in v])) for k, v in inner.items()}
        ia = {k: float(np.mean([m["auc"] for m in v])) for k, v in inner.items()}
        fold_auc = [float(np.mean([inner[k][f]["auc"] for k in keys])) for f in (0, 1)]
        tst = blk >= 4
        S = {k: torch.tensor(SC[f"{q}|{k}"], device=dev) for k in keys}
        rank = sorted(keys, key=lambda k: -ib[k])
        sel = {"ens8": keys, "best1": rank[:1], "top3": rank[:3]}
        res = {}
        for name, ks in sel.items():
            s = torch.stack([S[k] for k in ks]).mean(0)
            res[name] = L.subj_metrics(y[tst], s[tst], T0)
        R[q] = dict(test=res, inner_bacc=ib, inner_auc=ia, fold_auc=fold_auc, best=rank[0])
        print(q, f"inner best {rank[0]} {ib[rank[0]]:.2f} foldAUC {fold_auc[0]:.2f}/{fold_auc[1]:.2f} |",
              " ".join(f"{n}={v['auc']:.2f}/{v['bacc']:.2f}" for n, v in res.items()), flush=True)
    tab = {}
    for n in ("ens8", "best1", "top3"):
        tab[n] = {m: float(np.mean([R[q]["test"][n][m] for q in subs])) for m in ("bacc", "auc", "f1")}
        tab[n]["n90"] = int(sum(R[q]["test"][n]["bacc"] >= .9 for q in subs))
        print(f"  {n:6s} bACC {tab[n]['bacc']:.3f} AUC {tab[n]['auc']:.3f} F1 {tab[n]['f1']:.3f} n90 {tab[n]['n90']}")
    json.dump(dict(table=tab, per_subject=R), open(HERE / "inner.json", "w"), indent=1)


if __name__ == "__main__":
    main()
