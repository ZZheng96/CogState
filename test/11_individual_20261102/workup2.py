"""E25：逐被试细搜（14 档记忆 × 单候选/等权二候选），并给出嵌套口径。

配置空间：候选 ∈ 13 个（8 指数 + slope/cent/bt + own_k + bandmix + glb）；h ∈ {1..6,8,10,12,15,20,25,30,40} 秒；
组合 ∈ 单候选 + 等权二候选平均。阈值固定 0。

两种口径：
  best    在整个序列上取最优（用户所说的"每个个体最适合的方法"，含选择偏差）
  nested  每个留出块只用其余块选配置（不偷看留出块），是"一次定型"的诚实估计

运行：py -3.11 test/11_individual_20261102/workup2.py
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from subject_multi1 import ema

HERE = Path(__file__).resolve().parent
SC = HERE / "scores1"
HS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40)


def main():
    subs = sorted(p.stem.replace("sub-", "") for p in SC.glob("sub-*.npz"))
    out, rows = {}, []
    for q in subs:
        Z = np.load(SC / f"sub-{q}.npz")
        y, blk = Z["y"].astype(int), Z["blk"].astype(int)
        S = {k[7:]: Z[k].astype(float) for k in Z.files if k.startswith("score__")}
        E = {c: {h: ema(v, h) for h in HS} for c, v in S.items()}
        pairs = list(itertools.combinations(sorted(S), 2))
        cfg = [(c, None, h) for c in S for h in HS] + \
              [(a, b, h) for a, b in pairs for h in HS]
        val = {}
        for a, b, h in cfg:
            v = E[a][h] if b is None else 0.5 * (E[a][h] + E[b][h])
            val[(a, b, h)] = v
        pred = {k: (v >= 0).astype(int) for k, v in val.items()}
        best = max(cfg, key=lambda k: balanced_accuracy_score(y, pred[k]))

        nested = np.empty(len(y), int)
        for b in np.unique(blk):
            te, tr = blk == b, blk != b
            kk = max(cfg, key=lambda k: balanced_accuracy_score(y[tr], pred[k][tr]))
            nested[te] = pred[kk][te]

        rec = {"best_key": f"{best[0]}|{best[1]}|h{best[2]}",
               "best_bACC": float(balanced_accuracy_score(y, pred[best])),
               "best_F1": float(f1_score(y, pred[best])),
               "best_AUC": float(roc_auc_score(y, val[best])),
               "nested_bACC": float(balanced_accuracy_score(y, nested)),
               "nested_F1": float(f1_score(y, nested))}
        out[q] = rec
        rows.append((q, rec["best_bACC"], rec["best_key"], rec["nested_bACC"]))
        print(f"{q:4s} best {rec['best_bACC']:.3f} ({rec['best_key']}) "
              f"nested {rec['nested_bACC']:.3f}", flush=True)

    b = np.array([r[1] for r in rows])
    nst = np.array([r[3] for r in rows])
    print(f"\n[best]   mean={b.mean():.3f} median={np.median(b):.3f} "
          f"ge90={int((b >= .9).sum())} ge85={int((b >= .85).sum())}")
    print(f"[nested] mean={nst.mean():.3f} median={np.median(nst):.3f} "
          f"ge90={int((nst >= .9).sum())} ge85={int((nst >= .85).sum())}")
    out["summary"] = {"n": len(rows), "best_mean": float(b.mean()),
                      "best_ge90": int((b >= .9).sum()), "best_ge85": int((b >= .85).sum()),
                      "nested_mean": float(nst.mean()),
                      "nested_ge90": int((nst >= .9).sum()),
                      "nested_ge85": int((nst >= .85).sum()),
                      "n_cfg": len(cfg)}
    (HERE / "workup2.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    print("[done] workup2.json")


if __name__ == "__main__":
    main()
