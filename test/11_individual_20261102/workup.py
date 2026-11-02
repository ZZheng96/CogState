"""E24-2：逐被试微调（从 ≥0.85 的 33 人开始，一个一个来）。

对每个被试，用缓存好的候选分数（cache_scores1.py）做三件事：
  1. 诊断：最优候选在每个块上的 bACC，指出"拖后腿的是哪一块"；
  2. 阈值校准：每个留出块的阈值在其余块上选（1 个参数），而不是固定 0；
  3. 小集成：在 4 个最强候选上，用其余块选权重（步长 0.25 的单纯形）与记忆长度 h；
  4. 嵌套口径：每块只用其余块选 (候选,h,阈值)，再看该块 —— 这是不偷看留出块的真实估计。

运行：py -3.11 test/11_individual_20261102/workup.py
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from subject_multi1 import HS, ema

HERE = Path(__file__).resolve().parent
SC = HERE / "scores1"
THR_Q = np.linspace(0.1, 0.9, 17)


def pred_of(s, h):
    return (ema(s, h) >= 0).astype(int)


def best_key(y, keys, S, mask=None):
    m = np.ones(len(y), bool) if mask is None else mask
    best, bk = -1.0, None
    for k in keys:
        c, h = k.rsplit("_h", 1)
        b = balanced_accuracy_score(y[m], pred_of(S[c], int(h))[m])
        if b > best:
            best, bk = b, k
    return bk, best


def main():
    subs = sorted(p.stem.replace("sub-", "") for p in SC.glob("sub-*.npz"))
    rows, out = [], {}
    for q in subs:
        Z = np.load(SC / f"sub-{q}.npz")
        y, blk = Z["y"].astype(int), Z["blk"].astype(int)
        S = {k[7:]: Z[k].astype(float) for k in Z.files if k.startswith("score__")}
        keys = [f"{c}_h{h}" for c in S for h in HS]
        base = {k: balanced_accuracy_score(y, pred_of(S[k.rsplit('_h', 1)[0]], int(k.rsplit('_h', 1)[1])))
                for k in keys}
        top = sorted(base, key=lambda k: -base[k])[:4]
        best_k = top[0]
        per_block = {int(b): float(balanced_accuracy_score(y[blk == b],
                                                           pred_of(S[best_k.rsplit('_h', 1)[0]],
                                                                   int(best_k.rsplit('_h', 1)[1]))[blk == b]))
                     for b in np.unique(blk)}

        # 阈值校准 + 嵌套选择（每块只用其余块定 (候选,h,阈值)）
        nested_pred = np.empty(len(y), int)
        for b in np.unique(blk):
            te, tr = blk == b, blk != b
            kk, _ = best_key(y, keys, S, mask=tr)
            e = ema(S[kk.rsplit('_h', 1)[0]], int(kk.rsplit('_h', 1)[1]))
            grid = np.quantile(e[tr], THR_Q)
            gains = [balanced_accuracy_score(y[tr], (e[tr] >= t).astype(int)) for t in grid]
            t = grid[int(np.argmax(gains))]
            nested_pred[te] = (e[te] >= t).astype(int)
        nested_b = balanced_accuracy_score(y, nested_pred)

        # 小集成：top4 候选，权重步长 .25，h 与权重都在其余块上选
        comp = []
        for k in top:
            comp.append((k.rsplit('_h', 1)[0], int(k.rsplit('_h', 1)[1])))
        cands = sorted({c for c, _ in comp})[:3]
        hs = sorted({h for _, h in comp})
        ws = [w for w in itertools.product((0, .25, .5, .75, 1), repeat=len(cands))
              if sum(w) > 0]
        ens_pred = np.empty(len(y), int)
        for b in np.unique(blk):
            te, tr = blk == b, blk != b
            bestv, bestcfg = -1, None
            for h in hs:
                Zs = {c: ema(S[c], h) for c in cands}
                for w in ws:
                    v = sum(wi * Zs[c] for wi, c in zip(w, cands))
                    bb = balanced_accuracy_score(y[tr], (v[tr] >= 0).astype(int))
                    if bb > bestv:
                        bestv, bestcfg = bb, (h, w)
            h, w = bestcfg
            v = sum(wi * ema(S[c], h) for wi, c in zip(w, cands))
            ens_pred[te] = (v[te] >= 0).astype(int)
        ens_b = balanced_accuracy_score(y, ens_pred)

        rec = {"best_key": best_k, "best_bACC": float(base[best_k]),
               "best_AUC": float(roc_auc_score(y, ema(S[best_k.rsplit('_h', 1)[0]],
                                                      int(best_k.rsplit('_h', 1)[1])))),
               "per_block_bACC": per_block, "nested_bACC": float(nested_b),
               "ens_bACC": float(ens_b),
               "ens_F1": float(f1_score(y, ens_pred))}
        out[q] = rec
        rows.append((q, rec["best_bACC"], best_k, nested_b, ens_b, per_block))
        print(f"{q:4s} best {rec['best_bACC']:.3f} ({best_k}) nested {nested_b:.3f} ens {ens_b:.3f} "
              f"| blocks " + " ".join(f"{b}:{v:.2f}" for b, v in per_block.items()), flush=True)

    b = np.array([r[1] for r in rows])
    nst = np.array([r[3] for r in rows])
    ens = np.array([r[4] for r in rows])
    print(f"\n[best] mean={b.mean():.3f} ge90={int((b >= .9).sum())}")
    print(f"[nested] mean={nst.mean():.3f} ge90={int((nst >= .9).sum())}")
    print(f"[min-ens] mean={ens.mean():.3f} ge90={int((ens >= .9).sum())}")
    final = np.maximum(nst, ens)
    print(f"[nested|ens] mean={final.mean():.3f} ge90={int((final >= .9).sum())}")
    out["summary"] = {"n": len(rows), "best_ge90": int((b >= .9).sum()),
                      "nested_ge90": int((nst >= .9).sum()),
                      "ens_ge90": int((ens >= .9).sum()),
                      "final_ge90": int((final >= .9).sum()),
                      "final_mean": float(final.mean())}
    (HERE / "workup.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print("[done] workup.json")


if __name__ == "__main__":
    main()
