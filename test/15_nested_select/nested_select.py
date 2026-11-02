"""E15：嵌套口径下的稳健配置选择（外层留块评估，内层只用其余块选配置）。

基于 ../11_individual_20261102/scores1/ 的 14 个候选分数，配置空间同 workup2（单候选/等权二候选 × 14 档 h）。
每块单一类别，所以训练块上的 bACC = 两类各自块平均准确率的均值。

选择规则（全部只看训练块）：
  max      训练 bACC 最大（= workup2 的 nested）
  worst    训练块最差准确率最大（抗"坏块"）
  lcb      均值 - λ·块间标准差
  topK     训练 bACC 前 K 名配置的分数做符号投票（降低选择方差）
  prior    本人训练 bACC 与"其他被试在该配置上的平均 bACC"加权（群体先验，留一被试，不含本人）
运行：py -3.11 test/15_nested_select/nested_select.py
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

HERE = Path(__file__).resolve().parent
SC = HERE.parent / "11_individual_20261102" / "scores1"
HS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40)


def ema(x, h):
    a = 0.5 ** (1.0 / h)
    return lfilter([1 - a], [1, -a], x)


USE_SF2 = "--sf2" in sys.argv
USE_SF = "--sf" in sys.argv or USE_SF2
EXTRAS = [HERE / "scores_sf"] + ([HERE / "scores_sf2"] if USE_SF2 else [])


def load(q):
    Z = np.load(SC / f"sub-{q}.npz")
    y, blk = Z["y"].astype(int), Z["blk"].astype(int)
    S = {k[7:]: Z[k].astype(float) for k in Z.files if k.startswith("score__")}
    for EXTRA in (EXTRAS if USE_SF else []):
        if not (EXTRA / f"sub-{q}.npz").exists():
            continue
        W = np.load(EXTRA / f"sub-{q}.npz")
        assert np.array_equal(W["y"].astype(int), y) and np.array_equal(W["blk"].astype(int), blk)
        S.update({k[7:]: W[k].astype(float) for k in W.files if k.startswith("score__")})
    return y, blk, S


def build(S, cands):
    E = {c: {h: ema(S[c], h) for h in HS} for c in cands}
    keys, V = [], []
    for c in cands:
        for h in HS:
            keys.append((c, None, h)); V.append(E[c][h])
    for a, b in itertools.combinations(cands, 2):
        for h in HS:
            keys.append((a, b, h)); V.append(0.5 * (E[a][h] + E[b][h]))
    return keys, np.vstack(V)          # (n_cfg, T)


def block_acc(P, y, blk, ub):
    # P: (n_cfg, T) 0/1；返回 (n_cfg, n_blk) 块内准确率与块类别
    acc = np.stack([(P[:, blk == b] == y[blk == b][0]).mean(1) for b in ub], 1)
    cls = np.array([y[blk == b][0] for b in ub])
    return acc, cls


def bacc_from(acc, cls, mask):
    a1 = acc[:, mask & (cls == 1)].mean(1)
    a0 = acc[:, mask & (cls == 0)].mean(1)
    return 0.5 * (a0 + a1)


def frame_bacc(y, p):
    return 0.5 * ((p[y == 1] == 1).mean() + (p[y == 0] == 0).mean())


def main():
    subs = sorted(p.stem.replace("sub-", "") for p in SC.glob("sub-*.npz"))
    data = {q: load(q) for q in subs}
    cands = sorted(set.intersection(*[set(d[2]) for d in data.values()]))
    print("cands", cands)
    pre = {}
    for q in subs:
        y, blk, S = data[q]
        keys, V = build(S, cands)
        P = (V >= 0).astype(np.int8)
        ub = np.unique(blk)
        acc, cls = block_acc(P, y, blk, ub)
        pre[q] = dict(y=y, blk=blk, V=V, P=P, ub=ub, acc=acc, cls=cls,
                      full=bacc_from(acc, cls, np.ones(len(ub), bool)))
    full_all = np.vstack([pre[q]["full"] for q in subs])  # (n_sub, n_cfg)

    rules = {"max": None, "worst": None}
    for lam in (0.5, 1.0):
        rules[f"lcb{lam}"] = lam
    for K in (5, 20, 50):
        rules[f"top{K}"] = K
    for w in (0.25, 0.5, 1.0, 2.0):
        rules[f"prior{w}"] = w
    for w in (0.5, 1.0):
        rules[f"priortop20_{w}"] = w

    res = {r: {} for r in rules}
    for i, q in enumerate(subs):
        d = pre[q]
        y, blk, P, V, ub, acc, cls = d["y"], d["blk"], d["P"], d["V"], d["ub"], d["acc"], d["cls"]
        others = np.delete(full_all, i, 0).mean(0)
        preds = {r: np.empty(len(y), int) for r in rules}
        for j, b in enumerate(ub):
            m = np.ones(len(ub), bool); m[j] = False
            tr = bacc_from(acc, cls, m)
            te = blk == b
            # 块级"按类平均"后的块准确率用于 worst / lcb
            sub = acc[:, m]
            for r, par in rules.items():
                if r == "max":
                    k = np.argmax(tr); preds[r][te] = P[k, te]
                elif r == "worst":
                    s = sub.min(1) + 1e-3 * tr; preds[r][te] = P[np.argmax(s), te]
                elif r.startswith("lcb"):
                    s = tr - par * sub.std(1); preds[r][te] = P[np.argmax(s), te]
                elif r.startswith("top"):
                    ks = np.argsort(-tr)[:par]
                    preds[r][te] = (P[ks][:, te].mean(0) >= 0.5).astype(int)
                elif r.startswith("priortop"):
                    s = tr + par * others
                    ks = np.argsort(-s)[:20]
                    preds[r][te] = (P[ks][:, te].mean(0) >= 0.5).astype(int)
                elif r.startswith("prior"):
                    s = tr + par * others; preds[r][te] = P[np.argmax(s), te]
        for r in rules:
            res[r][q] = float(frame_bacc(y, preds[r]))
        print(q, " ".join(f"{r}={res[r][q]:.3f}" for r in rules), flush=True)

    res["best"] = {q: float(pre[q]["full"].max()) for q in subs}
    rules = dict(rules, best=None)
    summ = {}
    for r in rules:
        v = np.array(list(res[r].values()))
        summ[r] = dict(mean=float(v.mean()), median=float(np.median(v)),
                       ge90=int((v >= .9).sum()), ge85=int((v >= .85).sum()))
        print(f"[{r:14s}] mean={v.mean():.3f} med={np.median(v):.3f} "
              f"ge90={summ[r]['ge90']} ge85={summ[r]['ge85']}")
    (HERE / ("nested_select_sf2.json" if USE_SF2 else "nested_select_sf.json" if USE_SF else "nested_select.json")).write_text(
        json.dumps({"summary": summ, "per_subject": res, "cands": cands},
                   ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
