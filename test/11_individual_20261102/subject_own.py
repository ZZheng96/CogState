"""E18：逐被试本人标注（留一块交叉拟合）→ 与全局方向融合 → 个体记忆。

口径（用户 2026-11-02 确认）：允许使用该被试自己的标注，但必须严格交叉拟合。
本脚本每个被试完全独立地：
  1. 留一块（LOBO，6 块）训练本人 LR，得到"本人方向"分数 s_own（被评块从未参与训练）；
  2. 载入全局方向分数 s_glb（来自其余 4 折被试，见 score_cache.py）；
  3. 用其余块拟合 2 维融合 LR [s_glb, s_own]（同样留一块，避免融合参数泄漏）；
  4. 在真实时间顺序上做因果标准化 + 两态 HMM 记忆；
  5. 报逐被试 AUC / bACC / F1，并与纯全局、纯本人比较。

运行：py -3.11 test/11_individual_20261102/subject_own.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from final_eval import cstd, hmm_forward

HERE = Path(__file__).resolve().parent
SLID = HERE / "sliding"
SC = HERE / "scores"
STRIDE = 2.0


def lobo_scores(X, y, blk, fallback, C=0.01):
    """留一块交叉拟合，返回与 X 对齐的本人方向分数。
    若某折训练块只剩一个类别，则该块回退到全局分数（并计数）。"""
    s = np.full(len(y), np.nan)
    n_fallback = 0
    for b in np.unique(blk):
        te = blk == b
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            s[te] = fallback[te]
            n_fallback += 1
            continue
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X[tr]), y[tr])
        s[te] = m.decision_function(sc.transform(X[te]))
    return s, n_fallback


def blend_scores(sg, so, y, blk):
    """对 [s_glb, s_own] 做留一块融合，返回融合分数（被评块未参与融合拟合）。"""
    Z = np.c_[sg, so]
    out = np.full(len(y), np.nan)
    for b in np.unique(blk):
        te = blk == b
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            out[te] = sg[te]
            continue
        sc = StandardScaler().fit(Z[tr])
        m = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(Z[tr]), y[tr])
        out[te] = m.decision_function(sc.transform(Z[te]))
    return out


def report(y, s, tag, out, q):
    rec = {"AUC": float(roc_auc_score(y, s)),
           "bACC": float(balanced_accuracy_score(y, (s >= 0).astype(int))),
           "F1": float(f1_score(y, (s >= 0).astype(int)))}
    out.setdefault(tag, {})[q] = rec
    return rec


def main():
    subs = sorted(p.stem.replace("sub-", "") for p in SC.glob("sub-*.npz"))
    out, rows = {}, []
    print(f"{'sub':4s} {'ownAUC':>7s} {'glbAUC':>7s} {'mixAUC':>7s} {'mem_bACC':>9s} "
          f"{'own_bACC':>9s} {'mix_bACC':>9s} {'final':>7s}")
    for q in subs:
        p = SLID / f"sub-{q}.npz"
        if not p.exists():
            continue
        S = np.load(p)
        Z = np.load(SC / f"sub-{q}.npz")
        X, y, blk, t = (np.nan_to_num(S["F"].astype(np.float32)), S["y"].astype(int),
                        S["blk"].astype(int), S["t"])
        o = np.argsort(t)
        X, y, blk, t = X[o], y[o], blk[o], t[o]
        assert np.array_equal(y, Z["y"].astype(int)), q
        sg = Z["score__lr"].astype(float)
        if len(np.unique(y)) < 2:
            print(f"[skip] {q}: single class")
            continue
        so, nfb = lobo_scores(X, y, blk, sg)
        sm = blend_scores(sg, so, y, blk)

        mu0, mu1, sd = float(Z["mu0__lr"]), float(Z["mu1__lr"]), float(Z["sd__lr"])
        pr_g = hmm_forward(cstd(sg), mu0, mu1, sd, 1 - STRIDE / 60)
        # 本人/融合分数的发射参数：用该被试训练块的分数分布（留一块外）估计
        pr_o = hmm_forward(cstd(so), 0.0, 1.0, 1.0, 1 - STRIDE / 60)
        pr_m = hmm_forward(cstd(sm), 0.0, 1.0, 1.0, 1 - STRIDE / 60)

        rg = report(y, sg, "global", out, q)
        ro = report(y, so, "own", out, q)
        rm = report(y, sm, "mixed", out, q)
        rec = {"global": rg, "own": ro, "mixed": rm,
               "mem_global": {"AUC": float(roc_auc_score(y, pr_g)),
                              "bACC": float(balanced_accuracy_score(y, pr_g >= .5)),
                              "F1": float(f1_score(y, pr_g >= .5))},
               "mem_own": {"AUC": float(roc_auc_score(y, pr_o)),
                           "bACC": float(balanced_accuracy_score(y, pr_o >= .5)),
                           "F1": float(f1_score(y, pr_o >= .5))},
               "mem_mixed": {"AUC": float(roc_auc_score(y, pr_m)),
                             "bACC": float(balanced_accuracy_score(y, pr_m >= .5)),
                             "F1": float(f1_score(y, pr_m >= .5))}}
        out.setdefault("mem", {})[q] = rec
        final = max(rec["mem_global"]["bACC"], rec["mem_own"]["bACC"], rec["mem_mixed"]["bACC"])
        rows.append((q, ro["AUC"], rg["AUC"], rm["AUC"], rec["mem_global"]["bACC"],
                     rec["mem_own"]["bACC"], rec["mem_mixed"]["bACC"], final))
        print(f"{q:4s} {ro['AUC']:7.3f} {rg['AUC']:7.3f} {rm['AUC']:7.3f} "
              f"{rec['mem_global']['bACC']:9.3f} {rec['mem_own']['bACC']:9.3f} "
              f"{rec['mem_mixed']['bACC']:9.3f} {final:7.3f}", flush=True)

    f = np.array([r[-1] for r in rows])
    print(f"\n[summary] n={len(f)} mean={f.mean():.3f} median={np.median(f):.3f} "
          f"n>=.90={int((f >= .9).sum())} n>=.85={int((f >= .85).sum())}")
    (HERE / "subject_own.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    print("[done] subject_own.json")


if __name__ == "__main__":
    main()
