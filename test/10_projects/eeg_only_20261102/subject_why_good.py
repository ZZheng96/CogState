"""E12：效果好的被试为什么好？拆成「本人可分性」×「方向可迁移性」。

可用量（均为逐被试，来自 test/audit/direction_scoring 与 test/theory）：
  auc_own_cv   同人交叉拟合（用本人标签训练本人）→ 该人的**可分性上限**
  cos_ws_wxs   本人方向与全局方向的余弦 → 该人的**方向可迁移性**
  auc_xs       跨被试成绩（真正可部署）

模型：auc_xs ≈ f(auc_own_cv, cos_ws_wxs)。再按两维中位数切四象限，看每类人的
跨被试成绩与损失来源；最后对 top 被试逐个给出「亏损在哪」的结论。

运行：py -3.11 test/10_projects/eeg_only_20261102/subject_why_good.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    ds = json.loads((ROOT / "test/audit/direction_scoring/direction_scoring.json")
                    .read_text(encoding="utf-8"))["subjects"]
    bn = json.loads((ROOT / "test/08_diagnostics/bottleneck_diagnosis.json")
                    .read_text(encoding="utf-8"))
    bank_p = HERE / "method_bank.json"
    bank = json.loads(bank_p.read_text(encoding="utf-8")) if bank_p.exists() else None

    subs = sorted(ds)
    Q = [s.replace("sub-", "") for s in subs]
    own = np.array([ds[s]["auc_own_cv"] for s in subs])
    xs = np.array([ds[s]["auc_xs"] for s in subs])
    cos = np.array([ds[s]["cos_ws_wxs"] for s in subs])
    ori = np.array([ds[s]["auc_xs_oriented"] for s in subs])
    ws = np.array([bn["within_subject"]["subjects"][q]["AUC"] for q in Q])
    cs = np.array([bn["cross_subject"]["subjects"][q]["AUC"] for q in Q])

    # 方法银行：逐被试跨方法均值 / 最优
    if bank:
        bnames = list(bank["per_subject"])
        M = np.array([[bank["per_subject"][n][q]["auc"] if bank["per_subject"][n][q]["auc"] is not None
                       else np.nan for q in Q] for n in bnames], float)
        bmean, bmax = np.nanmean(M, 0), np.nanmax(M, 0)
        bbest = [bnames[int(np.nanargmax(M[:, j]))] for j in range(len(Q))]
    else:
        bmean, bmax, bbest = xs, xs, ["n/a"] * len(Q)

    print(f"[n] {len(subs)} 人;  平均 own_cv={own.mean():.3f} cos={cos.mean():+.3f} "
          f"xs={xs.mean():.3f} bank_mean={np.nanmean(bmean):.3f}")

    def rep(name, a, b):
        r = pearsonr(a, b)
        rho = spearmanr(a, b)
        print(f"  {name:34s} pearson r={r.statistic:+.3f} (p={r.pvalue:.1e})  spearman={rho.statistic:+.3f}")

    print("[关系] 跨被试 AUC 与各量的相关：")
    rep("own_cv（本人可分性）~ xs", own, xs)
    rep("cos（方向可迁移性）~ xs", np.abs(cos), xs)
    rep("bank_mean ~ xs", bmean, xs)
    rep("bank_mean ~ own_cv", bmean, own)
    rep("bank_mean ~ |cos|", np.abs(cos), bmean)

    # 二维解释
    X = np.c_[own, np.abs(cos)]
    lr = LinearRegression().fit(X, bmean)
    pred = lr.predict(X)
    ss = 1 - np.sum((bmean - pred) ** 2) / np.sum((bmean - bmean.mean()) ** 2)
    print(f"[回归] bank_mean = {lr.intercept_:.3f} + {lr.coef_[0]:+.3f}*own_cv "
          f"{lr.coef_[1]:+.3f}*|cos|   R²={ss:.3f}")

    # 四象限
    mo, mc = np.median(own), np.median(np.abs(cos))
    print(f"[四象限] 中位切分 own_cv={mo:.3f} |cos|={mc:.3f}")
    for lo_o in (0, 1):
        for lo_c in (0, 1):
            m = ((own > mo) == bool(lo_o)) & ((np.abs(cos) > mc) == bool(lo_c))
            tag = f"{'高' if lo_o else '低'}可分性×{'高' if lo_c else '低'}迁移"
            print(f"  {tag:12s} n={m.sum():2d} bank_mean={np.nanmean(bmean[m]):.3f} "
                  f"xs={xs[m].mean():.3f} own={own[m].mean():.3f} |cos|={np.abs(cos[m]).mean():.3f}")

    # top / bottom 列表
    order = np.argsort(-bmean)
    print("\n[最稳的前 12 人]")
    print("  sub   bank_mean  bank_max  best_method              xs    own_cv  |cos|  迁移效率  within")
    for i in order[:12]:
        print(f"  {Q[i]}   {bmean[i]:.3f}     {bmax[i]:.3f}   {bbest[i]:22s} {xs[i]:.3f}  {own[i]:.3f}  "
              f"{abs(cos[i]):.3f}   {xs[i]/own[i]:.2f}     {ws[i]:.3f}")
    print("\n[最差的 12 人]")
    for i in order[-12:]:
        print(f"  {Q[i]}   {bmean[i]:.3f}     {bmax[i]:.3f}   {bbest[i]:22s} {xs[i]:.3f}  {own[i]:.3f}  "
              f"{abs(cos[i]):.3f}   {xs[i]/own[i]:.2f}     {ws[i]:.3f}")

    # 上界核算
    print(f"\n[上界] 每人达到本人 own_cv → mac bACC 期望 ≈ {own.mean():.3f}；"
          f"每人达到 bank 内最优方法（无标签不可选）→ AUC 均值 {np.nanmean(bmax):.3f}")
    top = order[:12]
    print(f"[top12] 若按无标签共识集成，AUC 均值 {np.nanmean(bmean[top]):.3f}，"
          f"与本人上限的差距 {np.nanmean(own[top]) - np.nanmean(bmean[top]):+.3f}")
    print(f"[top12] 方向迁移效率(xs/own) 均值 {np.nanmean(xs[top]/own[top]):.2f}；"
          f"全体 {np.nanmean(xs/own):.2f}")

    out = {"rows": [{"subject": Q[i], "bank_mean": float(bmean[i]), "bank_max": float(bmax[i]),
                     "best_method": bbest[i], "xs": float(xs[i]), "own_cv": float(own[i]),
                     "cos": float(cos[i]), "within": float(ws[i]),
                     "transfer_eff": float(xs[i] / own[i])} for i in range(len(Q))],
           "regression": {"intercept": float(lr.intercept_), "coef": lr.coef_.tolist(), "r2": float(ss)},
           "corr": {"own_vs_bmean": float(pearsonr(own, bmean).statistic),
                    "abscos_vs_bmean": float(pearsonr(np.abs(cos), bmean).statistic),
                    "own_vs_xs": float(pearsonr(own, xs).statistic),
                    "abscos_vs_xs": float(pearsonr(np.abs(cos), xs).statistic)},
           "median_split": {"own_cv": float(mo), "abscos": float(mc)}}
    (HERE / "subject_why_good.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                encoding="utf-8")
    print("\n[done] subject_why_good.json")


if __name__ == "__main__":
    main()
