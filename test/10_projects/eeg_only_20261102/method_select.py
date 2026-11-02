"""E11：能否用**无标签**的逐被试判据，为每个人挑方法/做集成？

E10 的矩阵显示：同一被试在不同方法上的 AUC 差异很大（被试×方法占 34% 方差），
逐被试取最优方法的 oracle 均值 0.743，比跨方法均值高 +0.134。
问题：这个增益能否用无标签信息捕获？

本脚本只用 method_bank 的 OOF 分数（不含任何标签信息做选择），对每个被试试这些选择规则：
  consensus      各方法 z 分数均值（全局集成）
  median         各方法 z 分数中位
  bimodality     选分数分布双峰系数最大的方法
  otsu_sep       选 Otsu 切分后类间/总方差最大的方法
  agreement      选与其它方法共识相关最高（最"主流"）的方法
  top3_agree     取与共识相关最高的 3 个方法平均
  diverge        选与共识相关最低的方法（对照，预期差）
  oracle_rule    每人取表现最好的规则（上界，仅诊断）

并报告：池化 AUC/bACC/F1、subject-macro bACC、逐被试与 lr_c001 的差、
以及在 top-10 / bottom-10 被试上的分别变化。

运行：py -3.11 test/10_projects/eeg_only_20261102/method_select.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent


def bimodality(x):
    from scipy.stats import kurtosis
    n = len(x)
    if n < 10 or x.std() < 1e-12:
        return -9.0
    b = (kurtosis(x, fisher=True) + 3) ** 2 / 3.0  # SAS 双峰系数
    return float(b)


def otsu_sep(x):
    """一维 Otsu：返回类间方差 / 总方差。"""
    if x.std() < 1e-12 or len(x) < 10:
        return -9.0
    qs = np.quantile(x, np.linspace(0.05, 0.95, 37))
    best = -9.0
    for q in qs:
        a, b = x[x <= q], x[x > q]
        if len(a) < 3 or len(b) < 3:
            continue
        w = len(a) / len(x)
        v = w * (1 - w) * (a.mean() - b.mean()) ** 2
        if v > best:
            best = v
    return float(best / (x.var() + 1e-12))


def pooled(y, s, g, subs):
    yh = (s >= 0).astype(int)
    per = {q: balanced_accuracy_score(y[g == q], yh[g == q]) for q in subs}
    return {"AUC": float(roc_auc_score(y, s)), "bACC": float(balanced_accuracy_score(y, yh)),
            "F1_macro": float(f1_score(y, yh, average="macro")),
            "subject_macro_bACC": float(np.mean(list(per.values())))}


def per_sub_auc(y, s, g, subs):
    out = {}
    for q in subs:
        m = np.isfinite(s[g == q])
        yy, ss = y[g == q][m], s[g == q][m]
        out[q] = float(roc_auc_score(yy, ss)) if len(np.unique(yy)) > 1 else float("nan")
    return out


def main():
    Z = np.load(HERE / "method_bank_scores.npz")
    y, g = Z["y"].astype(int), Z["g"]
    subs = sorted(np.unique(g))
    names = [k[3:] for k in Z.files if k.startswith("s__")]
    S = {n: Z[f"s__{n}"] for n in names}
    S = {n: v for n, v in S.items() if np.isfinite(v).all()}  # 只用全覆盖方法
    names = list(S)
    print(f"[load] {len(names)} 个方法 × {len(y)} 窗 {names}", flush=True)

    def choose(score_fn):
        """按被试挑一个方法名，返回拼接后的 z 分数。"""
        s = np.zeros(len(y))
        pick = {}
        for q in subs:
            m = g == q
            vals = [(n, score_fn(S[n][m])) for n in names]
            vals = [(n, v) for n, v in vals if np.isfinite(v)]
            bn = max(vals, key=lambda z: z[1])[0] if vals else names[0]
            pick[q] = bn
            s[m] = S[bn][m]
        return s, pick

    rules = {}
    rules["consensus"] = np.nanmean(np.vstack([S[n] for n in names]), axis=0)
    rules["median"] = np.nanmedian(np.vstack([S[n] for n in names]), axis=0)
    rules["bimodality"], _ = choose(lambda v: bimodality(v))
    rules["otsu_sep"], _ = choose(lambda v: otsu_sep(v))

    # 逐被试：与共识的相关（最高=最主流 / 最低=最离群）
    s_ag, s_dv, s3 = (np.zeros(len(y)) for _ in range(3))
    pick_ag = {}
    for q in subs:
        m = g == q
        c = rules["consensus"][m]
        vals = [(n, float(np.corrcoef(S[n][m], c)[0, 1])) for n in names
                if np.std(S[n][m]) > 1e-12 and np.std(c) > 1e-12]
        top = sorted(vals, key=lambda z: -z[1])
        pick_ag[q] = top[0][0]
        s_ag[m] = S[top[0][0]][m]
        s_dv[m] = S[top[-1][0]][m]
        s3[m] = np.mean([S[n][m] for n, _ in top[:3]], axis=0)
    rules["agreement"] = s_ag
    rules["top3_agree"] = s3
    rules["diverge"] = s_dv

    res = {k: pooled(y, v, g, subs) for k, v in rules.items()}
    per = {k: per_sub_auc(y, v, g, subs) for k, v in rules.items()}
    base = per["consensus"]
    rank = np.array([base[q] for q in subs])
    order = np.argsort(-rank)
    top10 = [subs[i] for i in order[:10]]
    bot10 = [subs[i] for i in order[-10:]]
    print(f"{'rule':14s} {'AUC':>7s} {'bACC':>7s} {'F1m':>7s} {'smACC':>7s}  top10Δ  bot10Δ", flush=True)
    for k, m in res.items():
        dt = np.mean([per[k][q] - base[q] for q in top10])
        db = np.mean([per[k][q] - base[q] for q in bot10])
        print(f"{k:14s} {m['AUC']:7.4f} {m['bACC']:7.4f} {m['F1_macro']:7.4f} "
              f"{m['subject_macro_bACC']:7.4f} {dt:+.3f} {db:+.3f}", flush=True)

    # 每人规则 oracle（上界）
    s_or = np.zeros(len(y))
    best_rule = {}
    for q in subs:
        m = g == q
        br = max(per, key=lambda k: per[k][q] if np.isfinite(per[k][q]) else -1)
        best_rule[q] = br
        s_or[m] = rules[br][m]
    orc = pooled(y, s_or, g, subs)
    print(f"{'oracle_rule':14s} {orc['AUC']:7.4f} {orc['bACC']:7.4f} {orc['F1_macro']:7.4f} "
          f"{orc['subject_macro_bACC']:7.4f}   （每人挑最优规则，诊断上界）", flush=True)

    out = {"rules": res, "per_subject": per, "oracle_rule": orc,
           "oracle_rule_choice": best_rule,
           "pick_agreement": pick_ag,
           "top10": top10, "bottom10": bot10,
           "note": "所有规则只用无标签 OOF 分数分布；选择发生在被试内部，不使用该被试标签"}
    (HERE / "method_select.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    print("[done] method_select.json")


if __name__ == "__main__":
    main()
