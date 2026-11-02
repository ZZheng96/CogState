"""E7：逐被试审计 → 按信号属性分组 → 分组能否提升指标？

背景（E6）：跨被试 LR 的逐被试卷级 AUC 中位 0.669、最低 0.333、9/57 低于 0.5；
而给每人自己的方向（oracle）后中位 0.806。个体差异是主要来源。

本脚本做四件事（全部不使用次序/位置，分组特征也不使用标签）：
  1. 逐被试卷级 AUC / bACC（5 折 OOF），并汇总成三档；
  2. 计算**无标签**的个体属性：信号质量代理、α 相对功率、高频（肌电）占比、谱熵、
     窗数、类内离散度等；看它们与逐被试 AUC 的关系；
  3. 用这些属性 KMeans 分组（k=3/4），报各组的 AUC 与属性差异；
  4. 用属性做"可靠性预测"（留一被试）：① 对预测不可靠的被试输出 0.5（弃权）；
     ② 训练时按预测可靠性加权。比较这两种用法能否提升池化 bACC/AUC。

运行：py -3.11 test/10_projects/eeg_only_20261102/per_subject_audit.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
T0 = time.perf_counter()


def el():
    return f"elapsed={time.perf_counter() - T0:.1f}s"


def pooled(y, s, g):
    yh = (s >= 0).astype(int)
    per = {q: balanced_accuracy_score(y[g == q], yh[g == q]) for q in np.unique(g)}
    return {"AUC": float(roc_auc_score(y, s)), "bACC": float(balanced_accuracy_score(y, yh)),
            "F1_macro": float(f1_score(y, yh, average="macro")),
            "subject_macro_bACC": float(np.mean(list(per.values()))),
            "n_ge_0.9": int(np.sum(np.array(list(per.values())) >= 0.9))}


def main():
    D = np.load(ROOT / "test" / "10_projects" / "protocol_20261102" / "realorder.npz")
    F = np.nan_to_num(D["F"].astype(float))
    y, g, t = D["y"].astype(int), D["g"], D["t"]
    subs = sorted(np.unique(g))
    man = json.loads((ROOT / "database" / "manifest.json").read_text(encoding="utf-8"))
    names = man["feature_names"] + man["emg_feature_names"]
    print(f"[data] windows={len(y)} subs={len(subs)} dim={F.shape[1]} {el()}", flush=True)

    def idx(pat):
        return [i for i, n in enumerate(names) if pat in n]

    Q = idx("quality")                       # 质量代理
    A8 = [i for i in idx("stft_relative_8_13_mean")]
    HF = [i for i in idx("emg")] or list(range(len(names) - 20, len(names)))
    # 属性：每被试一组无标签统计
    attrs, anames = [], ["n_windows", "log_abs_power", "alpha_rel", "hf_rel", "spec_entropy",
                         "kurtosis_mean", "crest_mean", "bad_flat", "bad_spike", "dispersion"]
    for s in subs:
        q = g == s
        Zs = F[q]
        sd = Zs.std(0)
        Zt = (Zs - Zs.mean(0)) / np.where(sd > 1e-12, sd, 1.0)
        row = [np.log(q.sum()),
               float(np.mean(Zs[:, [i for i in idx("stft_log_power_1_4_mean")]] if idx("stft_log_power_1_4_mean") else np.zeros((q.sum(), 1)))),
               float(np.mean(Zs[:, A8])) if A8 else 0.0,
               float(np.mean(np.abs(Zs[:, HF]))) if HF else 0.0,
               float(np.mean(Zs[:, idx("stft_log_power_13_30_mean")])) if idx("stft_log_power_13_30_mean") else 0.0,
               float(np.mean(Zs[:, idx("stft_log_power_4_8_kurtosis")])) if idx("stft_log_power_4_8_kurtosis") else 0.0,
               float(np.mean(Zs[:, idx("stft_log_power_1_4_max")])) if idx("stft_log_power_1_4_max") else 0.0,
               float(np.mean(Zs[:, [i for i in Q if "flat" in names[i]]])) if [i for i in Q if "flat" in names[i]] else 0.0,
               float(np.mean(Zs[:, [i for i in Q if "spike" in names[i]]])) if [i for i in Q if "spike" in names[i]] else 0.0,
               float(np.linalg.norm(sd))]
        attrs.append(row)
    AT = np.asarray(attrs, float)
    AT = (AT - AT.mean(0)) / np.where(AT.std(0) > 1e-12, AT.std(0), 1.0)
    print(f"[attrs] {len(anames)} 个无标签个体属性 {el()}", flush=True)

    # 1) 5 折 OOF 的跨被试 LR
    folds = np.empty(len(y), int)
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y, g)):
        folds[te] = k
    s = np.full(len(y), np.nan)
    for k in range(5):
        tr, te = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        sc = StandardScaler().fit(F[tr])
        m = LogisticRegression(C=.01, max_iter=2000).fit(sc.transform(F[tr]), y[tr])
        s[te] = m.decision_function(sc.transform(F[te]))
    base = pooled(y, s, g)
    per_auc = {q: roc_auc_score(y[g == q], s[g == q]) for q in subs}
    per_bac = {q: balanced_accuracy_score(y[g == q], (s[g == q] >= 0).astype(int)) for q in subs}
    print(f"[base] AUC={base['AUC']:.4f} bACC={base['bACC']:.4f} F1m={base['F1_macro']:.4f}", flush=True)

    order = sorted(subs, key=lambda q: per_auc[q])
    print("[tiers] 按逐被试 AUC 分三档：", flush=True)
    tiers = {}
    for name, chunk in (("low", order[:19]), ("mid", order[19:38]), ("high", order[38:])):
        tiers[name] = {"subjects": chunk,
                       "auc_mean": float(np.mean([per_auc[q] for q in chunk])),
                       "bacc_mean": float(np.mean([per_bac[q] for q in chunk]))}
        print(f"  {name:5s} n={len(chunk)} AUC={tiers[name]['auc_mean']:.3f} bACC={tiers[name]['bacc_mean']:.3f} "
              f"subs={chunk}", flush=True)

    # 2) 属性 vs 逐被试 AUC
    ya = np.array([per_auc[q] for q in subs])
    rc = [np.corrcoef(AT[:, j], ya)[0, 1] for j in range(AT.shape[1])]
    print("[attr-corr] 与逐被试 AUC 的相关：", flush=True)
    for nm, c in sorted(zip(anames, rc), key=lambda z: -abs(z[1])):
        print(f"  {nm:16s} r={c:+.3f}", flush=True)

    # 3) KMeans 分组（属性，无标签）
    groups = {}
    for kk in (3, 4):
        lab = KMeans(kk, n_init=10, random_state=0).fit_predict(AT)
        groups[kk] = {}
        for c in range(kk):
            mem = [subs[i] for i in range(len(subs)) if lab[i] == c]
            groups[kk][c] = {"n": len(mem), "auc_mean": float(np.mean([per_auc[q] for q in mem])),
                             "attr_mean": dict(zip(anames, np.round(AT[lab == c].mean(0), 2).tolist()))}
        print(f"[km={kk}] " + " | ".join(f"c{c}:n={v['n']},AUC={v['auc_mean']:.3f}"
                                        for c, v in groups[kk].items()), flush=True)

    # 4) 可靠性预测（留一被试）与两种用法
    Xa = AT
    pred_auc = np.zeros(len(subs))
    pred_ok = np.zeros(len(subs))
    for i in range(len(subs)):
        m = np.arange(len(subs)) != i
        pred_auc[i] = Ridge(alpha=1.0).fit(Xa[m], ya[m]).predict(Xa[i:i + 1])[0]
        ok = (ya[m] > 0.5).astype(int)
        if ok.min() != ok.max():
            pred_ok[i] = LogisticRegression(C=.3, max_iter=1000).fit(Xa[m], ok).predict_proba(Xa[i:i + 1])[0, 1]
    rho = float(np.corrcoef(pred_auc, ya)[0, 1])
    print(f"[reliab] 留一被试预测逐被试 AUC 的 r={rho:+.3f}；预测>0.5 的比例={np.mean(pred_ok > .5):.2f}", flush=True)

    # 4a 弃权：预测可靠度最低的 25% 输出 0.5（中性）
    k_ab = max(1, int(round(0.25 * len(subs))))
    worst = set(np.argsort(pred_auc)[:k_ab])
    s_ab = s.copy()
    for i, q in enumerate(subs):
        if i in worst:
            s_ab[g == q] = 0.0
    ab = pooled(y, s_ab, g)
    # 4b 训练加权（用训练折内的可靠性预测，避免用测试被试的标签）
    s_w = np.full(len(y), np.nan)
    for k in range(5):
        tr, te = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        subs_tr = [q for q in subs if folds[g == q][0] != k]
        ya_tr = np.array([per_auc[q] for q in subs_tr])
        pr = Ridge(alpha=1.0).fit(AT[[subs.index(q) for q in subs_tr]], ya_tr).predict(
            AT[[subs.index(q) for q in subs_tr]])
        wmap = {q: float(np.clip(p, -1, 1) + 1.2) for q, p in zip(subs_tr, pr)}
        w = np.array([wmap[q] for q in g[tr]])
        sc = StandardScaler().fit(F[tr])
        m = LogisticRegression(C=.01, max_iter=2000).fit(sc.transform(F[tr]), y[tr], sample_weight=w)
        s_w[te] = m.decision_function(sc.transform(F[te]))
    wt = pooled(y, s_w, g)

    out = {"n_windows": int(len(y)), "n_subjects": len(subs), "base": base,
           "attr_names": anames, "attr_corr": dict(zip(anames, [float(c) for c in rc])),
           "reliability_loo_r": rho, "tiers": tiers, "kmeans": {str(k): v for k, v in groups.items()},
           "abstain_25pct": ab, "weighted_train": wt,
           "per_subject": {q: {"auc": float(per_auc[q]), "bacc": float(per_bac[q])} for q in subs},
           "seconds": 0.0}
    print(f"[abstain 25%] AUC={ab['AUC']:.4f} bACC={ab['bACC']:.4f} F1m={ab['F1_macro']:.4f}", flush=True)
    print(f"[weighted]    AUC={wt['AUC']:.4f} bACC={wt['bACC']:.4f} F1m={wt['F1_macro']:.4f}", flush=True)
    out["seconds"] = time.perf_counter() - T0
    (HERE / "per_subject_audit.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[done] {out['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
