"""E43：E42 高/低表现组诊断；分组不进入正式推理。

高组=E42 基线全段 bACC 高于57人中位数，低组=不高于中位数。
画像路由仅使用 subj_profile.py 中前4段可无标签计算的原始/频带字段；
目标被试的组别标签在留一训练时排除。画像诊断不等同于已部署路由。
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "42_e17_loso_hmm"))
sys.path.insert(0, str(HERE.parent / "17_compliant"))
import e42  # noqa: E402
import loso as L  # noqa: E402

SAFE = [
    "rel_delta", "rel_theta", "rel_alpha", "rel_beta", "rel_gamma",
    "hf_ratio_feat", "rms_feat", "flat_frac", "spike_frac", "nan_frac",
    "line50", "emg_hf", "alpha_raw", "theta_raw", "beta_raw",
    "amp_std_uv", "bad_zero_frac", "bad_ptp_frac",
]


def main():
    result42 = json.load(open(HERE.parent / "42_e17_loso_hmm" / "e42.json"))
    cache = np.load(HERE.parent / "42_e17_loso_hmm" / "e42_scores.npz")
    subs, D = L.load()
    fixed = e42.E41.HMM_PARAMS.index((300, 1.0))
    rows = []
    for q in subs:
        score = cache[q]
        hmm = e42.E41.hmm_forward_bank(torch.tensor(score, device="cuda"))[fixed].cpu().numpy()
        late = D[q][2] >= 4
        base_all = e42.metrics(D[q][1], score)
        hmm_all = e42.metrics(D[q][1], hmm)
        base_late = e42.metrics(D[q][1][late], score[late])
        hmm_late = e42.metrics(D[q][1][late], hmm[late])
        warm_base = e42.metrics(D[q][1][:600], score[:600])
        warm_hmm = e42.metrics(D[q][1][:600], hmm[:600])
        rows.append(dict(subject=q, base=base_all, hmm=hmm_all,
                         late_base=base_late, late_hmm=hmm_late,
                         warm_base=warm_base, warm_hmm=warm_hmm))
    threshold = float(np.median([r["base"]["bacc"] for r in rows]))
    for r in rows:
        r["group"] = "high" if r["base"]["bacc"] > threshold else "low"
        r["hmm_gain"] = r["hmm"]["bacc"] - r["base"]["bacc"]

    groups = {}
    for name, selected in (
        ("high", [r for r in rows if r["group"] == "high"]),
        ("low", [r for r in rows if r["group"] == "low"]),
        ("reversed_auc", [r for r in rows if r["base"]["auc"] < 0.5]),
        ("not_reversed", [r for r in rows if r["base"]["auc"] >= 0.5]),
    ):
        groups[name] = {"n": len(selected), "subjects": [r["subject"] for r in selected]}
        for key in ("base", "hmm", "late_base", "late_hmm", "warm_base", "warm_hmm"):
            groups[name][key] = e42.summary([r[key] for r in selected])
        groups[name]["hmm_gain_mean"] = float(np.mean([r["hmm_gain"] for r in selected]))
        groups[name]["hmm_gain_median"] = float(np.median([r["hmm_gain"] for r in selected]))

    profile_rows = {r["subject"]: r for r in csv.DictReader(open(
        HERE.parent / "23_synthesis" / "profile.csv", encoding="utf-8"))}
    X = np.array([[float(profile_rows[q][k] or "nan") for k in SAFE] for q in subs])
    y = np.array([r["group"] == "high" for r in rows], dtype=int)
    profile_summary = {}
    for j, key in enumerate(SAFE):
        low, high = X[y == 0, j], X[y == 1, j]
        pooled_sd = np.sqrt((np.nanvar(low) + np.nanvar(high)) / 2) + 1e-12
        profile_summary[key] = {
            "low_median": float(np.nanmedian(low)),
            "high_median": float(np.nanmedian(high)),
            "standardized_mean_diff": float((np.nanmean(high) - np.nanmean(low)) / pooled_sd),
        }
    oof = np.zeros(len(subs))
    for i in range(len(subs)):
        tr = np.arange(len(subs)) != i
        model = make_pipeline(SimpleImputer(), StandardScaler(),
                              LogisticRegression(C=1, class_weight="balanced", max_iter=2000))
        model.fit(X[tr], y[tr])
        oof[i] = model.predict_proba(X[i:i + 1])[0, 1]
    router = {
        "features": SAFE,
        "protocol": "57人留一；组别由E42基线bACC中位数定义；只用前4段无标签画像",
        "auc": float(roc_auc_score(y, oof)),
        "bacc_at_0.5": float(balanced_accuracy_score(y, oof >= 0.5)),
        "positive_rate": float(np.mean(oof >= 0.5)),
        "oof": {q: float(s) for q, s in zip(subs, oof)},
    }
    result = {"group_threshold_bacc": threshold, "groups": groups,
              "profile_summary": profile_summary, "profile_router": router}
    (HERE / "e43.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(HERE / "group_members.csv", "w", newline="", encoding="utf-8-sig") as f:
        fields = ["subject", "group"] + [f"{a}_{m}" for a in ("base", "hmm", "late_base", "late_hmm", "warm_base", "warm_hmm")
                                         for m in ("bacc", "auc", "f1")] + ["hmm_gain"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            row = {"subject": r["subject"], "group": r["group"], "hmm_gain": r["hmm_gain"]}
            for a in ("base", "hmm", "late_base", "late_hmm", "warm_base", "warm_hmm"):
                row.update({f"{a}_{m}": r[a][m] for m in ("bacc", "auc", "f1")})
            writer.writerow(row)
    print(json.dumps({"threshold": threshold, "groups": groups,
                      "router": {k: v for k, v in router.items() if k != "oof"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
