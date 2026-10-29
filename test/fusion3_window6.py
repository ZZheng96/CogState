"""4.29 严格 6 秒单窗三路等权 logit 融合：手工 MLP（seed_noise_band 种子平均）+ ResNet1d（raw_waveform_window6）+ 宽频新特征 LR（broadband_shape_window6）。
权重固定 1/3，不在测试折上选择；每路分数只依赖当前窗，可逐窗部署。"""
import json, time
import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

t0 = time.time()
d = "D:/BCIC2026/CogState/test/"
a = np.load(d + "raw_waveform_window6_oof.npz", allow_pickle=True)
b = np.load(d + "broadband_shape_window6_oof.npz", allow_pickle=True)
s = np.load(d + "seed_noise_band_oof.npz", allow_pickle=True)
y = s["y"]; g = s["groups"]
assert (a["y"] == y).all() and (b["y"] == y).all() and (a["groups"] == g).all() and (b["groups"] == g).all()
assert (b["fold_id"] == s["fold_id"]).all()


def to_logit(p):
    if p.min() >= 0 and p.max() <= 1:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))
    return p


src = {"hand": s["oof_seed_mean"], "resnet": a["resnet1d_seed_mean"], "new_lr": b["logreg__all_new"]}
L = {k: to_logit(v) for k, v in src.items()}
combos = {
    "hand": ["hand"], "resnet": ["resnet"], "new_lr": ["new_lr"],
    "hand+resnet": ["hand", "resnet"], "hand+new_lr": ["hand", "new_lr"], "resnet+new_lr": ["resnet", "new_lr"],
    "hand+resnet+new_lr": ["hand", "resnet", "new_lr"],
}
rng = np.random.default_rng(20261029)
subs = np.unique(g); idx = {u: np.where(g == u)[0] for u in subs}
boots = [np.concatenate([idx[u] for u in rng.choice(subs, len(subs))]) for _ in range(1000)]
base = L["hand"]; base_auc = roc_auc_score(y, base)
res = {}; out = {"y": y, "groups": g}
for i, (name, ks) in enumerate(combos.items()):
    f = np.mean([L[k] for k in ks], axis=0)
    auc = roc_auc_score(y, f)
    bacc0 = balanced_accuracy_score(y, f > 0)
    baccm = balanced_accuracy_score(y, f > np.median(f))
    per = [roc_auc_score(y[g == u], f[g == u]) for u in subs]
    dl = np.array([roc_auc_score(y[j], f[j]) - roc_auc_score(y[j], base[j]) for j in boots])
    res[name] = {"auc": auc, "bacc_logit0": bacc0, "bacc_pooled_median": baccm,
                 "subject_auc_median": float(np.median(per)), "subject_auc_lt_0.5_frac": float(np.mean(np.array(per) < 0.5)),
                 "delta_auc_vs_hand": auc - base_auc, "delta_ci95": [float(np.percentile(dl, 2.5)), float(np.percentile(dl, 97.5))],
                 "p_delta_gt0": float(np.mean(dl > 0))}
    out[name] = f
    print(f"[{i+1}/{len(combos)}] {name:20s} AUC {auc:.4f} bACC@0 {bacc0:.4f} bACC@med {baccm:.4f} "
          f"dAUC {auc-base_auc:+.4f} CI [{res[name]['delta_ci95'][0]:+.4f},{res[name]['delta_ci95'][1]:+.4f}] "
          f"subjAUC<0.5 {res[name]['subject_auc_lt_0.5_frac']:.3f}", flush=True)

json.dump({"protocol": {"test_subject_isolation": True, "test_history": False, "window_position_schedule_r_t": False,
                        "test_subject_calibration": False, "fusion_weights": "fixed equal, not tuned on test folds",
                        "bacc_pooled_median_note": "pooled OOF median threshold, evaluation only"},
           "bootstrap": {"n": 1000, "seed": 20261029, "unit": "subject"},
           "results": res, "elapsed_s": time.time() - t0},
          open(d + "fusion3_window6_results.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
np.savez(d + "fusion3_window6_oof.npz", **out)
print("done", round(time.time() - t0, 1), "s", flush=True)
