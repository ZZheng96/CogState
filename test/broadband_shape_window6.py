"""宽频（45-125 Hz）/眼动/谱形状/尺度不变特征族（当前 6 秒窗，严格受试者隔离）。

所有特征只由当前窗 raw 波形计算；不使用窗口位置、时间表、测试历史或测试受试者校准。
外层 StratifiedGroupKFold(5, shuffle, rs=0)；标准化、常量列剔除、C 选择均只在训练折内完成。
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis, skew
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "test"
FS = 250.0
C_GRID = (0.001, 0.01, 0.1, 1.0)
HGB_SEEDS = (0, 1, 2)
N_BOOT = 1000
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ---------------- 频谱审计 ----------------
def spectrum_audit(f, P, y):
    valid = P.sum(1) > 0
    n_flat = int((~valid).sum())
    P, y = P[valid], y[valid]
    mean_all = P.mean(0)
    med_all = np.median(P, 0)

    def bp(lo, hi, spec):
        s = (f >= lo) & (f < hi)
        return float(spec[s].mean())

    def peak_db(f0, spec):
        c = (f >= f0 - 0.5) & (f <= f0 + 0.5)
        fl = ((f >= f0 - 5) & (f < f0 - 2)) | ((f > f0 + 2) & (f <= f0 + 5))
        return float(10 * np.log10(spec[c].max() / np.median(spec[fl])))

    ref = bp(10, 40, med_all)
    band_db = {f"{lo}-{hi}Hz": float(10 * np.log10(bp(lo, hi, med_all) / ref))
               for lo, hi in ((1, 4), (8, 13), (30, 45), (45, 48), (52, 70), (70, 100), (100, 120), (120, 125))}
    # 每窗 50 Hz 峰（窄带/邻带）分布
    c = (f >= 49.5) & (f <= 50.5)
    fl = ((f >= 45) & (f < 48)) | ((f > 52) & (f <= 55))
    per_win = 10 * np.log10(P[:, c].max(1) / np.median(P[:, fl], 1))
    # 45 Hz 以上是否有信号：100-125 与 30-45 比，若低通应接近数值地板（<-60 dB）
    hf_rel = 10 * np.log10(P[:, (f >= 100) & (f <= 124)].mean(1) / P[:, (f >= 30) & (f < 45)].mean(1))
    lo45 = (f >= 40) & (f < 45)
    hi50 = (f >= 52) & (f < 60)
    edge_drop = float(10 * np.log10(med_all[hi50].mean() / med_all[lo45].mean()))
    # 每 5 Hz 中位 PSD（dB re 10-40 Hz）
    curve = {f"{a}": float(10 * np.log10(bp(a, a + 5, med_all) / ref)) for a in range(30, 125, 5)}
    cls = {}
    for k in (0, 1):
        m = np.median(P[y == k], 0)
        cls[("relax", "task")[k]] = {f"{lo}-{hi}Hz_dB_re_10_40": float(10 * np.log10(bp(lo, hi, m) / bp(10, 40, m)))
                                     for lo, hi in ((45, 70), (70, 100), (100, 125))}
    res = {
        "welch": "nperseg=500 (2 s), noverlap=250, hann, 0.5 Hz 分辨率, 每窗独立",
        "n_all_zero_windows_excluded_from_audit": n_flat,
        "line_noise_50Hz_peak_dB_over_neighbors_mean_psd": peak_db(50, mean_all),
        "line_noise_50Hz_peak_dB_over_neighbors_median_psd": peak_db(50, med_all),
        "line_noise_100Hz_peak_dB_median_psd": peak_db(100, med_all),
        "per_window_50Hz_peak_dB_quantiles": {q: float(np.percentile(per_win, int(q[1:]))) for q in ("p10", "p50", "p90", "p99")},
        "frac_windows_50Hz_peak_gt_6dB": float((per_win > 6).mean()),
        "median_psd_band_dB_re_10_40Hz": band_db,
        "median_psd_5Hz_bins_dB_re_10_40Hz": curve,
        "drop_40_45_to_52_60_dB": edge_drop,
        "per_window_100_124_vs_30_45_dB_quantiles": {q: float(np.percentile(hf_rel, int(q[1:]))) for q in ("p10", "p50", "p90")},
        "by_class_median_psd": cls,
    }
    floor = res["median_psd_band_dB_re_10_40Hz"]["100-120Hz"]
    res["conclusion"] = (
        f"50 Hz 峰比邻带高 {res['line_noise_50Hz_peak_dB_over_neighbors_median_psd']:.1f} dB（中位谱）；"
        f"40-45→52-60 Hz 下降 {edge_drop:.1f} dB；100-120 Hz 相对 10-40 Hz 为 {floor:.1f} dB。"
        + ("45 Hz 以上存在明显陡降，疑似已低通滤波，高频特征主要反映残余/噪声。" if edge_drop < -20 or floor < -50
           else "45 Hz 以上未见陡降截止，存在可用的宽带信号（含 EMG/噪声）。")
    )
    return res


# ---------------- 特征 ----------------
def band_pow(f, P, lo, hi, excl=None):
    s = (f >= lo) & (f < hi)
    if excl is not None:
        s &= ~((f >= excl[0]) & (f <= excl[1]))
    return P[:, s].sum(1) * (f[1] - f[0])


def compute_features(raw, f, P):
    x = raw.astype("float64")
    x = x - x.mean(1, keepdims=True)
    eps = 1e-30
    fam = {"a_hf_emg": {}, "b_eye": {}, "c_shape": {}, "d_scaleinv": {}}

    # a) 高频/EMG
    p_1_45 = band_pow(f, P, 1, 45)
    for lo, hi in ((45, 70), (70, 100), (100, 125.1)):
        pw = band_pow(f, P, lo, hi, excl=(48, 52) if lo == 45 else (98, 102))
        nm = f"{lo}_{int(hi)}"
        fam["a_hf_emg"][f"logpow_{nm}"] = np.log(pw + eps)
        fam["a_hf_emg"][f"logratio_{nm}_vs_1_45"] = np.log(pw + eps) - np.log(p_1_45 + eps)
    hf_all = band_pow(f, P, 45, 125.1, excl=(48, 52))
    fam["a_hf_emg"]["logratio_45_125_vs_1_45"] = np.log(hf_all + eps) - np.log(p_1_45 + eps)

    # b) 眼动/大幅偏转（0.5-3 Hz 用整窗 FFT，分辨率 1/6 Hz）
    n = x.shape[1]
    spec = np.abs(np.fft.rfft(x * np.hanning(n), axis=1)) ** 2
    ff = np.fft.rfftfreq(n, 1 / FS)
    fam["b_eye"]["logpow_0p5_3"] = np.log(spec[:, (ff >= 0.5) & (ff < 3)].sum(1) + eps)
    fam["b_eye"]["logratio_0p5_3_vs_1_45"] = fam["b_eye"]["logpow_0p5_3"] - np.log(spec[:, (ff >= 1) & (ff < 45)].sum(1) + eps)
    fam["b_eye"]["log_ptp"] = np.log(np.ptp(x, 1) + eps)
    fam["b_eye"]["log_maxabs"] = np.log(np.abs(x).max(1) + eps)
    d1 = np.abs(np.diff(x, axis=1))
    fam["b_eye"]["log_maxdiff"] = np.log(d1.max(1) + eps)
    sd = x.std(1)
    fam["b_eye"]["ptp_over_std"] = np.ptp(x, 1) / np.maximum(sd, eps)
    fam["b_eye"]["maxdiff_over_meandiff"] = d1.max(1) / np.maximum(d1.mean(1), eps)
    med = np.median(x, 1, keepdims=True)
    mad = np.median(np.abs(x - med), 1, keepdims=True)
    fam["b_eye"]["frac_gt_3mad"] = (np.abs(x - med) > 3 * np.maximum(mad, eps)).mean(1)
    fam["b_eye"]["kurtosis"] = kurtosis(x, axis=1)
    fam["b_eye"]["skewness"] = skew(x, axis=1)
    fam["b_eye"]["abs_skewness"] = np.abs(fam["b_eye"]["skewness"])

    # c) 谱形状：1-40 Hz log-log 拟合（排除 48-52 无关），去非周期后峰
    s = (f >= 1) & (f <= 40)
    lf = np.log10(f[s])
    lp = np.log10(P[:, s] + eps)
    A = np.vstack([lf, np.ones_like(lf)]).T
    coef, *_ = np.linalg.lstsq(A, lp.T, rcond=None)
    slope, intercept = coef[0], coef[1]
    resid = lp - (np.outer(slope, lf) + intercept[:, None])
    fam["c_shape"]["aperiodic_slope_1_40"] = slope
    fam["c_shape"]["aperiodic_intercept_1_40"] = intercept
    fam["c_shape"]["aperiodic_fit_rmse"] = np.sqrt((resid ** 2).mean(1))
    fs_ = f[s]
    for nm, lo, hi in (("theta", 4, 8), ("alpha", 8, 13), ("beta", 13, 30)):
        b = (fs_ >= lo) & (fs_ < hi)
        r = resid[:, b]
        idx = r.argmax(1)
        fam["c_shape"][f"{nm}_peak_height"] = r.max(1)
        fam["c_shape"][f"{nm}_peak_freq"] = fs_[b][idx]
        fam["c_shape"][f"{nm}_resid_mean"] = r.mean(1)
    s45 = (f >= 1) & (f <= 45)
    p45 = P[:, s45]
    cs = np.cumsum(p45, 1) / np.maximum(p45.sum(1, keepdims=True), eps)
    f45 = f[s45]
    fam["c_shape"]["spectral_centroid_1_45"] = (p45 * f45).sum(1) / np.maximum(p45.sum(1), eps)
    fam["c_shape"]["sef50_1_45"] = f45[(cs < 0.5).sum(1).clip(max=len(f45) - 1)]
    fam["c_shape"]["sef90_1_45"] = f45[(cs < 0.9).sum(1).clip(max=len(f45) - 1)]
    s125 = (f >= 1) & (f <= 124) & ~((f >= 48) & (f <= 52))
    p125 = P[:, s125]
    cs2 = np.cumsum(p125, 1) / np.maximum(p125.sum(1, keepdims=True), eps)
    fam["c_shape"]["sef95_1_124"] = f[s125][(cs2 < 0.95).sum(1).clip(max=s125.sum() - 1)]
    fam["c_shape"]["spectral_centroid_1_124"] = (p125 * f[s125]).sum(1) / np.maximum(p125.sum(1), eps)

    # d) 尺度不变：相对功率、CLR、成对 log-ratio、softmax、排序
    bands = (("delta", 1, 4), ("theta", 4, 8), ("alpha", 8, 13), ("beta1", 13, 20),
             ("beta2", 20, 30), ("gamma", 30, 45), ("hf", 52, 100))
    bp = np.stack([band_pow(f, P, lo, hi) for _, lo, hi in bands], 1) + eps
    names = [b[0] for b in bands]
    rel = bp / bp.sum(1, keepdims=True)
    lbp = np.log(bp)
    clr = lbp - lbp.mean(1, keepdims=True)
    soft = np.exp(clr) / np.exp(clr).sum(1, keepdims=True)  # 等价于 rel，保留 CLR 温度=0.5 版本
    soft_t = np.exp(0.5 * clr) / np.exp(0.5 * clr).sum(1, keepdims=True)
    rank = lbp.argsort(1).argsort(1).astype(float)
    for i, nm in enumerate(names):
        fam["d_scaleinv"][f"relpow_{nm}"] = rel[:, i]
        fam["d_scaleinv"][f"clr_{nm}"] = clr[:, i]
        fam["d_scaleinv"][f"softmax_t0.5_{nm}"] = soft_t[:, i]
        fam["d_scaleinv"][f"rank_{nm}"] = rank[:, i]
    del soft
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            fam["d_scaleinv"][f"logratio_{names[i]}_{names[j]}"] = lbp[:, i] - lbp[:, j]
    fam["d_scaleinv"]["theta_beta_over_alpha"] = np.log((bp[:, 1] + bp[:, 3] + bp[:, 4]) / bp[:, 2])
    fam["d_scaleinv"]["spectral_entropy_1_45"] = -(rel[:, :6] / rel[:, :6].sum(1, keepdims=True) *
                                                  np.log(rel[:, :6] / rel[:, :6].sum(1, keepdims=True))).sum(1)

    out = {}
    for k, d in fam.items():
        out[k] = (np.stack(list(d.values()), 1).astype("float64"), [f"{k}:{n}" for n in d])
    return out


# ---------------- 模型 ----------------
def drop_const(Xtr, Xte):
    keep = Xtr.std(0) > 1e-12
    return Xtr[:, keep], Xte[:, keep]


def fit_logreg(Xtr, ytr, gtr, Xte):
    Xtr, Xte = drop_const(Xtr, Xte)
    inner = GroupKFold(n_splits=3)
    scores = {}
    for C in C_GRID:
        aucs = []
        for itr, iva in inner.split(Xtr, ytr, gtr):
            sc = StandardScaler().fit(Xtr[itr])
            m = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
            m.fit(sc.transform(Xtr[itr]), ytr[itr])
            aucs.append(roc_auc_score(ytr[iva], m.decision_function(sc.transform(Xtr[iva]))))
        scores[C] = float(np.mean(aucs))
    best = max(scores, key=scores.get)
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=best, max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    ptr = m.predict_proba(sc.transform(Xtr))[:, 1]
    pte = m.predict_proba(sc.transform(Xte))[:, 1]
    return pte, ptr, {"C": best, "inner_auc": scores}


def fit_hgb(Xtr, ytr, gtr, Xte):
    Xtr, Xte = drop_const(Xtr, Xte)
    pte, ptr = np.zeros(len(Xte)), np.zeros(len(Xtr))
    for s in HGB_SEEDS:
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                           min_samples_leaf=40, l2_regularization=1.0,
                                           max_features=0.5, early_stopping=False, random_state=s)
        m.fit(Xtr, ytr)
        pte += m.predict_proba(Xte)[:, 1] / len(HGB_SEEDS)
        ptr += m.predict_proba(Xtr)[:, 1] / len(HGB_SEEDS)
    return pte, ptr, {}


def evaluate(name, X, y, groups, folds, fitter):
    oof = np.zeros(len(y))
    pred05 = np.zeros(len(y), int)
    predmed = np.zeros(len(y), int)
    per_fold, info = [], []
    t = time.time()
    for k, (tr, te) in enumerate(folds):
        pte, ptr, extra = fitter(X[tr], y[tr], groups[tr], X[te])
        oof[te] = pte
        thr = float(np.median(ptr))
        pred05[te] = (pte >= 0.5).astype(int)
        predmed[te] = (pte >= thr).astype(int)
        fa = roc_auc_score(y[te], pte)
        fb = balanced_accuracy_score(y[te], pred05[te])
        fm = balanced_accuracy_score(y[te], predmed[te])
        per_fold.append({"auc": fa, "bacc_p05": fb, "bacc_trainmedian": fm, "train_median_thr": thr, **extra})
        log(f"  {name} fold {k + 1}/5  AUC={fa:.4f} bACC(0.5)={fb:.4f} bACC(med)={fm:.4f} "
            f"{('C=' + str(extra['C'])) if 'C' in extra else ''} dim={X.shape[1]}")
        info.append(extra)
    r = {
        "dim": int(X.shape[1]),
        "oof_auc": float(roc_auc_score(y, oof)),
        "oof_bacc_p05": float(balanced_accuracy_score(y, pred05)),
        "oof_bacc_trainmedian": float(balanced_accuracy_score(y, predmed)),
        "fold_mean_auc": float(np.mean([p["auc"] for p in per_fold])),
        "fold_std_auc": float(np.std([p["auc"] for p in per_fold])),
        "fold_mean_bacc_p05": float(np.mean([p["bacc_p05"] for p in per_fold])),
        "fold_mean_bacc_trainmedian": float(np.mean([p["bacc_trainmedian"] for p in per_fold])),
        "per_fold": per_fold,
        "seconds": time.time() - t,
    }
    log(f"== {name}: OOF AUC={r['oof_auc']:.4f} bACC(0.5)={r['oof_bacc_p05']:.4f} "
        f"bACC(med)={r['oof_bacc_trainmedian']:.4f} foldmean AUC={r['fold_mean_auc']:.4f} ({r['seconds']:.0f}s)")
    return r, oof


# ---------------- 方向一致率 ----------------
def direction_consistency(X, names, y, groups):
    subs = np.unique(groups)
    Xs = (X - X.mean(0)) / np.where(X.std(0) > 1e-12, X.std(0), 1.0)
    overall = np.sign(Xs[y == 1].mean(0) - Xs[y == 0].mean(0))
    signs = []
    for s in subs:
        m = groups == s
        if len(np.unique(y[m])) < 2:
            continue
        signs.append(np.sign(Xs[m & (y == 1)].mean(0) - Xs[m & (y == 0)].mean(0)))
    signs = np.array(signs)
    agree = (signs == overall).mean(0)
    pooled_d = Xs[y == 1].mean(0) - Xs[y == 0].mean(0)
    order = np.argsort(-agree)
    top = [{"feature": names[i], "agree_frac": float(agree[i]), "pooled_std_diff": float(pooled_d[i])}
           for i in order[:15]]
    return top, agree, len(signs)


def bootstrap_delta(y, groups, a, b, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    subs = np.unique(groups)
    idx = {s: np.where(groups == s)[0] for s in subs}
    d = []
    for _ in range(n):
        pick = rng.choice(subs, len(subs), replace=True)
        ii = np.concatenate([idx[s] for s in pick])
        if len(np.unique(y[ii])) < 2:
            continue
        d.append(roc_auc_score(y[ii], b[ii]) - roc_auc_score(y[ii], a[ii]))
    d = np.array(d)
    return {"delta_point": float(roc_auc_score(y, b) - roc_auc_score(y, a)),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_delta_gt0": float((d > 0).mean()), "n_boot": int(len(d))}


def main():
    log("loading data")
    d = np.load(OUT / "highdim_deep_features.npz")
    log(f"keys={d.files}")
    raw, y, groups, Xorig = d["raw"], d["y"], d["groups"], d["X"]
    orig_names = [f"orig:X_{i}" for i in range(Xorig.shape[1])]
    log(f"raw={raw.shape} X={Xorig.shape} subjects={len(np.unique(groups))}")

    log("Welch PSD (每窗独立)")
    f, P = welch(raw.astype("float64"), fs=FS, nperseg=500, noverlap=250, axis=1, detrend="constant")
    audit = spectrum_audit(f, P, y)
    log("AUDIT: " + audit["conclusion"])

    log("计算新特征族")
    fams = compute_features(raw, f, P)
    for k, (Xk, nk) in fams.items():
        bad = ~np.isfinite(Xk)
        if bad.any():
            Xk[bad] = 0.0
        log(f"  {k}: {Xk.shape[1]} 维, 非有限值 {int(bad.sum())}")
    all_new = np.hstack([fams[k][0] for k in fams])
    all_new_names = sum([fams[k][1] for k in fams], [])
    np.savez_compressed(OUT / "broadband_shape_features.npz", X=all_new.astype("float32"),
                        names=np.array(all_new_names), y=y, groups=groups,
                        **{f"fam_{k}": np.array(v[1]) for k, v in fams.items()})
    log(f"saved broadband_shape_features.npz ({all_new.shape[1]} 维)")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    folds = list(sgkf.split(Xorig, y, groups))
    seed_npz = np.load(OUT / "seed_noise_band_oof.npz")
    log(f"seed_noise_band_oof keys={seed_npz.files}")
    fold_id = np.zeros(len(y), int)
    for k, (_, te) in enumerate(folds):
        fold_id[te] = k + 1  # 与 seed_noise_band.py 一致，1-based
    same_folds = bool((fold_id == seed_npz["fold_id"]).all())
    log(f"折划分与 seed_noise_band 一致: {same_folds}")

    configs = {k: fams[k][0] for k in fams}
    configs["all_new"] = all_new
    configs["orig856"] = Xorig
    configs["new+orig856"] = np.hstack([all_new, Xorig])

    results, oofs = {}, {}
    for mname, fitter in (("logreg", fit_logreg), ("hgb", fit_hgb)):
        for cname, Xc in configs.items():
            key = f"{mname}|{cname}"
            log(f"### config {key} dim={Xc.shape[1]}")
            r, oof = evaluate(key, Xc, y, groups, folds, fitter)
            results[key] = r
            oofs[key] = oof
            with open(OUT / "broadband_shape_window6_results.json", "w", encoding="utf-8") as fh:
                json.dump({"partial": True, "configs": results}, fh, ensure_ascii=False, indent=1)

    log("方向一致率审计")
    top_new, agree_new, n_sub = direction_consistency(all_new, all_new_names, y, groups)
    keep = Xorig.std(0) > 1e-12
    top_orig, agree_orig, _ = direction_consistency(Xorig[:, keep], [n for n, k in zip(orig_names, keep) if k], y, groups)
    for t in top_new[:5]:
        log(f"  NEW  {t['feature']}: {t['agree_frac']:.3f}")
    for t in top_orig[:5]:
        log(f"  ORIG {t['feature']}: {t['agree_frac']:.3f}")

    log("bootstrap ΔAUC")
    base = seed_npz["oof_seed_mean"].astype("float64")
    cand = {k: v for k, v in results.items() if k.split("|")[1] != "orig856"}
    best_key = max(cand, key=lambda k: cand[k]["oof_auc"])
    boot = {"baseline": "seed_noise_band_oof.npz:oof_seed_mean",
            "baseline_oof_auc": float(roc_auc_score(y, base)),
            "baseline_oof_bacc_p05": float(balanced_accuracy_score(y, (base >= 0.5).astype(int))),
            "best_new_config": best_key,
            "best_vs_seedmean": bootstrap_delta(y, groups, base, oofs[best_key])}
    best_new_only = max([k for k in cand if "orig" not in k], key=lambda k: cand[k]["oof_auc"])
    boot["best_new_only_config"] = best_new_only
    boot["best_new_only_vs_seedmean"] = bootstrap_delta(y, groups, base, oofs[best_new_only])
    for m in ("logreg", "hgb"):
        boot[f"{m}_new+orig_vs_{m}_orig856"] = bootstrap_delta(y, groups, oofs[f"{m}|orig856"], oofs[f"{m}|new+orig856"])
    for k, v in boot.items():
        if isinstance(v, dict):
            log(f"  {k}: Δ={v['delta_point']:+.4f} CI=[{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}] P(Δ>0)={v['p_delta_gt0']:.3f}")
    boot["note"] = "种子噪声带 AUC 0.6092±0.0108；种子平均基线 0.6183/0.5870；<0.03 的差异不可靠"

    out = {
        "protocol": {
            "test_subject_isolation": True, "test_history": False,
            "window_position_schedule_r_t": False, "test_subject_calibration": False,
            "features": "仅当前 6 秒窗 raw（1500 点）逐窗计算",
            "outer_cv": "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0), groups=受试者",
            "folds_identical_to_seed_noise_band": same_folds,
            "logreg": "L2, class_weight=balanced, 训练折内 StandardScaler, 内层 GroupKFold(3) 按受试者选 C∈{0.001,0.01,0.1,1}(AUC)",
            "hgb": "max_iter=200, lr=0.05, max_leaf_nodes=15, min_samples_leaf=40, l2=1.0, max_features=0.5, 无早停, 种子 0/1/2 概率平均, 无调参",
            "constant_columns": "每折按训练折剔除常量列",
            "bacc_threshold": "bacc_p05: 固定 0.5 概率阈值；bacc_trainmedian: 训练折内样本预测概率中位数为阈值（仅用训练折）",
            "direction_consistency": "全体 z-score 后，每受试者 task-relax 均值差符号与总体符号一致的受试者比例（描述性审计，非模型选择）",
            "bootstrap": f"{N_BOOT} 次受试者级有放回重采样, ΔAUC = 新配置 − 基线",
        },
        "spectrum_audit": audit,
        "feature_families": {k: {"dim": int(v[0].shape[1]), "names": v[1]} for k, v in fams.items()},
        "configs": results,
        "summary_table": {k: {m: round(v[m], 4) for m in ("oof_auc", "oof_bacc_p05", "oof_bacc_trainmedian",
                                                          "fold_mean_auc", "fold_mean_bacc_p05")} for k, v in results.items()},
        "direction_consistency": {"n_subjects_both_classes": n_sub,
                                  "new_top15": top_new, "orig856_top15": top_orig,
                                  "new_mean_agree": float(agree_new.mean()), "orig_mean_agree": float(agree_orig.mean()),
                                  "new_max_agree": float(agree_new.max()), "orig_max_agree": float(agree_orig.max())},
        "bootstrap": boot,
        "elapsed_seconds": time.time() - T0,
    }
    with open(OUT / "broadband_shape_window6_results.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    np.savez_compressed(OUT / "broadband_shape_window6_oof.npz", y=y, groups=groups, fold_id=fold_id,
                        **{k.replace("|", "__").replace("+", "_plus_"): v.astype("float32") for k, v in oofs.items()})
    log(f"done, total {time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
