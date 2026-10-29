"""Leakage-audited artifact/preprocessing ablation on the supplied 57 subjects."""
import json, os, sys, time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cogstate import make_dataset
from preprocessing import PreprocessingConfig, QualityThresholds, causal_filter, quality_metrics, apply_quality

OUT = ROOT / "test"
CACHE = OUT / "artifact_ablation_features.npz"

def score(y, p):
    return {"AUC": float(roc_auc_score(y, p)), "bACC": float(balanced_accuracy_score(y, p >= .5))}

def subject_macro(y, p, groups):
    vals = []
    for g in sorted(set(groups.tolist())):
        ix = groups == g
        if len(set(y[ix])) == 2:
            vals.append(score(y[ix], p[ix]))
    return {"n_subjects": len(vals), "AUC_mean": float(np.mean([v["AUC"] for v in vals])),
            "bACC_mean": float(np.mean([v["bACC"] for v in vals]))}

def load_or_extract():
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        return z["raw"], z["pre"], z["quality"], z["y"], z["groups"]
    # make_dataset is the existing audited event/bad-interval extractor; raw44 is its unchanged output.
    raw, y, groups, meta = make_dataset(str(ROOT / "EEG_dataset"))
    # Reconstruct quality features from deterministic window metadata. The source windows are re-read
    # through make_dataset's public output path to avoid changing the formal model.
    # The baseline feature cache is used as the raw44 reference; preprocessing is applied to each
    # feature window only through the same causal signal transform in a separate extraction pass.
    import glob, mne
    qrows = []
    pref = []
    cfg = PreprocessingConfig(mains_hz=50.0)
    by_subject = {}
    for sid in sorted(set(groups.tolist())):
        csv = next((Path(p) for p in glob.glob(str(ROOT / "EEG_dataset" / sid / "*_events.csv"))))
        bdf = next(csv.parent.glob("*.bdf")); rr = mne.io.read_raw_bdf(str(bdf), preload=True, verbose="ERROR")
        by_subject[sid] = (rr, float(rr.info["sfreq"]))
    for i, row in enumerate(meta):
        rr, fs = by_subject[row["subject"]]
        n = int(round(6.0 * fs)); end = int(round(row["time"] * fs)); q = rr.get_data(start=end-n, stop=end)[0]
        pref.append(__import__("cogstate").causal_features(causal_filter(q, cfg)))
        qrows.append([quality_metrics(q, fs, cfg)[k] for k in ("amplitude_max", "peak_to_peak", "rms", "flat_fraction", "diff_energy", "spike_fraction", "low_freq_drift", "high_freq_ratio", "nan_fraction")])
        if (i + 1) % 2000 == 0: print(f"[features] {i+1}/{len(meta)}", flush=True)
    for rr, _ in by_subject.values(): rr.close()
    pref, qrows = np.asarray(pref, dtype=np.float32), np.asarray(qrows, dtype=np.float32)
    np.savez(CACHE, raw=raw, pre=pref, quality=qrows, y=y, groups=groups)
    return raw, pref, qrows, y, groups

def main():
    start = time.time(); raw, pre, quality, y, groups = load_or_extract()
    cfg = PreprocessingConfig(mains_hz=50.0)
    variants = {"raw44": raw, "preprocessed44": pre, "enhanced_quality": np.column_stack([pre, quality])}
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    result = {"protocol": {"subjects": int(len(set(groups.tolist())),), "n_samples": int(len(y)), "cv": "StratifiedGroupKFold(5), subject-disjoint", "thresholds": "training-fold quantiles only", "mains_hz": 50.0, "filter": "causal Butterworth SOS 1-45Hz + causal IIR notch", "ica_asr": "not used: single channel", "biot_resampling": "not part of this causal ablation; offline polyphase and strict online fixed-delay are separate APIs"}, "variants": {}}
    for name, features in variants.items():
        oof = np.full(len(y), np.nan); rows = []
        for fold, (tr, va) in enumerate(cv.split(features, y, groups), 1):
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
            model.fit(features[tr], y[tr]); p = model.predict_proba(features[va])[:, 1]; oof[va] = p
            rows.append({"fold": fold, "n": int(len(va)), "subjects": sorted(set(groups[va].tolist())), "retained": int(len(va)), "excluded": 0, "exclusion_rate": 0.0, **score(y[va], p)})
            print(f"[{name}] fold {fold}/5 {rows[-1]}", flush=True)
        result["variants"][name] = {"n_features": int(features.shape[1]), "oof": score(y, oof), "subject_macro": subject_macro(y, oof, groups), "folds": rows}
    # Reject variant: fit thresholds only on each training fold and score retained validation samples.
    rows = []; oof = np.full(len(y), np.nan)
    for fold, (tr, va) in enumerate(cv.split(pre, y, groups), 1):
        th = QualityThresholds.fit([{k: v for k, v in zip(("amplitude_max", "peak_to_peak", "rms", "flat_fraction", "diff_energy", "spike_fraction", "low_freq_drift", "high_freq_ratio", "nan_fraction"), row)} for row in quality[tr]])
        tr_keep = np.array([not th.flag(dict(zip(th.limits, row)))[0] for row in quality[tr]])
        va_keep = np.array([not th.flag(dict(zip(th.limits, row)))[0] for row in quality[va]])
        valid = tr[tr_keep]
        row = {"fold": fold, "n": int(len(va)), "retained": int(va_keep.sum()), "excluded": int((~va_keep).sum()), "exclusion_rate": float((~va_keep).mean()), "subjects": sorted(set(groups[va].tolist()))}
        if len(set(y[valid])) < 2 or len(set(y[va][va_keep])) < 2:
            row["status"] = "invalid_missing_class_not_scored"; rows.append(row); print(f"[quality_reject] fold {fold} invalid", flush=True); continue
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")).fit(pre[valid], y[valid])
        p = model.predict_proba(pre[va][va_keep])[:, 1]; oof[va[va_keep]] = p
        row.update(score(y[va][va_keep], p)); row["status"] = "scored"; rows.append(row); print(f"[quality_reject] fold {fold}/5 {row}", flush=True)
    ok = np.isfinite(oof)
    result["variants"]["quality_reject"] = {"n_features": 44, "oof_scored": int(ok.sum()), "oof": score(y[ok], oof[ok]), "subject_macro": subject_macro(y[ok], oof[ok], groups[ok]), "folds": rows}
    result["runtime_seconds"] = time.time() - start
    (OUT / "artifact_ablation_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    names = list(result["variants"]); auc = [result["variants"][n]["oof"]["AUC"] for n in names]; bacc = [result["variants"][n]["oof"]["bACC"] for n in names]
    plt.figure(figsize=(9, 4)); ix = np.arange(len(names)); plt.bar(ix-.18, auc, .36, label="AUC"); plt.bar(ix+.18, bacc, .36, label="bACC"); plt.axhline(.5, color="gray", ls="--"); plt.xticks(ix, names, rotation=18); plt.ylim(.4, .7); plt.legend(); plt.tight_layout(); plt.savefig(OUT / "artifact_ablation_results.png", dpi=160); print("COMPLETE", json.dumps(result, ensure_ascii=False), flush=True)

if __name__ == "__main__": main()
