"""Strictly causal location/scale normalization of the best split6 OOF score."""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logit
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

OUT = Path(__file__).resolve().parent
SOURCE = "mean_std_p10_p90__fusion_score"
WARMUP = 120
MIN_HISTORY = 20
SMOOTHS = (10, 5)  # s10 is primary; s5 is the preregistered robustness check.
VARIANTS = ("location", "mad", "iqr")
SCALE_FLOOR = 0.05  # Logit units; fixed before evaluation.
SEED = 0
N_BOOT = 2000
START = time.perf_counter()


class Tee:
    def __init__(self, stream, path):
        self.stream = stream
        self.file = path.open("w", encoding="utf-8")

    def write(self, text):
        self.stream.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def log(message):
    print(f"[elapsed {time.perf_counter() - START:6.2f}s] {message}", flush=True)


def causal_smooth(values, groups, width):
    out = np.full(values.shape, np.nan, dtype=np.float64)
    for subject in np.unique(groups):
        idx = np.flatnonzero(groups == subject)
        x = values[idx]
        for t, i in enumerate(idx):
            history = x[max(0, t - width + 1):t + 1]
            history = history[np.isfinite(history)]
            if history.size:
                out[i] = history.mean()
    return out


def causal_normalize(values, groups, mask, kind):
    """At t, read only earlier evaluated scores from the same subject."""
    x = logit(np.clip(values, 1e-6, 1.0 - 1e-6))
    out = np.full(values.shape, np.nan, dtype=np.float64)
    for subject in np.unique(groups):
        idx = np.flatnonzero((groups == subject) & mask & np.isfinite(x))
        for t, i in enumerate(idx):
            if t < MIN_HISTORY:
                center, scale = 0.0, 1.0
            else:
                history = x[idx[:t]]
                center = float(np.median(history))
                if kind == "location":
                    scale = 1.0
                elif kind == "mad":
                    scale = 1.4826 * float(np.median(np.abs(history - center)))
                elif kind == "iqr":
                    q25, q75 = np.percentile(history, (25, 75))
                    scale = float(q75 - q25) / 1.3489795003921634
                else:
                    raise ValueError(kind)
                scale = max(scale, SCALE_FLOOR)
            out[i] = (x[i] - center) / scale
    return out


def per_subject_bacc(y, groups, mask, pred):
    values = []
    for subject in np.unique(groups[mask]):
        subject_mask = mask & (groups == subject)
        if np.unique(y[subject_mask]).size == 2:
            values.append(balanced_accuracy_score(y[subject_mask], pred[subject_mask]))
    return np.asarray(values, dtype=np.float64)


def metric_block(y, groups, mask, raw_score, pred):
    macro_values = per_subject_bacc(y, groups, mask, pred)
    return {
        "subject_macro_bACC": float(macro_values.mean()),
        "window_micro_bACC": float(balanced_accuracy_score(y[mask], pred[mask])),
        "AUC": float(roc_auc_score(y[mask], raw_score[mask])),
        "n_eval": int(mask.sum()),
        "n_subjects": int(np.unique(groups[mask]).size),
        "n_macro_subjects": int(macro_values.size),
    }


def select_threshold(y, groups, mask, score):
    """Maximize training-subject macro bACC; break ties by proximity to zero."""
    train_subjects = np.unique(groups[mask])
    candidates = np.unique(np.r_[0.0, np.quantile(score[mask], np.linspace(0.02, 0.98, 97))])
    best_value, best_threshold = -np.inf, np.nan
    for threshold in candidates:
        pred = score >= threshold
        values = []
        for subject in train_subjects:
            subject_mask = mask & (groups == subject)
            if np.unique(y[subject_mask]).size == 2:
                values.append(balanced_accuracy_score(y[subject_mask], pred[subject_mask]))
        value = float(np.mean(values))
        if value > best_value + 1e-12 or (
                abs(value - best_value) <= 1e-12
                and abs(threshold) < abs(best_threshold)):
            best_value, best_threshold = value, float(threshold)
    return best_threshold, best_value


def paired_bootstrap(y, groups, mask, baseline_pred, variant_pred):
    baseline = per_subject_bacc(y, groups, mask, baseline_pred)
    variant = per_subject_bacc(y, groups, mask, variant_pred)
    assert baseline.shape == variant.shape
    delta = variant - baseline
    rng = np.random.default_rng(SEED)
    sampled = delta[rng.integers(0, delta.size, size=(N_BOOT, delta.size))].mean(axis=1)
    return {
        "mean_delta": float(delta.mean()),
        "ci95": [float(v) for v in np.percentile(sampled, (2.5, 97.5))],
        "n_subjects": int(delta.size),
        "n_bootstrap": N_BOOT,
    }


def main():
    tee = Tee(sys.stdout, OUT / "causal_score_normalization.log")
    sys.stdout = tee
    try:
        with np.load(OUT / "history_split6_rich_oof.npz", allow_pickle=False) as rich:
            y = rich["y"].astype(np.int64)
            groups = rich["groups"]
            raw = rich[SOURCE].astype(np.float64)
        with np.load(OUT / "history_leakfree_oof.npz", allow_pickle=False) as old:
            assert np.array_equal(y, old["y"])
            assert np.array_equal(groups, old["groups"])
            eval_mask = old["eval_mask"].astype(bool)

        n = len(y)
        assert n == 16067 and raw.shape == (n,) and set(np.unique(y)) == {0, 1}
        position = np.empty(n, dtype=np.int64)
        for subject in np.unique(groups):
            idx = np.flatnonzero(groups == subject)
            assert np.array_equal(idx, np.arange(idx[0], idx[-1] + 1))
            position[idx] = np.arange(idx.size)
        assert np.array_equal(eval_mask, position >= WARMUP)
        splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(y, y, groups))
        fold_id = np.full(n, -1, dtype=np.int64)
        for fold, (train_idx, test_idx) in enumerate(splits):
            assert not set(groups[train_idx]).intersection(groups[test_idx])
            fold_id[test_idx] = fold
        assert np.all(fold_id >= 0)
        for subject in np.unique(groups):
            assert np.unique(fold_id[groups == subject]).size == 1

        result = {
            "meta": {
                "source": SOURCE,
                "primary_smooth": 10,
                "robustness_smooth": 5,
                "warmup": WARMUP,
                "min_history": MIN_HISTORY,
                "variants": list(VARIANTS),
                "observation": "logit(clip(score,1e-6,1-1e-6))",
                "scale_floor_logit": SCALE_FLOOR,
                "threshold_selection": "per test fold: maximum subject-macro bACC on other-fold OOF; 97 quantile candidates plus zero",
                "tie_break": "threshold closest to zero",
                "strict_subject_disjoint": True,
                "strict_causal_history": True,
                "test_labels_used_for_selection": False,
                "bootstrap": "paired subject resampling, seed=0",
            },
            "results": {},
            "folds": {},
        }
        saved = {"y": y, "groups": groups, "eval_mask": eval_mask, "raw": raw}
        for width in SMOOTHS:
            tag = f"s{width}"
            score = causal_smooth(raw, groups, width)
            valid = eval_mask & np.isfinite(score)
            assert int(valid.sum()) == 9227
            baseline_z = causal_normalize(score, groups, eval_mask, "location")
            baseline_pred = np.zeros(n, dtype=np.int8)
            baseline_pred[valid] = (baseline_z[valid] >= 0.0).astype(np.int8)
            baseline_metrics = metric_block(y, groups, valid, score, baseline_pred)
            result["results"][tag] = {"causal_median_baseline": baseline_metrics, "variants": {}}
            result["folds"][tag] = {}
            saved[f"{tag}__score"] = score
            saved[f"{tag}__baseline_pred"] = baseline_pred
            log(f"{tag} baseline macro={baseline_metrics['subject_macro_bACC']:.6f} "
                f"micro={baseline_metrics['window_micro_bACC']:.6f} AUC={baseline_metrics['AUC']:.6f}")

            for kind in VARIANTS:
                normalized = causal_normalize(score, groups, eval_mask, kind)
                assert np.array_equal(np.isfinite(normalized), valid)
                fixed_pred = np.zeros(n, dtype=np.int8)
                fixed_pred[valid] = (normalized[valid] >= 0.0).astype(np.int8)
                assert np.array_equal(fixed_pred, baseline_pred)
                crossfit_pred = np.zeros(n, dtype=np.int8)
                fold_summaries = []
                for fold in range(5):
                    train = valid & (fold_id != fold)
                    test = valid & (fold_id == fold)
                    threshold, train_macro = select_threshold(y, groups, train, normalized)
                    crossfit_pred[test] = (normalized[test] >= threshold).astype(np.int8)
                    fold_summaries.append({
                        "fold": fold,
                        "threshold": threshold,
                        "train_subject_macro_bACC": train_macro,
                        "n_train": int(train.sum()),
                        "n_test": int(test.sum()),
                        "train_subjects": [str(v) for v in np.unique(groups[train])],
                        "test_subjects": [str(v) for v in np.unique(groups[test])],
                    })
                fixed_metrics = metric_block(y, groups, valid, score, fixed_pred)
                crossfit_metrics = metric_block(y, groups, valid, score, crossfit_pred)
                bootstrap = paired_bootstrap(y, groups, valid, baseline_pred, crossfit_pred)
                result["results"][tag]["variants"][kind] = {
                    "fixed_zero": fixed_metrics,
                    "crossfit_train_threshold": crossfit_metrics,
                    "paired_bootstrap_vs_baseline": bootstrap,
                }
                result["folds"][tag][kind] = fold_summaries
                saved[f"{tag}__{kind}__normalized"] = normalized
                saved[f"{tag}__{kind}__crossfit_pred"] = crossfit_pred
                log(f"{tag} {kind:8s} crossfit macro={crossfit_metrics['subject_macro_bACC']:.6f} "
                    f"micro={crossfit_metrics['window_micro_bACC']:.6f} "
                    f"delta={bootstrap['mean_delta']:+.6f} CI={bootstrap['ci95']}")

        result["seconds"] = time.perf_counter() - START
        np.savez_compressed(OUT / "causal_score_normalization_oof.npz", **saved)
        (OUT / "causal_score_normalization_results.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"done seconds={result['seconds']:.2f}")
    finally:
        sys.stdout = tee.stream
        tee.close()


if __name__ == "__main__":
    main()
