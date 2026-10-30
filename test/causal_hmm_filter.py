"""Subject-disjoint, strictly causal online two-state HMM filtering."""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logit, logsumexp
from scipy.stats import norm, t as student_t
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

OUT = Path(__file__).resolve().parent
SOURCE = "mean_std_p10_p90__fusion_score"
WARMUP = 120
SEED = 0
N_SPLITS = 5
DF = 5
SCALE_FLOOR = 1e-6
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
    print(f"[elapsed {time.perf_counter() - START:7.2f}s] {message}", flush=True)


def metric_block(y, groups, mask, score, pred):
    per_subject = []
    for subject in np.unique(groups[mask]):
        subject_mask = mask & (groups == subject)
        if np.unique(y[subject_mask]).size == 2:
            per_subject.append(balanced_accuracy_score(y[subject_mask], pred[subject_mask]))
    return {
        "subject_macro_bACC": float(np.mean(per_subject)),
        "window_micro_bACC": float(balanced_accuracy_score(y[mask], pred[mask])),
        "AUC": float(roc_auc_score(y[mask], score[mask])),
        "n_eval": int(mask.sum()),
        "n_subjects": int(np.unique(groups[mask]).size),
        "n_macro_subjects": len(per_subject),
    }


def stationary_distribution(transition):
    values, vectors = np.linalg.eig(transition.T)
    vector = np.real(vectors[:, np.argmin(np.abs(values - 1.0))])
    vector *= np.sign(vector.sum())
    vector /= vector.sum()
    assert np.all(vector > 0) and np.isclose(vector.sum(), 1.0)
    return vector


def fit_fold(fold, y, groups, eval_mask, fold_id, x):
    train = eval_mask & (fold_id != fold) & np.isfinite(x)
    train_subjects = np.unique(groups[fold_id != fold])
    test_subjects = np.unique(groups[fold_id == fold])
    assert not set(train_subjects).intersection(test_subjects)

    counts = np.zeros((2, 2), dtype=np.int64)
    for subject in train_subjects:
        idx = np.flatnonzero(groups == subject)
        labels = y[idx]
        valid = eval_mask[idx]
        adjacent = valid[:-1] & valid[1:]
        previous = labels[:-1][adjacent]
        current = labels[1:][adjacent]
        np.add.at(counts, (previous, current), 1)
    transition = (counts + 1.0) / (counts + 1.0).sum(axis=1, keepdims=True)
    initial = stationary_distribution(transition)

    emissions = {"gaussian": [], "student_t": []}
    class_counts = []
    for state in (0, 1):
        values = x[train & (y == state)]
        assert values.size >= 2
        class_counts.append(int(values.size))
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        emissions["gaussian"].append({
            "mean": float(np.mean(values)),
            "std": max(float(np.std(values)), SCALE_FLOOR),
        })
        emissions["student_t"].append({
            "location": median,
            "scale": max(mad / 0.67448975, SCALE_FLOOR),
        })

    summary = {
        "fold": fold,
        "train_subjects": [str(v) for v in train_subjects],
        "test_subjects": [str(v) for v in test_subjects],
        "n_emission_train": int(train.sum()),
        "emission_class_counts": class_counts,
        "transition_counts": counts.tolist(),
        "transition_matrix": transition.tolist(),
        "initial_probability": initial.tolist(),
        "emissions": emissions,
    }
    return transition, initial, emissions, summary


def emission_logpdf(kind, value, parameters):
    if kind == "gaussian":
        return np.array([
            norm.logpdf(value, loc=p["mean"], scale=p["std"])
            for p in parameters
        ])
    return np.array([
        student_t.logpdf(value, df=DF, loc=p["location"], scale=p["scale"])
        for p in parameters
    ])


def online_filter(values, transition, initial, kind, emissions):
    """Return P(state=1 | observations up to t), never reading future rows."""
    output = np.full(values.size, np.nan, dtype=np.float64)
    posterior = None
    for t, value in enumerate(values):
        if posterior is None:
            if not np.isfinite(value):
                continue
            predicted = initial.copy()
        else:
            predicted = posterior @ transition
            predicted /= predicted.sum()

        if np.isfinite(value):
            log_posterior = (np.log(np.maximum(predicted, 1e-300))
                             + emission_logpdf(kind, value, emissions[kind]))
            log_posterior -= logsumexp(log_posterior)
            posterior = np.exp(log_posterior)
        else:
            # Missing observation: transition prediction only.
            posterior = predicted
        assert np.all(np.isfinite(posterior)) and np.isclose(posterior.sum(), 1.0)
        output[t] = posterior[1]
    return output


def main():
    tee = Tee(sys.stdout, OUT / "causal_hmm_filter.log")
    sys.stdout = tee
    try:
        with np.load(OUT / "history_split6_rich_oof.npz", allow_pickle=False) as rich:
            y = rich["y"].astype(np.int64)
            groups = rich["groups"]
            rich_mask = rich["eval_mask"].astype(bool)
            raw = rich[SOURCE].astype(np.float64)
        with np.load(OUT / "history_leakfree_oof.npz", allow_pickle=False) as old:
            assert np.array_equal(y, old["y"])
            assert np.array_equal(groups, old["groups"])
            old_mask = old["eval_mask"].astype(bool)

        n = len(y)
        assert n == 16067 and raw.shape == (n,)
        assert set(np.unique(y)) == {0, 1}
        position = np.empty(n, dtype=np.int64)
        for subject in np.unique(groups):
            idx = np.flatnonzero(groups == subject)
            assert np.array_equal(idx, np.arange(idx[0], idx[-1] + 1)), \
                "Subject rows must be contiguous and non-interleaved in time order"
            position[idx] = np.arange(idx.size)
        eval_mask = position >= WARMUP
        assert np.array_equal(eval_mask, old_mask)
        assert np.all(~eval_mask | rich_mask), "Rich mask must contain the canonical mask"
        rich_mask_mismatch = not np.array_equal(rich_mask, eval_mask)
        if rich_mask_mismatch:
            log("audit: rich saved eval_mask is a superset; using canonical position>=120 "
                f"mask ({eval_mask.sum()} rows), identical to leakfree and s10 baseline")

        x = logit(np.clip(raw, 1e-6, 1.0 - 1e-6))
        splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        splits = list(splitter.split(y, y, groups))
        fold_id = np.full(n, -1, dtype=np.int64)
        for fold, (train_idx, test_idx) in enumerate(splits):
            fold_id[test_idx] = fold
            assert not set(groups[train_idx]).intersection(groups[test_idx])
        assert np.all(fold_id >= 0)
        for subject in np.unique(groups):
            assert np.unique(fold_id[groups == subject]).size == 1

        scores = {kind: np.full(n, np.nan, dtype=np.float64)
                  for kind in ("gaussian", "student_t")}
        fold_summaries = []
        for fold in range(N_SPLITS):
            transition, initial, emissions, summary = fit_fold(
                fold, y, groups, eval_mask, fold_id, x)
            fold_summaries.append(summary)
            for subject in summary["test_subjects"]:
                idx = np.flatnonzero(groups == subject)
                for kind in scores:
                    scores[kind][idx] = online_filter(
                        x[idx], transition, initial, kind, emissions)
            log(f"fold {fold + 1}/5 train={summary['n_emission_train']} "
                f"transitions={int(np.sum(summary['transition_counts']))} "
                f"test_subjects={len(summary['test_subjects'])}")

        variants = {}
        valid_masks = []
        for kind, score in scores.items():
            assert np.any(np.isfinite(score)) and not np.any(np.isinf(score))
            valid = eval_mask & np.isfinite(score)
            valid_masks.append(valid)
            assert np.all(np.isfinite(raw[valid])), "Raw control requires identical finite rows"
            pred = np.full(n, -1, dtype=np.int8)
            pred[np.isfinite(score)] = (score[np.isfinite(score)] >= 0.5).astype(np.int8)
            variants[kind] = metric_block(y, groups, valid, score, pred)
            variants[kind]["raw_fixed_0.5"] = metric_block(
                y, groups, valid, raw, (raw >= 0.5).astype(np.int8))
            log(f"{kind}: macro={variants[kind]['subject_macro_bACC']:.6f} "
                f"micro={variants[kind]['window_micro_bACC']:.6f} "
                f"AUC={variants[kind]['AUC']:.6f} n_eval={valid.sum()}")
        assert np.array_equal(valid_masks[0], valid_masks[1])

        baseline_valid = eval_mask & np.isfinite(raw)
        baseline_n_eval = int(baseline_valid.sum())
        same_baseline_rows = all(np.array_equal(valid, baseline_valid) for valid in valid_masks)
        result = {
            "meta": {
                "source": SOURCE,
                "observation": "logit(clip(score,1e-6,1-1e-6)); no pre-smoothing",
                "warmup": WARMUP,
                "folds": N_SPLITS,
                "seed": SEED,
                "threshold": 0.5,
                "student_t_df": DF,
                "scale_floor": SCALE_FLOOR,
                "transition_smoothing": "2x2 counts + 1 Laplace",
                "initial_probability": "stationary distribution via eig(A.T)",
                "strict_subject_disjoint": True,
                "strict_online_forward_filtering": True,
                "rich_mask_differs_from_canonical": rich_mask_mismatch,
                "canonical_mask": "within-subject array position >= 120; equals leakfree eval_mask",
                "undefined_prediction_value": -1,
            },
            "folds": fold_summaries,
            "variants": variants,
            "baseline_reference": {
                "name": "split6_best s10 causal median",
                "subject_macro_bACC": 0.7851981510051427,
                "window_micro_bACC": 0.7857248463262942,
                "AUC": 0.8085795468018964,
                "n_eval": baseline_n_eval,
                "same_eval_rows": same_baseline_rows,
                "directly_comparable": same_baseline_rows,
            },
        }

        saved = {
            "y": y,
            "groups": groups,
            "eval_mask": eval_mask,
            "fold_id": fold_id,
            "x": x,
        }
        for kind, score in scores.items():
            pred = np.full(n, -1, dtype=np.int8)
            finite = np.isfinite(score)
            pred[finite] = (score[finite] >= 0.5).astype(np.int8)
            saved[f"{kind}_filtered_score"] = score
            saved[f"{kind}_prediction"] = pred
        np.savez_compressed(OUT / "causal_hmm_filter_oof.npz", **saved)
        result["seconds"] = time.perf_counter() - START
        (OUT / "causal_hmm_filter_results.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"done; outputs saved, seconds={result['seconds']:.2f}")
    finally:
        sys.stdout = tee.stream
        tee.close()


if __name__ == "__main__":
    main()
