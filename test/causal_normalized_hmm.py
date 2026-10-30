"""Nested causal HMM filtering after robust per-subject score normalization."""
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
SMOOTHS = (10, 5)  # s10 primary; s5 preregistered robustness check.
KINDS = ("gaussian", "student_t")
DF = 5
BLEND_WEIGHT = 0.5
SCALE_FLOOR = 1e-6
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


def stationary_distribution(transition):
    values, vectors = np.linalg.eig(transition.T)
    vector = np.real(vectors[:, np.argmin(np.abs(values - 1.0))])
    vector *= np.sign(vector.sum())
    vector /= vector.sum()
    assert np.all(vector > 0) and np.isclose(vector.sum(), 1.0)
    return vector


def fit_parameters(y, groups, fit_mask, x):
    """Fit only on source subjects; no target-subject labels enter this function."""
    counts = np.zeros((2, 2), dtype=np.int64)
    for subject in np.unique(groups[fit_mask]):
        idx = np.flatnonzero(groups == subject)
        valid = fit_mask[idx]
        labels = y[idx]
        adjacent = valid[:-1] & valid[1:]
        previous = labels[:-1][adjacent]
        current = labels[1:][adjacent]
        np.add.at(counts, (previous, current), 1)
    transition = (counts + 1.0) / (counts + 1.0).sum(axis=1, keepdims=True)
    initial = stationary_distribution(transition)
    emissions = {kind: [] for kind in KINDS}
    for state in (0, 1):
        values = x[fit_mask & (y == state)]
        assert values.size >= 2 and np.all(np.isfinite(values))
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        emissions["gaussian"].append({
            "mean": float(np.mean(values)),
            "std": max(float(np.std(values)), SCALE_FLOOR),
        })
        emissions["student_t"].append({
            "location": median,
            "scale": max(mad / student_t.ppf(0.75, DF), SCALE_FLOOR),
        })
    summary = {
        "n_fit": int(fit_mask.sum()),
        "fit_subjects": [str(v) for v in np.unique(groups[fit_mask])],
        "transition_counts": counts.tolist(),
        "transition_matrix": transition.tolist(),
        "initial_probability": initial.tolist(),
        "emissions": emissions,
    }
    return transition, initial, emissions, summary


def emission_logpdf(kind, value, parameters):
    if kind == "gaussian":
        return np.array([norm.logpdf(value, loc=p["mean"], scale=p["std"])
                         for p in parameters])
    return np.array([student_t.logpdf(value, df=DF, loc=p["location"], scale=p["scale"])
                     for p in parameters])


def online_filter(values, transition, initial, kind, emissions):
    """P(state=1 | observations up to t); missing rows only propagate state."""
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
                             + emission_logpdf(kind, value, emissions))
            log_posterior -= logsumexp(log_posterior)
            posterior = np.exp(log_posterior)
        else:
            posterior = predicted
        assert np.all(np.isfinite(posterior)) and np.isclose(posterior.sum(), 1.0)
        output[t] = posterior[1]
    return output


def filter_subjects(x, groups, subjects, transition, initial, kind, emissions):
    output = np.full(x.shape, np.nan, dtype=np.float64)
    for subject in subjects:
        idx = np.flatnonzero(groups == subject)
        output[idx] = online_filter(x[idx], transition, initial, kind, emissions[kind])
    return output


def per_subject_bacc(y, groups, mask, pred):
    values = []
    for subject in np.unique(groups[mask]):
        subject_mask = mask & (groups == subject)
        if np.unique(y[subject_mask]).size == 2:
            values.append(balanced_accuracy_score(y[subject_mask], pred[subject_mask]))
    return np.asarray(values, dtype=np.float64)


def metric_block(y, groups, mask, score, pred):
    macro = per_subject_bacc(y, groups, mask, pred)
    return {
        "subject_macro_bACC": float(macro.mean()),
        "window_micro_bACC": float(balanced_accuracy_score(y[mask], pred[mask])),
        "AUC": float(roc_auc_score(y[mask], score[mask])),
        "n_eval": int(mask.sum()),
        "n_subjects": int(np.unique(groups[mask]).size),
        "n_macro_subjects": int(macro.size),
    }


def select_threshold(y, groups, mask, score):
    candidates = np.unique(np.r_[0.0, np.quantile(score[mask], np.linspace(0.02, 0.98, 97))])
    best_value, best_threshold = -np.inf, np.nan
    for threshold in candidates:
        value = float(per_subject_bacc(y, groups, mask, score >= threshold).mean())
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


def causality_check(x, groups, subject, transition, initial, kind, emissions):
    idx = np.flatnonzero(groups == subject)
    values = x[idx]
    finite = np.flatnonzero(np.isfinite(values))
    cut = int(finite[min(30, finite.size - 1)])
    perturbed = values.copy()
    perturbed[cut:] = perturbed[cut:] + 100.0
    before = online_filter(values, transition, initial, kind, emissions)
    after = online_filter(perturbed, transition, initial, kind, emissions)
    assert np.array_equal(before[:cut], after[:cut], equal_nan=True)
    return {"subject": str(subject), "cut_within_subject": cut,
            "past_outputs_bitwise_identical": True}


def main():
    tee = Tee(sys.stdout, OUT / "causal_normalized_hmm.log")
    sys.stdout = tee
    try:
        with np.load(OUT / "causal_score_normalization_oof.npz", allow_pickle=False) as data:
            y = data["y"].astype(np.int64)
            groups = data["groups"]
            eval_mask = data["eval_mask"].astype(bool)
            normalized = {width: data[f"s{width}__mad__normalized"].astype(np.float64)
                          for width in SMOOTHS}
            smoothed = {width: data[f"s{width}__score"].astype(np.float64)
                        for width in SMOOTHS}
            direct_reference = {width: data[f"s{width}__mad__crossfit_pred"].astype(np.int8)
                                for width in SMOOTHS}
            median_reference = {width: data[f"s{width}__baseline_pred"].astype(np.int8)
                                for width in SMOOTHS}

        n = len(y)
        assert n == 16067 and set(np.unique(y)) == {0, 1}
        splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(y, y, groups))
        fold_id = np.full(n, -1, dtype=np.int64)
        for fold, (train_idx, test_idx) in enumerate(splits):
            assert not set(groups[train_idx]).intersection(groups[test_idx])
            fold_id[test_idx] = fold
        assert np.all(fold_id >= 0)
        for subject in np.unique(groups):
            idx = np.flatnonzero(groups == subject)
            assert np.array_equal(idx, np.arange(idx[0], idx[-1] + 1))
            assert np.unique(fold_id[idx]).size == 1

        result = {
            "meta": {
                "source": "causal_score_normalization_oof.npz: split6 fusion, causal s10/s5 + median/MAD",
                "primary_smooth": 10,
                "robustness_smooth": 5,
                "observation": "past-only per-subject median/MAD normalized logit score",
                "variants": ["HMM logit", "fixed 0.5 normalized observation + 0.5 HMM logit"],
                "blend_weight_hmm": BLEND_WEIGHT,
                "student_t_df": DF,
                "student_t_scale": "MAD / t.ppf(0.75, df=5)",
                "nested_protocol": "outer subject fold; HMM threshold selected on four inner subject-fold OOF predictions; final HMM refit on all outer-train subjects",
                "threshold_selection": "maximum inner-OOF source-subject macro bACC; 97 quantiles plus zero; ties nearest zero",
                "strict_subject_disjoint": True,
                "strict_causal_history": True,
                "test_labels_used_for_fit_or_selection": False,
                "bootstrap": "paired subject resampling, seed=0",
            },
            "results": {},
            "folds": {},
            "causality_checks": {},
        }
        saved = {"y": y, "groups": groups, "eval_mask": eval_mask, "fold_id": fold_id}

        for width in SMOOTHS:
            tag = f"s{width}"
            x = normalized[width]
            valid = eval_mask & np.isfinite(x)
            assert int(valid.sum()) == 9227
            result["results"][tag] = {}
            result["folds"][tag] = {}
            saved[f"{tag}__normalized"] = x
            direct_pred = np.zeros(n, dtype=np.int8)
            hmm_scores = {kind: np.full(n, np.nan, dtype=np.float64) for kind in KINDS}
            blend_scores = {kind: np.full(n, np.nan, dtype=np.float64) for kind in KINDS}
            hmm_preds = {kind: np.zeros(n, dtype=np.int8) for kind in KINDS}
            blend_preds = {kind: np.zeros(n, dtype=np.int8) for kind in KINDS}
            fold_summaries = {kind: [] for kind in KINDS}

            for fold in range(5):
                train = valid & (fold_id != fold)
                test = valid & (fold_id == fold)
                direct_threshold, direct_train_macro = select_threshold(y, groups, train, x)
                direct_pred[test] = (x[test] >= direct_threshold).astype(np.int8)

                for kind in KINDS:
                    inner_hmm = np.full(n, np.nan, dtype=np.float64)
                    for inner in range(5):
                        if inner == fold:
                            continue
                        inner_fit = valid & (fold_id != fold) & (fold_id != inner)
                        transition, initial, emissions, _ = fit_parameters(y, groups, inner_fit, x)
                        target_subjects = np.unique(groups[fold_id == inner])
                        values = filter_subjects(x, groups, target_subjects,
                                                 transition, initial, kind, emissions)
                        inner_hmm[fold_id == inner] = values[fold_id == inner]
                    assert np.all(np.isfinite(inner_hmm[train]))
                    inner_logit = logit(np.clip(inner_hmm, 1e-6, 1.0 - 1e-6))
                    inner_blend = ((1.0 - BLEND_WEIGHT) * x
                                   + BLEND_WEIGHT * inner_logit)
                    hmm_threshold, hmm_train_macro = select_threshold(
                        y, groups, train, inner_logit)
                    blend_threshold, blend_train_macro = select_threshold(
                        y, groups, train, inner_blend)

                    transition, initial, emissions, summary = fit_parameters(y, groups, train, x)
                    test_subjects = np.unique(groups[test])
                    posterior = filter_subjects(x, groups, test_subjects,
                                                transition, initial, kind, emissions)
                    assert np.all(np.isfinite(posterior[test]))
                    posterior_logit = logit(np.clip(posterior, 1e-6, 1.0 - 1e-6))
                    blend = (1.0 - BLEND_WEIGHT) * x + BLEND_WEIGHT * posterior_logit
                    hmm_scores[kind][test] = posterior[test]
                    blend_scores[kind][test] = blend[test]
                    hmm_preds[kind][test] = (posterior_logit[test] >= hmm_threshold).astype(np.int8)
                    blend_preds[kind][test] = (blend[test] >= blend_threshold).astype(np.int8)
                    summary.update({
                        "fold": fold,
                        "test_subjects": [str(v) for v in test_subjects],
                        "direct_threshold": direct_threshold,
                        "direct_train_macro_bACC": direct_train_macro,
                        "hmm_threshold_inner_oof": hmm_threshold,
                        "hmm_train_macro_bACC_inner_oof": hmm_train_macro,
                        "blend_threshold_inner_oof": blend_threshold,
                        "blend_train_macro_bACC_inner_oof": blend_train_macro,
                    })
                    fold_summaries[kind].append(summary)
                log(f"{tag} fold {fold + 1}/5 nested HMM complete")

            assert np.array_equal(direct_pred, direct_reference[width]), \
                "Direct baseline must exactly reproduce prior normalization experiment"
            direct_metrics = metric_block(y, groups, valid, smoothed[width], direct_pred)
            median_metrics = metric_block(y, groups, valid, smoothed[width], median_reference[width])
            result["results"][tag]["causal_median_baseline"] = median_metrics
            result["results"][tag]["normalized_direct"] = direct_metrics
            result["folds"][tag] = fold_summaries
            saved[f"{tag}__direct_pred"] = direct_pred

            for kind in KINDS:
                for variant, score, pred in (
                        ("hmm", hmm_scores[kind], hmm_preds[kind]),
                        ("blend", blend_scores[kind], blend_preds[kind])):
                    assert np.all(np.isfinite(score[valid]))
                    metrics = metric_block(y, groups, valid, score, pred)
                    bootstrap = paired_bootstrap(y, groups, valid, direct_pred, pred)
                    median_bootstrap = paired_bootstrap(
                        y, groups, valid, median_reference[width], pred)
                    result["results"][tag][f"{kind}_{variant}"] = {
                        "metrics": metrics,
                        "paired_bootstrap_vs_normalized_direct": bootstrap,
                        "paired_bootstrap_vs_causal_median": median_bootstrap,
                    }
                    saved[f"{tag}__{kind}_{variant}__score"] = score
                    saved[f"{tag}__{kind}_{variant}__pred"] = pred
                    log(f"{tag} {kind:9s} {variant:5s} "
                        f"macro={metrics['subject_macro_bACC']:.6f} "
                        f"micro={metrics['window_micro_bACC']:.6f} "
                        f"AUC={metrics['AUC']:.6f} delta={bootstrap['mean_delta']:+.6f} "
                        f"CI={bootstrap['ci95']}")

            first_fold = fold_summaries["gaussian"][0]
            subject = first_fold["test_subjects"][0]
            result["causality_checks"][tag] = causality_check(
                x, groups, subject,
                np.asarray(first_fold["transition_matrix"], dtype=np.float64),
                np.asarray(first_fold["initial_probability"], dtype=np.float64),
                "gaussian", first_fold["emissions"]["gaussian"])

        result["seconds"] = time.perf_counter() - START
        np.savez_compressed(OUT / "causal_normalized_hmm_oof.npz", **saved)
        (OUT / "causal_normalized_hmm_results.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"done seconds={result['seconds']:.2f}")
    finally:
        sys.stdout = tee.stream
        tee.close()


if __name__ == "__main__":
    main()
