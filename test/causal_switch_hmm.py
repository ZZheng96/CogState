"""Nested causal Student-t HMM with innovation-triggered prior resets."""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logit, logsumexp
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold

from causal_normalized_hmm import (
    BLEND_WEIGHT,
    emission_logpdf,
    fit_parameters,
    metric_block,
    paired_bootstrap,
    select_threshold,
)

OUT = Path(__file__).resolve().parent
SMOOTHS = (10, 5)  # s10 primary; s5 preregistered robustness check.
GATE_QUANTILES = (None, 0.90, 0.75, 0.50)
SEED = 0
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


def innovation_threshold(x, groups, mask, quantile):
    if quantile is None:
        return float("inf")
    innovations = []
    for subject in np.unique(groups[mask]):
        values = x[(groups == subject) & mask]
        values = values[np.isfinite(values)]
        if values.size >= 2:
            innovations.append(np.abs(np.diff(values)))
    innovations = np.concatenate(innovations)
    assert np.all(np.isfinite(innovations))
    return float(np.quantile(innovations, quantile))


def online_filter(values, transition, initial, emissions, gate_threshold):
    """Use a stationary prior instead of the sticky prior after a large innovation."""
    output = np.full(values.size, np.nan, dtype=np.float64)
    posterior = None
    previous_value = np.nan
    n_reset = 0
    for t, value in enumerate(values):
        if posterior is None:
            if not np.isfinite(value):
                continue
            predicted = initial.copy()
        else:
            reset = (np.isfinite(value) and np.isfinite(previous_value)
                     and abs(value - previous_value) > gate_threshold)
            if reset:
                predicted = initial.copy()
                n_reset += 1
            else:
                predicted = posterior @ transition
                predicted /= predicted.sum()
        if np.isfinite(value):
            log_posterior = (np.log(np.maximum(predicted, 1e-300))
                             + emission_logpdf("student_t", value, emissions))
            log_posterior -= logsumexp(log_posterior)
            posterior = np.exp(log_posterior)
            previous_value = value
        else:
            posterior = predicted
        assert np.all(np.isfinite(posterior)) and np.isclose(posterior.sum(), 1.0)
        output[t] = posterior[1]
    return output, n_reset


def filter_subjects(x, groups, subjects, transition, initial, emissions, gate_threshold):
    output = np.full(x.shape, np.nan, dtype=np.float64)
    n_reset = 0
    for subject in subjects:
        idx = np.flatnonzero(groups == subject)
        output[idx], count = online_filter(
            x[idx], transition, initial, emissions, gate_threshold)
        n_reset += count
    return output, n_reset


def transition_diagnostics(y, groups, mask, pred):
    distance = np.full(y.size, 999, dtype=np.int64)
    for subject in np.unique(groups[mask]):
        idx = np.flatnonzero((groups == subject) & mask)
        labels = y[idx]
        changes = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
        last_change = np.maximum.accumulate(np.where(
            np.isin(np.arange(idx.size), changes), np.arange(idx.size), 0))
        distance[idx] = np.arange(idx.size) - last_change
    result = {}
    for name, selected in (("first_10_after_true_switch", mask & (distance < 10)),
                           ("stable_at_least_10", mask & (distance >= 10))):
        result[name] = {
            "n": int(selected.sum()),
            "window_micro_bACC": float(balanced_accuracy_score(y[selected], pred[selected])),
            "relax_recall": float(recall_score(y[selected], pred[selected], pos_label=0)),
            "task_recall": float(recall_score(y[selected], pred[selected], pos_label=1)),
        }
    return result


def causality_check(x, groups, subject, transition, initial, emissions, gate_threshold):
    idx = np.flatnonzero(groups == subject)
    values = x[idx]
    finite = np.flatnonzero(np.isfinite(values))
    cut = int(finite[min(30, finite.size - 1)])
    perturbed = values.copy()
    perturbed[cut:] += 100.0
    before, _ = online_filter(values, transition, initial, emissions, gate_threshold)
    after, _ = online_filter(perturbed, transition, initial, emissions, gate_threshold)
    assert np.array_equal(before[:cut], after[:cut], equal_nan=True)
    return {"subject": str(subject), "cut_within_subject": cut,
            "past_outputs_bitwise_identical": True}


def quantile_name(quantile):
    return "no_reset" if quantile is None else f"q{quantile:.2f}"


def main():
    tee = Tee(sys.stdout, OUT / "causal_switch_hmm.log")
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
        with np.load(OUT / "causal_normalized_hmm_oof.npz", allow_pickle=False) as data:
            fixed_scores = {width: data[f"s{width}__student_t_blend__score"].astype(np.float64)
                            for width in SMOOTHS}
            fixed_preds = {width: data[f"s{width}__student_t_blend__pred"].astype(np.int8)
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
                "source": "split6 fusion; causal s10/s5; past-only median/MAD normalized logit",
                "primary_smooth": 10,
                "robustness_smooth": 5,
                "emission": "Student-t df=5; same robust fit as causal_normalized_hmm",
                "blend": "fixed 0.5 normalized observation + 0.5 HMM logit",
                "new_mechanism": "large absolute current-minus-previous normalized observation replaces predicted prior with stationary initial distribution for that time step",
                "gate_candidates_inner_only": [quantile_name(q) for q in GATE_QUANTILES],
                "nested_protocol": "outer subject fold; gate and threshold selected on four inner subject-fold OOF predictions; final refit on all outer-train subjects",
                "selection": "maximum inner-OOF source-subject macro bACC; ties prefer less resetting, then threshold nearest zero",
                "strict_subject_disjoint": True,
                "strict_causal_history": True,
                "test_labels_used_for_fit_or_selection": False,
                "transition_diagnostics": "post-hoc only; never used for fitting or selection",
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
            direct_pred = np.zeros(n, dtype=np.int8)
            adaptive_score = np.full(n, np.nan, dtype=np.float64)
            adaptive_pred = np.zeros(n, dtype=np.int8)
            fold_summaries = []

            for fold in range(5):
                train = valid & (fold_id != fold)
                test = valid & (fold_id == fold)
                direct_threshold, direct_train_macro = select_threshold(y, groups, train, x)
                direct_pred[test] = (x[test] >= direct_threshold).astype(np.int8)

                inner_posteriors = {q: np.full(n, np.nan, dtype=np.float64)
                                    for q in GATE_QUANTILES}
                inner_gates = {q: {} for q in GATE_QUANTILES}
                for inner in range(5):
                    if inner == fold:
                        continue
                    inner_fit = valid & (fold_id != fold) & (fold_id != inner)
                    transition, initial, emissions, _ = fit_parameters(
                        y, groups, inner_fit, x)
                    target_subjects = np.unique(groups[fold_id == inner])
                    for quantile in GATE_QUANTILES:
                        gate = innovation_threshold(x, groups, inner_fit, quantile)
                        posterior, _ = filter_subjects(
                            x, groups, target_subjects, transition, initial,
                            emissions["student_t"], gate)
                        inner_posteriors[quantile][fold_id == inner] = posterior[fold_id == inner]
                        inner_gates[quantile][str(inner)] = gate if np.isfinite(gate) else None

                candidates = []
                for quantile in GATE_QUANTILES:
                    inner_hmm = inner_posteriors[quantile]
                    assert np.all(np.isfinite(inner_hmm[train]))
                    inner_blend = ((1.0 - BLEND_WEIGHT) * x
                                   + BLEND_WEIGHT * logit(np.clip(inner_hmm, 1e-6, 1.0 - 1e-6)))
                    threshold, train_macro = select_threshold(y, groups, train, inner_blend)
                    candidates.append({
                        "quantile": quantile,
                        "name": quantile_name(quantile),
                        "inner_oof_macro_bACC": train_macro,
                        "threshold": threshold,
                        "inner_gate_thresholds": inner_gates[quantile],
                    })
                selected = max(candidates, key=lambda row: (
                    row["inner_oof_macro_bACC"],
                    float("inf") if row["quantile"] is None else row["quantile"]))

                transition, initial, emissions, fit_summary = fit_parameters(y, groups, train, x)
                quantile = selected["quantile"]
                gate = innovation_threshold(x, groups, train, quantile)
                test_subjects = np.unique(groups[test])
                posterior, n_reset = filter_subjects(
                    x, groups, test_subjects, transition, initial,
                    emissions["student_t"], gate)
                posterior_logit = logit(np.clip(posterior, 1e-6, 1.0 - 1e-6))
                blend = (1.0 - BLEND_WEIGHT) * x + BLEND_WEIGHT * posterior_logit
                adaptive_score[test] = blend[test]
                adaptive_pred[test] = (posterior_logit[test] * BLEND_WEIGHT
                                       + x[test] * (1.0 - BLEND_WEIGHT)
                                       >= selected["threshold"]).astype(np.int8)
                fold_summaries.append({
                    "fold": fold,
                    "fit_subjects": fit_summary["fit_subjects"],
                    "test_subjects": [str(v) for v in test_subjects],
                    "transition_matrix": np.asarray(transition).tolist(),
                    "initial_probability": np.asarray(initial).tolist(),
                    "emissions": emissions["student_t"],
                    "direct_threshold": direct_threshold,
                    "direct_train_macro_bACC": direct_train_macro,
                    "selected_gate": selected["name"],
                    "selected_gate_threshold": gate if np.isfinite(gate) else None,
                    "selected_threshold_inner_oof": selected["threshold"],
                    "selected_train_macro_bACC_inner_oof": selected["inner_oof_macro_bACC"],
                    "n_test_reset": int(n_reset),
                    "n_test": int(test.sum()),
                    "inner_candidates": candidates,
                })
                log(f"{tag} fold {fold + 1}/5 gate={selected['name']} "
                    f"inner={selected['inner_oof_macro_bACC']:.6f} resets={n_reset}/{test.sum()}")

            assert np.array_equal(direct_pred, direct_reference[width])
            assert np.all(np.isfinite(adaptive_score[valid]))
            direct_metrics = metric_block(y, groups, valid, smoothed[width], direct_pred)
            fixed_metrics = metric_block(y, groups, valid, fixed_scores[width], fixed_preds[width])
            adaptive_metrics = metric_block(y, groups, valid, adaptive_score, adaptive_pred)
            result["results"][tag] = {
                "normalized_direct": direct_metrics,
                "fixed_student_t_blend": fixed_metrics,
                "adaptive_student_t_blend": {
                    "metrics": adaptive_metrics,
                    "paired_bootstrap_vs_normalized_direct": paired_bootstrap(
                        y, groups, valid, direct_pred, adaptive_pred),
                    "paired_bootstrap_vs_fixed_blend": paired_bootstrap(
                        y, groups, valid, fixed_preds[width], adaptive_pred),
                },
                "post_hoc_transition_diagnostics": {
                    "normalized_direct": transition_diagnostics(y, groups, valid, direct_pred),
                    "fixed_student_t_blend": transition_diagnostics(
                        y, groups, valid, fixed_preds[width]),
                    "adaptive_student_t_blend": transition_diagnostics(
                        y, groups, valid, adaptive_pred),
                },
            }
            result["folds"][tag] = fold_summaries
            saved[f"{tag}__adaptive_score"] = adaptive_score
            saved[f"{tag}__adaptive_pred"] = adaptive_pred
            saved[f"{tag}__direct_pred"] = direct_pred
            saved[f"{tag}__fixed_pred"] = fixed_preds[width]
            log(f"{tag} adaptive macro={adaptive_metrics['subject_macro_bACC']:.6f} "
                f"micro={adaptive_metrics['window_micro_bACC']:.6f} "
                f"AUC={adaptive_metrics['AUC']:.6f}")

            first_fold = fold_summaries[0]
            subject = first_fold["test_subjects"][0]
            result["causality_checks"][tag] = causality_check(
                x, groups, subject,
                np.asarray(first_fold["transition_matrix"], dtype=np.float64),
                np.asarray(first_fold["initial_probability"], dtype=np.float64),
                first_fold["emissions"],
                float("inf") if first_fold["selected_gate_threshold"] is None
                else first_fold["selected_gate_threshold"])

        result["seconds"] = time.perf_counter() - START
        np.savez_compressed(OUT / "causal_switch_hmm_oof.npz", **saved)
        (OUT / "causal_switch_hmm_results.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"done seconds={result['seconds']:.2f}")
    finally:
        sys.stdout = tee.stream
        tee.close()


if __name__ == "__main__":
    main()
