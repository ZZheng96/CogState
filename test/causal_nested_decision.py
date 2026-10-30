"""Nested bACC-first selection of direct scores and causal innovation-gated HMM blends."""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import logit
from sklearn.model_selection import StratifiedGroupKFold

from causal_normalized_hmm import fit_parameters, metric_block, paired_bootstrap, select_threshold
from causal_switch_hmm import filter_subjects, innovation_threshold, online_filter, quantile_name

OUT = Path(__file__).resolve().parent
SMOOTHS = (10, 5)  # s10 primary; s5 preregistered robustness check.
HMM_WEIGHTS = (0.25, 0.50, 0.75, 1.00)
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


def select_threshold_fast(y, groups, mask, score):
    """Exact equivalent of select_threshold, vectorized within each source subject."""
    candidates = np.unique(np.r_[0.0, np.quantile(score[mask], np.linspace(0.02, 0.98, 97))])
    values = np.zeros(candidates.size, dtype=np.float64)
    n_subjects = 0
    for subject in np.unique(groups[mask]):
        subject_mask = mask & (groups == subject)
        labels = y[subject_mask]
        if np.unique(labels).size != 2:
            continue
        pred = score[subject_mask, None] >= candidates[None, :]
        positive = labels == 1
        values += 0.5 * (pred[positive].mean(axis=0) + (~pred[~positive]).mean(axis=0))
        n_subjects += 1
    values /= n_subjects
    best_value, best_threshold = -np.inf, np.nan
    for threshold, value in zip(candidates, values):
        if value > best_value + 1e-12 or (
                abs(value - best_value) <= 1e-12
                and abs(threshold) < abs(best_threshold)):
            best_value, best_threshold = float(value), float(threshold)
    return best_threshold, best_value


def candidate_name(weight, quantile):
    if weight == 0.0:
        return "direct"
    return f"w{weight:.2f}_{quantile_name(quantile)}"


def selection_key(row):
    """Prefer direct/lower HMM weight and less resetting only for exact metric ties."""
    quantile = row["quantile"]
    return (
        row["inner_oof_macro_bACC"],
        -row["weight"],
        float("inf") if quantile is None else quantile,
        -abs(row["threshold"]),
    )


def causality_checks(x, groups, subject, transition, initial, emissions, gate):
    idx = np.flatnonzero(groups == subject)
    values = x[idx]
    finite = np.flatnonzero(np.isfinite(values))
    checks = []
    for position in (10, 30, 60):
        cut = int(finite[min(position, finite.size - 1)])
        perturbed = values.copy()
        perturbed[cut:] += 100.0
        before, _ = online_filter(values, transition, initial, emissions, gate)
        after, _ = online_filter(perturbed, transition, initial, emissions, gate)
        assert np.array_equal(before[:cut], after[:cut], equal_nan=True)
        checks.append({"subject": str(subject), "cut_within_subject": cut,
                       "past_outputs_bitwise_identical": True})
    return checks


def main():
    tee = Tee(sys.stdout, OUT / "causal_nested_decision.log")
    sys.stdout = tee
    try:
        with np.load(OUT / "causal_score_normalization_oof.npz", allow_pickle=False) as data:
            y = data["y"].astype(np.int64)
            groups = data["groups"]
            eval_mask = data["eval_mask"].astype(bool)
            normalized = {width: data[f"s{width}__mad__normalized"].astype(np.float64)
                          for width in SMOOTHS}
            direct_reference = {width: data[f"s{width}__mad__crossfit_pred"].astype(np.int8)
                                for width in SMOOTHS}
        with np.load(OUT / "causal_normalized_hmm_oof.npz", allow_pickle=False) as data:
            fixed_scores = {width: data[f"s{width}__student_t_blend__score"].astype(np.float64)
                            for width in SMOOTHS}
            fixed_preds = {width: data[f"s{width}__student_t_blend__pred"].astype(np.int8)
                           for width in SMOOTHS}
        with np.load(OUT / "causal_switch_hmm_oof.npz", allow_pickle=False) as data:
            adaptive_scores = {width: data[f"s{width}__adaptive_score"].astype(np.float64)
                               for width in SMOOTHS}
            adaptive_preds = {width: data[f"s{width}__adaptive_pred"].astype(np.int8)
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
                "candidates": ["direct"] + [candidate_name(w, q) for w in HMM_WEIGHTS
                                             for q in GATE_QUANTILES],
                "new_mechanism": "joint inner-OOF selection of direct fallback, HMM logit weight, and innovation reset gate",
                "nested_protocol": "outer subject fold; all candidates and thresholds selected on four inner subject-fold OOF predictions; final HMM refit on all outer-train subjects",
                "selection": "maximum inner-OOF source-subject macro bACC; exact ties prefer lower HMM weight, less resetting, then threshold nearest zero",
                "strict_subject_disjoint": True,
                "strict_causal_history": True,
                "test_labels_used_for_fit_or_selection": False,
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
            selected_score = np.full(n, np.nan, dtype=np.float64)
            selected_pred = np.zeros(n, dtype=np.int8)
            fold_summaries = []

            for fold in range(5):
                train = valid & (fold_id != fold)
                test = valid & (fold_id == fold)
                direct_threshold, direct_train_macro = select_threshold_fast(y, groups, train, x)
                direct_pred[test] = (x[test] >= direct_threshold).astype(np.int8)

                inner_posteriors = {q: np.full(n, np.nan, dtype=np.float64)
                                    for q in GATE_QUANTILES}
                inner_gates = {q: {} for q in GATE_QUANTILES}
                for inner in range(5):
                    if inner == fold:
                        continue
                    inner_fit = valid & (fold_id != fold) & (fold_id != inner)
                    transition, initial, emissions, _ = fit_parameters(y, groups, inner_fit, x)
                    target_subjects = np.unique(groups[fold_id == inner])
                    for quantile in GATE_QUANTILES:
                        gate = innovation_threshold(x, groups, inner_fit, quantile)
                        posterior, _ = filter_subjects(
                            x, groups, target_subjects, transition, initial,
                            emissions["student_t"], gate)
                        target = fold_id == inner
                        inner_posteriors[quantile][target] = posterior[target]
                        inner_gates[quantile][str(inner)] = gate if np.isfinite(gate) else None

                candidates = [{
                    "name": "direct",
                    "weight": 0.0,
                    "quantile": None,
                    "inner_oof_macro_bACC": direct_train_macro,
                    "threshold": direct_threshold,
                    "inner_gate_thresholds": {},
                }]
                for quantile in GATE_QUANTILES:
                    inner_hmm = inner_posteriors[quantile]
                    assert np.all(np.isfinite(inner_hmm[train]))
                    inner_logit = logit(np.clip(inner_hmm, 1e-6, 1.0 - 1e-6))
                    for weight in HMM_WEIGHTS:
                        inner_blend = (1.0 - weight) * x + weight * inner_logit
                        threshold, train_macro = select_threshold_fast(y, groups, train, inner_blend)
                        candidates.append({
                            "name": candidate_name(weight, quantile),
                            "weight": weight,
                            "quantile": quantile,
                            "inner_oof_macro_bACC": train_macro,
                            "threshold": threshold,
                            "inner_gate_thresholds": inner_gates[quantile],
                        })
                selected = max(candidates, key=selection_key)
                weight = selected["weight"]
                quantile = selected["quantile"]

                transition, initial, emissions, fit_summary = fit_parameters(y, groups, train, x)
                test_subjects = np.unique(groups[test])
                gate = innovation_threshold(x, groups, train, quantile)
                if weight == 0.0:
                    selected_score[test] = x[test]
                    n_reset = 0
                else:
                    posterior, n_reset = filter_subjects(
                        x, groups, test_subjects, transition, initial,
                        emissions["student_t"], gate)
                    posterior_logit = logit(np.clip(posterior, 1e-6, 1.0 - 1e-6))
                    selected_score[test] = ((1.0 - weight) * x[test]
                                            + weight * posterior_logit[test])
                selected_pred[test] = (selected_score[test] >= selected["threshold"]).astype(np.int8)
                fold_summaries.append({
                    "fold": fold,
                    "fit_subjects": fit_summary["fit_subjects"],
                    "test_subjects": [str(v) for v in test_subjects],
                    "transition_matrix": np.asarray(transition).tolist(),
                    "initial_probability": np.asarray(initial).tolist(),
                    "emissions": emissions["student_t"],
                    "direct_threshold": direct_threshold,
                    "direct_train_macro_bACC": direct_train_macro,
                    "selected_candidate": selected["name"],
                    "selected_weight": weight,
                    "selected_gate_threshold": gate if np.isfinite(gate) else None,
                    "selected_threshold_inner_oof": selected["threshold"],
                    "selected_train_macro_bACC_inner_oof": selected["inner_oof_macro_bACC"],
                    "n_test_reset": int(n_reset),
                    "n_test": int(test.sum()),
                    "inner_candidates": candidates,
                })
                log(f"{tag} fold {fold + 1}/5 selected={selected['name']} "
                    f"inner={selected['inner_oof_macro_bACC']:.6f} "
                    f"resets={n_reset}/{test.sum()}")

            assert np.array_equal(direct_pred, direct_reference[width])
            assert np.all(np.isfinite(selected_score[valid]))
            direct_metrics = metric_block(y, groups, valid, x, direct_pred)
            selected_metrics = metric_block(y, groups, valid, selected_score, selected_pred)
            result["results"][tag] = {
                "normalized_direct": direct_metrics,
                "fixed_student_t_blend": metric_block(
                    y, groups, valid, fixed_scores[width], fixed_preds[width]),
                "adaptive_fixed_weight_blend": metric_block(
                    y, groups, valid, adaptive_scores[width], adaptive_preds[width]),
                "nested_selected_decision": {
                    "metrics": selected_metrics,
                    "paired_bootstrap_vs_normalized_direct": paired_bootstrap(
                        y, groups, valid, direct_pred, selected_pred),
                    "selection_counts": dict(Counter(
                        row["selected_candidate"] for row in fold_summaries)),
                },
            }
            result["folds"][tag] = fold_summaries
            saved[f"{tag}__selected_score"] = selected_score
            saved[f"{tag}__selected_pred"] = selected_pred
            saved[f"{tag}__direct_pred"] = direct_pred
            log(f"{tag} selected macro={selected_metrics['subject_macro_bACC']:.6f} "
                f"micro={selected_metrics['window_micro_bACC']:.6f} "
                f"AUC={selected_metrics['AUC']:.6f}")

            result["causality_checks"][tag] = []
            for row in fold_summaries:
                subject = row["test_subjects"][0]
                result["causality_checks"][tag].extend(causality_checks(
                    x, groups, subject,
                    np.asarray(row["transition_matrix"], dtype=np.float64),
                    np.asarray(row["initial_probability"], dtype=np.float64),
                    row["emissions"],
                    float("inf") if row["selected_gate_threshold"] is None
                    else row["selected_gate_threshold"]))

        result["seconds"] = time.perf_counter() - START
        np.savez_compressed(OUT / "causal_nested_decision_oof.npz", **saved)
        (OUT / "causal_nested_decision_results.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"done seconds={result['seconds']:.2f}")
    finally:
        sys.stdout = tee.stream
        tee.close()


if __name__ == "__main__":
    main()
