"""Strictly causal empirical-rank calibration of the best split6 OOF score."""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from causal_nested_decision import select_threshold_fast
from causal_normalized_hmm import metric_block, paired_bootstrap

OUT = Path(__file__).resolve().parent
SMOOTHS = (10, 5)  # s10 primary; s5 preregistered robustness check.
MIN_HISTORY = 20
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


def causal_empirical_rank(values, groups, mask):
    """Mid-rank of the current score among earlier evaluated scores of that subject."""
    out = np.full(values.shape, np.nan, dtype=np.float64)
    for subject in np.unique(groups):
        idx = np.flatnonzero((groups == subject) & mask & np.isfinite(values))
        subject_values = values[idx]
        for t, i in enumerate(idx):
            if t < MIN_HISTORY:
                out[i] = 0.5
                continue
            history = subject_values[:t]
            out[i] = (np.count_nonzero(history < subject_values[t])
                      + 0.5 * np.count_nonzero(history == subject_values[t])) / t
    return out


def causality_checks(values, groups, mask, rank):
    checks = []
    for subject in np.unique(groups):
        idx = np.flatnonzero((groups == subject) & mask & np.isfinite(values))
        subject_values = values[idx]
        for position in (10, 30, 60):
            cut = min(position, idx.size - 1)
            perturbed = subject_values.copy()
            perturbed[cut:] += 100.0
            before = rank[idx]
            after = np.empty_like(before)
            for t in range(idx.size):
                if t < MIN_HISTORY:
                    after[t] = 0.5
                else:
                    history = perturbed[:t]
                    after[t] = (np.count_nonzero(history < perturbed[t])
                                + 0.5 * np.count_nonzero(history == perturbed[t])) / t
            assert np.array_equal(before[:cut], after[:cut])
            checks.append({
                "subject": str(subject),
                "cut_within_evaluated_subject": cut,
                "past_outputs_bitwise_identical": True,
            })
    return checks


def main():
    tee = Tee(sys.stdout, OUT / "causal_empirical_rank.log")
    sys.stdout = tee
    try:
        with np.load(OUT / "causal_score_normalization_oof.npz", allow_pickle=False) as data:
            y = data["y"].astype(np.int64)
            groups = data["groups"]
            eval_mask = data["eval_mask"].astype(bool)
            raw = {width: data[f"s{width}__score"].astype(np.float64)
                   for width in SMOOTHS}
            direct_reference = {
                width: data[f"s{width}__mad__crossfit_pred"].astype(np.int8)
                for width in SMOOTHS}
            direct_score = {
                width: data[f"s{width}__mad__normalized"].astype(np.float64)
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
                "source": "split6 fusion; causal s10/s5 smoothed score",
                "primary_smooth": 10,
                "robustness_smooth": 5,
                "calibration": "current score mid-rank among all earlier evaluated scores of the same subject",
                "min_history": MIN_HISTORY,
                "decision": "fixed rank>=0.5, fixed rank>0.5 (ties negative), and crossfit training-fold threshold; threshold maximizes source-subject macro bACC",
                "threshold_candidates": "97 training-score quantiles plus zero",
                "selection": "no score/calibration selection; s10 is primary and s5 is a robustness check",
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
            rank = causal_empirical_rank(raw[width], groups, eval_mask)
            valid = eval_mask & np.isfinite(rank)
            assert int(valid.sum()) == 9227
            assert np.all((rank[valid] >= 0.0) & (rank[valid] <= 1.0))
            direct_metrics = metric_block(
                y, groups, valid, direct_score[width], direct_reference[width])
            fixed_ge_pred = np.zeros(n, dtype=np.int8)
            fixed_ge_pred[valid] = (rank[valid] >= 0.5).astype(np.int8)
            fixed_gt_pred = np.zeros(n, dtype=np.int8)
            fixed_gt_pred[valid] = (rank[valid] > 0.5).astype(np.int8)
            crossfit_pred = np.zeros(n, dtype=np.int8)
            fold_summaries = []

            for fold in range(5):
                train = valid & (fold_id != fold)
                test = valid & (fold_id == fold)
                threshold, train_macro = select_threshold_fast(y, groups, train, rank)
                crossfit_pred[test] = (rank[test] >= threshold).astype(np.int8)
                fold_summaries.append({
                    "fold": fold,
                    "threshold": threshold,
                    "train_subject_macro_bACC": train_macro,
                    "n_train": int(train.sum()),
                    "n_test": int(test.sum()),
                    "train_subjects": [str(v) for v in np.unique(groups[train])],
                    "test_subjects": [str(v) for v in np.unique(groups[test])],
                })
                log(f"{tag} fold {fold + 1}/5 threshold={threshold:.6f} "
                    f"train_macro={train_macro:.6f}")

            fixed_ge_metrics = metric_block(y, groups, valid, rank, fixed_ge_pred)
            fixed_gt_metrics = metric_block(y, groups, valid, rank, fixed_gt_pred)
            crossfit_metrics = metric_block(y, groups, valid, rank, crossfit_pred)
            result["results"][tag] = {
                "normalized_direct": direct_metrics,
                "empirical_rank_fixed_ge_0.5": {
                    "metrics": fixed_ge_metrics,
                    "paired_bootstrap_vs_normalized_direct": paired_bootstrap(
                        y, groups, valid, direct_reference[width], fixed_ge_pred),
                },
                "empirical_rank_fixed_gt_0.5": {
                    "metrics": fixed_gt_metrics,
                    "paired_bootstrap_vs_normalized_direct": paired_bootstrap(
                        y, groups, valid, direct_reference[width], fixed_gt_pred),
                },
                "empirical_rank_crossfit_threshold": {
                    "metrics": crossfit_metrics,
                    "paired_bootstrap_vs_normalized_direct": paired_bootstrap(
                        y, groups, valid, direct_reference[width], crossfit_pred),
                },
            }
            result["folds"][tag] = fold_summaries
            result["causality_checks"][tag] = causality_checks(
                raw[width], groups, eval_mask, rank)
            saved[f"{tag}__rank"] = rank
            saved[f"{tag}__fixed_ge_pred"] = fixed_ge_pred
            saved[f"{tag}__fixed_gt_pred"] = fixed_gt_pred
            saved[f"{tag}__crossfit_pred"] = crossfit_pred
            log(f"{tag} fixed>=0.5 macro={fixed_ge_metrics['subject_macro_bACC']:.6f} "
                f"micro={fixed_ge_metrics['window_micro_bACC']:.6f}")
            log(f"{tag} fixed>0.5 macro={fixed_gt_metrics['subject_macro_bACC']:.6f} "
                f"micro={fixed_gt_metrics['window_micro_bACC']:.6f}")
            log(f"{tag} crossfit macro={crossfit_metrics['subject_macro_bACC']:.6f} "
                f"micro={crossfit_metrics['window_micro_bACC']:.6f} "
                f"AUC={crossfit_metrics['AUC']:.6f}")

        result["seconds"] = time.perf_counter() - START
        np.savez_compressed(OUT / "causal_empirical_rank_oof.npz", **saved)
        (OUT / "causal_empirical_rank_results.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"done seconds={result['seconds']:.2f}")
    finally:
        sys.stdout = tee.stream
        tee.close()


if __name__ == "__main__":
    main()
