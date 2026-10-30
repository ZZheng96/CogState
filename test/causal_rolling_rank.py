"""Rolling causal empirical rank with warmup-matched and steady-state controls."""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from causal_empirical_rank import causal_empirical_rank
from causal_normalized_hmm import metric_block, paired_bootstrap

OUT = Path(__file__).resolve().parent
SMOOTHS = (10, 5)  # s10 primary; s5 preregistered robustness check.
HISTORY = 20
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


def causal_rolling_rank(values, groups, mask):
    """Mid-rank among the previous 20 evaluated scores; fixed negative warmup."""
    out = np.full(values.shape, np.nan, dtype=np.float64)
    for subject in np.unique(groups):
        idx = np.flatnonzero((groups == subject) & mask & np.isfinite(values))
        subject_values = values[idx]
        for t, i in enumerate(idx):
            if t < HISTORY:
                out[i] = 0.5
                continue
            history = subject_values[t - HISTORY:t]
            out[i] = (np.count_nonzero(history < subject_values[t])
                      + 0.5 * np.count_nonzero(history == subject_values[t])) / HISTORY
    return out


def evaluated_positions(groups, mask):
    position = np.full(len(groups), -1, dtype=np.int64)
    for subject in np.unique(groups):
        idx = np.flatnonzero((groups == subject) & mask)
        position[idx] = np.arange(idx.size)
    return position


def transition_error_block(y, groups, mask, pred):
    """Post-hoc error description only; labels never affect predictions."""
    distance = np.full(len(y), -1, dtype=np.int64)
    for subject in np.unique(groups):
        idx = np.flatnonzero((groups == subject) & mask)
        labels = y[idx]
        switches = np.flatnonzero(labels[1:] != labels[:-1]) + 1
        for t, i in enumerate(idx):
            distance[i] = min(np.abs(switches - t), default=idx.size)
    blocks = {}
    for name, selected in (("near_switch_le_5", mask & (distance <= 5)),
                           ("away_from_switch_gt_5", mask & (distance > 5))):
        blocks[name] = {
            "n": int(selected.sum()),
            "n_errors": int(np.count_nonzero(pred[selected] != y[selected])),
            "error_rate": float(np.mean(pred[selected] != y[selected])),
        }
    return blocks


def causality_checks(values, groups, mask, rank):
    checks = []
    for subject in np.unique(groups):
        idx = np.flatnonzero((groups == subject) & mask & np.isfinite(values))
        subject_values = values[idx]
        for position in (10, 30, 60):
            cut = min(position, idx.size - 1)
            perturbed = subject_values.copy()
            perturbed[cut:] += 100.0
            after = np.empty(idx.size, dtype=np.float64)
            for t in range(idx.size):
                if t < HISTORY:
                    after[t] = 0.5
                else:
                    history = perturbed[t - HISTORY:t]
                    after[t] = (np.count_nonzero(history < perturbed[t])
                                + 0.5 * np.count_nonzero(history == perturbed[t])) / HISTORY
            assert np.array_equal(rank[idx][:cut], after[:cut])
            checks.append({
                "subject": str(subject),
                "cut_within_evaluated_subject": cut,
                "past_outputs_bitwise_identical": True,
            })
    return checks


def comparison(y, groups, mask, baseline_pred, variant_pred):
    return paired_bootstrap(y, groups, mask, baseline_pred, variant_pred)


def main():
    tee = Tee(sys.stdout, OUT / "causal_rolling_rank.log")
    sys.stdout = tee
    try:
        with np.load(OUT / "causal_score_normalization_oof.npz", allow_pickle=False) as data:
            y = data["y"].astype(np.int64)
            groups = data["groups"]
            eval_mask = data["eval_mask"].astype(bool)
            raw = {width: data[f"s{width}__score"].astype(np.float64)
                   for width in SMOOTHS}
            direct_score = {width: data[f"s{width}__mad__normalized"].astype(np.float64)
                            for width in SMOOTHS}
            direct_pred = {width: data[f"s{width}__mad__crossfit_pred"].astype(np.int8)
                           for width in SMOOTHS}
        with np.load(OUT / "causal_empirical_rank_oof.npz", allow_pickle=False) as data:
            assert np.array_equal(y, data["y"])
            assert np.array_equal(groups, data["groups"])
            assert np.array_equal(eval_mask, data["eval_mask"].astype(bool))
            fold_id = data["fold_id"].astype(np.int64)
            saved_expanding_rank = {width: data[f"s{width}__rank"].astype(np.float64)
                                    for width in SMOOTHS}

        n = len(y)
        assert n == 16067 and set(np.unique(y)) == {0, 1}
        splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(y, y, groups))
        reconstructed_fold = np.full(n, -1, dtype=np.int64)
        for fold, (train_idx, test_idx) in enumerate(splits):
            assert not set(groups[train_idx]).intersection(groups[test_idx])
            reconstructed_fold[test_idx] = fold
        assert np.array_equal(fold_id, reconstructed_fold)
        position = evaluated_positions(groups, eval_mask)
        valid = eval_mask & (position >= 0)
        steady = valid & (position >= HISTORY)
        warmup = valid & (position < HISTORY)
        assert int(valid.sum()) == 9227 and int(warmup.sum()) == 1121

        result = {
            "meta": {
                "source": "split6 fusion; causal s10/s5 smoothed score",
                "primary_smooth": 10,
                "robustness_smooth": 5,
                "new_method": "current score mid-rank among the immediately preceding 20 evaluated scores of the same subject",
                "history": HISTORY,
                "decision": "fixed rank>0.5; exact ties and the first 20 evaluated positions are negative",
                "controls": "same OOF expanding rank; direct crossfit decision with identical negative warmup; steady-state metrics exclude the first 20 evaluated positions for every method",
                "transition_strata": "post-hoc near switch <=5 versus >5 evaluated positions; descriptive only",
                "selection": "single fixed rolling extension; no history-length, threshold, or test-OOF model selection",
                "strict_subject_disjoint": True,
                "strict_causal_history": True,
                "test_labels_used_for_fit_or_selection": False,
            },
            "cohort": {
                "n_valid": int(valid.sum()),
                "n_warmup": int(warmup.sum()),
                "n_steady": int(steady.sum()),
                "warmup_class_counts": {str(k): int(v) for k, v in
                                        zip(*np.unique(y[warmup], return_counts=True))},
            },
            "results": {},
            "causality_checks": {},
        }
        saved = {"y": y, "groups": groups, "eval_mask": eval_mask,
                 "fold_id": fold_id, "evaluated_position": position}

        for width in SMOOTHS:
            tag = f"s{width}"
            expanding = causal_empirical_rank(raw[width], groups, eval_mask)
            rolling = causal_rolling_rank(raw[width], groups, eval_mask)
            assert np.array_equal(expanding, saved_expanding_rank[width], equal_nan=True)
            expanding_pred = np.zeros(n, dtype=np.int8)
            rolling_pred = np.zeros(n, dtype=np.int8)
            expanding_pred[valid] = (expanding[valid] > 0.5).astype(np.int8)
            rolling_pred[valid] = (rolling[valid] > 0.5).astype(np.int8)
            direct_matched = direct_pred[width].copy()
            direct_matched[warmup] = 0

            methods = {
                "direct_crossfit": (direct_score[width], direct_pred[width]),
                "direct_crossfit_warmup_matched": (direct_score[width], direct_matched),
                "expanding_rank": (expanding, expanding_pred),
                "rolling20_rank": (rolling, rolling_pred),
            }
            result["results"][tag] = {}
            for name, (score, pred) in methods.items():
                result["results"][tag][name] = {
                    "overall": metric_block(y, groups, valid, score, pred),
                    "steady": metric_block(y, groups, steady, score, pred),
                    "transition_errors_overall": transition_error_block(
                        y, groups, valid, pred),
                }
                log(f"{tag} {name:34s} overall={result['results'][tag][name]['overall']['subject_macro_bACC']:.6f} "
                    f"steady={result['results'][tag][name]['steady']['subject_macro_bACC']:.6f}")

            result["results"][tag]["paired_comparisons"] = {
                "expanding_minus_original_direct_overall": comparison(
                    y, groups, valid, direct_pred[width], expanding_pred),
                "expanding_minus_warmup_matched_direct_overall": comparison(
                    y, groups, valid, direct_matched, expanding_pred),
                "expanding_minus_direct_steady": comparison(
                    y, groups, steady, direct_pred[width], expanding_pred),
                "rolling_minus_expanding_overall": comparison(
                    y, groups, valid, expanding_pred, rolling_pred),
                "rolling_minus_expanding_steady": comparison(
                    y, groups, steady, expanding_pred, rolling_pred),
                "rolling_minus_warmup_matched_direct_overall": comparison(
                    y, groups, valid, direct_matched, rolling_pred),
                "rolling_minus_direct_steady": comparison(
                    y, groups, steady, direct_pred[width], rolling_pred),
            }
            for name, row in result["results"][tag]["paired_comparisons"].items():
                log(f"{tag} {name:48s} delta={row['mean_delta']:+.6f} CI={row['ci95']}")

            result["causality_checks"][tag] = causality_checks(
                raw[width], groups, eval_mask, rolling)
            saved[f"{tag}__rolling_rank"] = rolling
            saved[f"{tag}__rolling_pred"] = rolling_pred
            saved[f"{tag}__expanding_pred"] = expanding_pred
            saved[f"{tag}__direct_warmup_matched_pred"] = direct_matched

        result["seconds"] = time.perf_counter() - START
        np.savez_compressed(OUT / "causal_rolling_rank_oof.npz", **saved)
        (OUT / "causal_rolling_rank_results.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"done seconds={result['seconds']:.2f}")
    finally:
        sys.stdout = tee.stream
        tee.close()


if __name__ == "__main__":
    main()
