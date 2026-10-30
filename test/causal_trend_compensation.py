"""固定多尺度趋势补偿及其暖机匹配、稳态对照。"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logit
from sklearn.model_selection import StratifiedGroupKFold

from causal_nested_decision import select_threshold_fast
from causal_rolling_rank import evaluated_positions, transition_error_block
from causal_score_normalization import (
    SCALE_FLOOR, causal_normalize, causal_smooth, metric_block, paired_bootstrap,
)

OUT = Path(__file__).resolve().parent
SEED = 0
MIN_HISTORY = 20
TREND_WEIGHT = 0.8
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


def causality_checks(values, groups, mask, normalized):
    """扰动未来补偿分数，逐位检查过去的归一化输出。"""
    checks = []
    for subject in np.unique(groups):
        idx = np.flatnonzero((groups == subject) & mask & np.isfinite(values))
        subject_values = values[idx]
        for position in (10, 30, 60):
            cut = min(position, idx.size - 1)
            perturbed = subject_values.copy()
            perturbed[cut:] += 100.0
            x = logit(np.clip(perturbed, 1e-6, 1.0 - 1e-6))
            after = np.empty(idx.size, dtype=np.float64)
            for t in range(idx.size):
                if t < MIN_HISTORY:
                    center, scale = 0.0, 1.0
                else:
                    history = x[:t]
                    center = float(np.median(history))
                    scale = max(1.4826 * float(np.median(np.abs(history - center))),
                                SCALE_FLOOR)
                after[t] = (x[t] - center) / scale
            assert np.array_equal(normalized[idx][:cut], after[:cut])
            checks.append({
                "subject": str(subject),
                "cut_within_evaluated_subject": cut,
                "past_outputs_bitwise_identical": True,
            })
    return checks


def main():
    tee = Tee(sys.stdout, OUT / "causal_trend_compensation.log")
    sys.stdout = tee
    try:
        with np.load(OUT / "causal_score_normalization_oof.npz", allow_pickle=False) as data:
            y = data["y"].astype(np.int64)
            groups = data["groups"]
            eval_mask = data["eval_mask"].astype(bool)
            raw = data["raw"].astype(np.float64)
            s5 = data["s5__score"].astype(np.float64)
            s10 = data["s10__score"].astype(np.float64)
            baseline_score = data["s10__mad__normalized"].astype(np.float64)
            baseline_pred = data["s10__mad__crossfit_pred"].astype(np.int8)

        n = len(y)
        assert n == 16067 and set(np.unique(y)) == {0, 1}
        assert np.array_equal(s5, causal_smooth(raw, groups, 5), equal_nan=True)
        assert np.array_equal(s10, causal_smooth(raw, groups, 10), equal_nan=True)
        splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(y, y, groups))
        fold_id = np.full(n, -1, dtype=np.int64)
        for fold, (train_idx, test_idx) in enumerate(splits):
            assert not set(groups[train_idx]).intersection(groups[test_idx])
            fold_id[test_idx] = fold
        assert np.all(fold_id >= 0)
        position = evaluated_positions(groups, eval_mask)
        valid = eval_mask & (position >= 0)
        steady = valid & (position >= MIN_HISTORY)
        warmup = valid & (position < MIN_HISTORY)
        assert int(valid.sum()) == 9227 and int(steady.sum()) == 8106

        # 对线性趋势，两个 trailing mean 的滞后差为 2.5 个步长，固定外推当前值。
        trend_score = (1.0 + TREND_WEIGHT) * s5 - TREND_WEIGHT * s10
        normalized = causal_normalize(trend_score, groups, eval_mask, "mad")
        assert np.array_equal(np.isfinite(normalized), valid)
        trend_pred = np.zeros(n, dtype=np.int8)
        fold_summaries = []
        for fold in range(5):
            train = valid & (fold_id != fold)
            test = valid & (fold_id == fold)
            threshold, train_macro = select_threshold_fast(y, groups, train, normalized)
            trend_pred[test] = (normalized[test] >= threshold).astype(np.int8)
            fold_summaries.append({
                "fold": fold,
                "threshold": threshold,
                "train_subject_macro_bACC": train_macro,
                "n_train": int(train.sum()),
                "n_test": int(test.sum()),
                "train_subjects": [str(v) for v in np.unique(groups[train])],
                "test_subjects": [str(v) for v in np.unique(groups[test])],
            })
            log(f"fold {fold + 1}/5 threshold={threshold:.6f} train_macro={train_macro:.6f}")

        baseline_matched = baseline_pred.copy()
        trend_matched = trend_pred.copy()
        baseline_matched[warmup] = 0
        trend_matched[warmup] = 0
        methods = {
            "s10_direct": (baseline_score, baseline_pred),
            "s10_direct_warmup_matched": (baseline_score, baseline_matched),
            "trend_compensated": (normalized, trend_pred),
            "trend_compensated_warmup_matched": (normalized, trend_matched),
        }
        result = {
            "meta": {
                "source": "split6 fusion; existing causal s5 and s10 smoothed probabilities",
                "formula": "q=1.8*s5-0.8*s10; logit(clip(q,1e-6,1-1e-6)); past-only expanding median/MAD",
                "derivation": "for a linear ramp, s5 and s10 lag the current value by 2 and 4.5 evaluated steps; 0.8=2/(4.5-2)",
                "selection": "single fixed extrapolation weight; no weight search or test-OOF model/threshold selection; original training-fold threshold grid unchanged",
                "decision": "same five-fold crossfit threshold protocol as s10 direct; primary overall comparison applies identical negative warmup to both methods",
                "steady": "exclude the first 20 evaluated positions for every method",
                "strict_subject_disjoint": True,
                "strict_causal_history": True,
                "test_labels_used_for_fit_or_selection": False,
            },
            "score_diagnostics": {
                "valid_min": float(np.min(trend_score[valid])),
                "valid_max": float(np.max(trend_score[valid])),
                "n_below_zero": int(np.count_nonzero(trend_score[valid] < 0.0)),
                "n_above_one": int(np.count_nonzero(trend_score[valid] > 1.0)),
                "clipped_fraction": float(np.mean((trend_score[valid] < 0.0)
                                                   | (trend_score[valid] > 1.0))),
            },
            "results": {},
            "folds": fold_summaries,
            "causality_checks": causality_checks(trend_score, groups, eval_mask, normalized),
        }
        for name, (score, pred) in methods.items():
            result["results"][name] = {
                "overall": metric_block(y, groups, valid, score, pred),
                "steady": metric_block(y, groups, steady, score, pred),
                "transition_errors_overall": transition_error_block(y, groups, valid, pred),
                "transition_errors_steady": transition_error_block(y, groups, steady, pred),
            }
            log(f"{name:38s} overall={result['results'][name]['overall']['subject_macro_bACC']:.6f} "
                f"steady={result['results'][name]['steady']['subject_macro_bACC']:.6f} "
                f"AUC={result['results'][name]['overall']['AUC']:.6f}")

        result["paired_comparisons"] = {
            "trend_minus_s10_original_overall": paired_bootstrap(
                y, groups, valid, baseline_pred, trend_pred),
            "trend_minus_s10_warmup_matched_overall": paired_bootstrap(
                y, groups, valid, baseline_matched, trend_matched),
            "trend_minus_s10_steady": paired_bootstrap(
                y, groups, steady, baseline_pred, trend_pred),
        }
        for name, row in result["paired_comparisons"].items():
            log(f"{name:44s} delta={row['mean_delta']:+.6f} CI={row['ci95']}")

        result["seconds"] = time.perf_counter() - START
        np.savez_compressed(
            OUT / "causal_trend_compensation_oof.npz",
            y=y, groups=groups, eval_mask=eval_mask, fold_id=fold_id,
            evaluated_position=position, trend_score=trend_score,
            normalized=normalized, trend_pred=trend_pred,
            trend_warmup_matched_pred=trend_matched,
            baseline_warmup_matched_pred=baseline_matched,
        )
        (OUT / "causal_trend_compensation_results.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"done seconds={result['seconds']:.2f}")
    finally:
        sys.stdout = tee.stream
        tee.close()


if __name__ == "__main__":
    main()
