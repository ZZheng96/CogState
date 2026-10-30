"""无测试标签的因果阈值实验：历史统计、滞后双阈值与训练折预注册阈值。"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

OUT = Path(__file__).resolve().parent
K0 = 120
MIN_HISTORY = 20
SMOOTHS = (1, 5, 10, 20)
QUANTILES = (0.40, 0.45, 0.50, 0.55, 0.60)
TRIM = 0.10
SEED = 0


def log(msg, start):
    print(f"[{time.perf_counter() - start:7.1f}s] {msg}", flush=True)


def causal_threshold(x, kind):
    """Threshold at t uses only x[:t]; short history falls back to .5."""
    out = np.full(len(x), 0.5, dtype=float)
    for t in range(len(x)):
        h = x[:t]
        h = h[np.isfinite(h)]
        if len(h) < MIN_HISTORY:
            continue
        if kind == "median":
            out[t] = np.median(h)
        elif kind == "trimmed_mean":
            hs = np.sort(h)
            n = int(len(hs) * TRIM)
            out[t] = np.mean(hs[n:len(hs) - n] if len(hs) > 2 * n else hs)
        elif kind.startswith("q"):
            out[t] = np.quantile(h, float(kind[1:]))
        else:
            raise ValueError(kind)
    return out


def causal_hysteresis(x, enter_q, exit_q):
    """Causal state machine; enter/exit quantiles are computed from prior scores."""
    out = np.zeros(len(x), dtype=bool)
    state = False
    for t in range(len(x)):
        h = x[:t]
        h = h[np.isfinite(h)]
        if len(h) >= MIN_HISTORY and np.isfinite(x[t]):
            enter = np.quantile(h, enter_q)
            leave = np.quantile(h, exit_q)
            if not state and x[t] >= enter:
                state = True
            elif state and x[t] < leave:
                state = False
        out[t] = state
    return out


def causal_smooth(v, groups, k):
    out = np.full(len(v), np.nan)
    for s in np.unique(groups):
        ii = np.where(groups == s)[0]
        x = v[ii]
        for t, i in enumerate(ii):
            h = x[max(0, t - k + 1):t + 1]
            h = h[np.isfinite(h)]
            if len(h):
                out[i] = h.mean()
    return out


def threshold_predictions(score, groups, strategy):
    pred = np.zeros(len(score), dtype=bool)
    for s in np.unique(groups):
        ii = np.where(groups == s)[0]
        x = score[ii]
        if strategy.startswith("hysteresis"):
            enter_q, exit_q = (0.60, 0.40) if strategy == "hysteresis_q60_q40" else (0.55, 0.45)
            pred[ii] = causal_hysteresis(x, enter_q, exit_q)
        else:
            pred[ii] = x >= causal_threshold(x, strategy)
    return pred


def train_fold_thresholds(score, y, groups, mask, splits):
    """Cross-fitted thresholds: each test fold sees labels/predictions from other folds only."""
    fold_id = np.full(len(y), -1, dtype=int)
    for f, (_, te) in enumerate(splits):
        fold_id[te] = f
    out = {"train_oof_median": np.full(len(y), .5), "train_oof_youden": np.full(len(y), .5)}
    for f in range(len(splits)):
        train = mask & (fold_id != f) & np.isfinite(score)
        test = mask & (fold_id == f) & np.isfinite(score)
        if not np.any(test) or not np.any(train):
            continue
        vals = score[train]
        out["train_oof_median"][test] = np.median(vals)
        candidates = np.unique(np.quantile(vals, np.linspace(.02, .98, 97)))
        best = max(candidates, key=lambda t: balanced_accuracy_score(y[train], vals >= t))
        out["train_oof_youden"][test] = best
    return out


def metrics(score, pred, y, groups, mask):
    m = mask & np.isfinite(score)
    micro = float(balanced_accuracy_score(y[m], pred[m]))
    per_subject = []
    for s in np.unique(groups[m]):
        q = m & (groups == s)
        if np.unique(y[q]).size >= 2:
            per_subject.append(balanced_accuracy_score(y[q], pred[q]))
    macro = float(np.mean(per_subject)) if per_subject else float("nan")
    return {"subject_macro_bACC": macro, "window_micro_bACC": micro,
            "AUC": float(roc_auc_score(y[m], score[m]))}


def main():
    start = time.perf_counter()
    old_path = OUT / "history_leakfree_oof.npz"
    new_path = OUT / "history_spectral_random_oof.npz"
    old = np.load(old_path)
    new = np.load(new_path)
    y = old["y"].astype(int)
    groups = old["groups"]
    mask = old["eval_mask"].astype(bool)
    assert np.array_equal(y, new["y"]) and np.array_equal(groups, new["groups"])
    assert np.array_equal(mask, new["eval_mask"])
    sources = {
        "old_best_fusion": old["D_hgb+single6_memz"],
        "new_spectral_fusion": new["D_hgb+old_best_fusion_score"],
        "new_spectral_D_hgb": new["D_hgb"],
    }
    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(y, y, groups))
    strategies = ["fixed_0.5", "median", "trimmed_mean"] + [f"q{q:.2f}" for q in QUANTILES]
    strategies += ["hysteresis_q60_q40", "hysteresis_q55_q45", "train_oof_median", "train_oof_youden"]
    result = {"meta": {"warmup": K0, "min_history": MIN_HISTORY, "trim_fraction": TRIM,
                         "quantiles": QUANTILES, "smooths": SMOOTHS,
                         "fixed_threshold": 0.5,
                         "hysteresis": "enter/exit thresholds are prior-score quantiles; no labels",
                         "train_thresholds": "per test fold from other folds' OOF predictions and labels",
                         "sources": list(sources)}, "sources": {}}
    saved = {"y": y, "groups": groups, "eval_mask": mask}
    for name, raw in sources.items():
        result["sources"][name] = {}
        log(f"source={name} start", start)
        for k in SMOOTHS:
            score = causal_smooth(raw, groups, k)
            thresholds = train_fold_thresholds(score, y, groups, mask, splits)
            block = {}
            for strategy in strategies:
                if strategy == "fixed_0.5":
                    pred = score >= .5
                elif strategy.startswith("train_oof"):
                    pred = score >= thresholds[strategy]
                else:
                    pred = threshold_predictions(score, groups, strategy)
                block[strategy] = metrics(score, pred, y, groups, mask)
                saved[f"{name}__s{k}__{strategy}"] = pred.astype(np.uint8)
            result["sources"][name][f"s{k}"] = block
            best = max(block, key=lambda z: block[z]["subject_macro_bACC"])
            log(f"source={name} s{k} best_macro={best}={block[best]['subject_macro_bACC']:.4f} "
                f"fixed={block['fixed_0.5']['subject_macro_bACC']:.4f}", start)
        saved[f"{name}__raw"] = raw
    result["seconds"] = time.perf_counter() - start
    np.savez_compressed(OUT / "causal_bacc_thresholds_oof.npz", **saved)
    (OUT / "causal_bacc_thresholds_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"done seconds={result['seconds']:.1f}", start)


if __name__ == "__main__":
    main()
