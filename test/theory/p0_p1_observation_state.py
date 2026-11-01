"""Minimal falsifiable P0/P1 experiment for the feasible generative theory.

P0: strict current-window observation scores from the existing 6-second,
subject-disjoint broadband feature OOF cache.
P1: same observation plus a causal finite-memory two-state filter. The filter
uses only the current and previous K score observations of each test subject;
its transition/emission parameters are fitted on outer-training subjects.

No raw history, test labels, test-subject calibration, or position features are
used. This script deliberately reuses the audited OOF cache rather than
re-extracting BDF windows.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm, t as student_t
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import importlib.util
_spec = importlib.util.spec_from_file_location("broadband_shape_window6", Path(__file__).resolve().parent.parent / "broadband_shape_window6.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
fit_hgb, fit_logreg = _mod.fit_hgb, _mod.fit_logreg

OUT = Path(__file__).resolve().parent.parent
CACHE = OUT / "broadband_shape_window6_oof.npz"
RES = OUT / "theory_p0_p1_observation_state_results.json"
N_SPLITS = 5
SEED = 0
# Minimal registered run: K=8 Student-t was the stable candidate in the prior smoke results.
KS = (8,)
KINDS = ("student_t",)
START = time.perf_counter()


def log(msg):
    print(f"[{time.perf_counter() - START:7.1f}s] {msg}", flush=True)


def stationary(T):
    vals, vecs = np.linalg.eig(T.T)
    p = np.real(vecs[:, np.argmin(np.abs(vals - 1.0))])
    p *= np.sign(p.sum())
    return p / p.sum()


def fit_params(y, groups, tr, score):
    tr = tr & np.isfinite(score)
    train_groups = np.unique(groups[tr])
    counts = np.ones((2, 2), float)
    for s in train_groups:
        ix = np.flatnonzero((groups == s) & tr & np.isfinite(score))
        if len(ix) > 1:
            np.add.at(counts, (y[ix[:-1]], y[ix[1:]]), 1.)
    T = counts / counts.sum(1, keepdims=True)
    pi = stationary(T)
    z = np.clip(score[tr], 1e-5, 1 - 1e-5)
    z = np.log(z / (1 - z))
    em = {}
    for kind in KINDS:
        pars = []
        for c in (0, 1):
            v = z[y[tr] == c]
            if kind == "gaussian":
                pars.append((float(v.mean()), max(float(v.std()), 1e-4)))
            else:
                med = float(np.median(v)); mad = float(np.median(np.abs(v - med)))
                pars.append((med, max(mad / 0.67448975, 1e-4)))
        em[kind] = pars
    return T, pi, em


def inner_oof_scores(X, y, groups, outer_train, fitter, name):
    scores = np.full(len(y), np.nan)
    Xtr, ytr, gtr = X[outer_train], y[outer_train], groups[outer_train]
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=0)
    for inner_fold, (itr, iva) in enumerate(splitter.split(Xtr, ytr, gtr), 1):
        pva, _, _ = fitter(Xtr[itr], ytr[itr], gtr[itr], Xtr[iva])
        scores[outer_train[iva]] = pva
        log(f"{name} inner fold {inner_fold}/4")
    return scores


def ll(kind, v, pars):
    if kind == "gaussian":
        return np.array([norm.logpdf(v, loc=m, scale=s) for m, s in pars])
    return np.array([student_t.logpdf(v, df=5, loc=m, scale=s) for m, s in pars])


def finite_filters(values, T, pi, kind, pars):
    """Return all registered finite-history filters in one causal pass."""
    outs = {K: np.full(len(values), np.nan) for K in KS}
    max_k = max(KS)
    history = []
    logT = np.log(np.maximum(T, 1e-300))
    logpi = np.log(np.maximum(pi, 1e-300))
    const = np.log(2.0 * np.pi)
    df = 5.0
    for i, v in enumerate(values):
        history.append(v)
        if len(history) > max_k:
            history.pop(0)
        emits = []
        for value in history:
            if not np.isfinite(value):
                emits.append(None)
                continue
            if kind == "gaussian":
                emits.append(np.array([-0.5 * (((value - m) / s) ** 2 + const) - np.log(s) for m, s in pars]))
            else:
                # Fixed-df Student-t log density; constants cancel but retain them for stability.
                emits.append(np.array([student_t.logpdf(value, df=df, loc=m, scale=s) for m, s in pars]))
        for K in KS:
            post = logpi.copy()
            for j, lp_emit in enumerate(emits[-K:]):
                if j > 0:
                    post = np.array([logsumexp(post + logT[:, state]) for state in range(2)])
                if lp_emit is not None:
                    post += lp_emit
                    post -= logsumexp(post)
            outs[K][i] = np.exp(post[1])
    return outs


def metrics(y, g, p):
    ok = np.isfinite(p)
    hard = p[ok] >= .5
    subs = []
    for s in np.unique(g[ok]):
        q = ok & (g == s)
        if np.unique(y[q]).size == 2:
            subs.append(balanced_accuracy_score(y[q], p[q] >= .5))
    return {
        "auc": float(roc_auc_score(y[ok], p[ok])),
        "bacc": float(balanced_accuracy_score(y[ok], hard)),
        "subject_macro_bacc": float(np.mean(subs)),
        "n_subjects": len(subs),
        "n_eval": int(ok.sum()),
    }


def main():
    d = np.load(CACHE, allow_pickle=False)
    y = d["y"].astype(int); groups = d["groups"]; fold_id = d["fold_id"].astype(int)
    assert np.array_equal(y, d["y"]) and len(np.unique(fold_id)) == N_SPLITS
    # Pre-registered minimal comparison: one linear and one nonlinear observation model.
    names = [k for k in ("logreg__all_new", "hgb__all_new") if k in d.files]
    feat = np.load(OUT / "broadband_shape_features.npz", allow_pickle=False)
    X = feat["X"]
    fitters = {"logreg__all_new": fit_logreg, "hgb__all_new": fit_hgb}
    result = {"protocol": {"P0": "outer-test stored current 6-second OOF base score", "P1": "causal finite-memory two-state filter", "K": list(KS), "K_interpretation": "历史8窗/约48秒，不是strict6s", "emissions": list(KINDS), "outer": "same stored StratifiedGroupKFold(5), subject-disjoint", "inner_oof": "outer-train subjects only, StratifiedGroupKFold(4, shuffle=True, random_state=0), per base model", "test_history": False, "labels_used_for_test": False}, "variants": {}}
    RES.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for name in names:
        base = d[name].astype(float)
        p0 = np.full(len(y), np.nan); p1 = {(kind, K): np.full(len(y), np.nan) for kind in KINDS for K in KS}
        for fold in range(N_SPLITS):
            fold_value = fold + 1 if np.min(fold_id) == 1 else fold
            tr = fold_id != fold_value; te = fold_id == fold_value
            inner = inner_oof_scores(X, y, groups, np.flatnonzero(tr), fitters[name], name)
            T, pi, em = fit_params(y, groups, tr, inner)
            for s in np.unique(groups[te]):
                ix = np.flatnonzero((groups == s) & te)
                p0[ix] = base[ix]
                vals = np.log(np.clip(base[ix], 1e-5, 1 - 1e-5) / np.clip(1 - base[ix], 1e-5, 1 - 1e-5))
                for kind in KINDS:
                    filtered = finite_filters(vals, T, pi, kind, em[kind])
                    for K in KS:
                        p1[kind, K][ix] = filtered[K]
            log(f"{name} fold {fold + 1}/{N_SPLITS} test_subjects={int(np.unique(groups[te]).size)}")
        result["variants"][name] = {"P0": metrics(y, groups, p0), "P1": {f"{kind}_K{K}": metrics(y, groups, p1[kind, K]) for kind in KINDS for K in KS}}
        # Persist after every observation model so an interrupted run remains auditable.
        result["elapsed_seconds"] = time.perf_counter() - START
        RES.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"{name} P0={result['variants'][name]['P0']}")
        for key, val in result["variants"][name]["P1"].items():
            log(f"{name} {key}={val}")
    result["elapsed_seconds"] = time.perf_counter() - START
    RES.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"COMPLETE {RES.name}")


if __name__ == "__main__":
    main()
