"""pBCI theta negative-amplitude marker (PMC10790894 style) on our single channel.

Per window: theta band-pass (4-8 Hz, Butterworth-4 zero-phase) -> local minima below 0 (negative half-wave troughs)
-> f1 = log mean |trough|, f2 = log median |trough|, f3 = log(mean |trough| / std(1-45 Hz broadband)) (relative).
The paper uses a frontal channel (Fz/FCz); we have one channel only, so 'frontal' cannot be chosen -> applicability note.
Classifier: LDA (lsqr, auto shrinkage) on the 3 features (and on f1 alone); single-feature AUCs (sign fixed on the
training folds: pooled AUC orientation of train rows) are reported per protocol.
Protocols: strict (subject folds, train StandardScaler), lit_kfold, sessnorm (whole-session per-subject z-score),
           causal_norm (strict, per-subject z-score with past windows only, stream seeds 0,1,2; prior = train stats,
           pseudo-count 16).
Run: python theta_negamp.py [--smoke]
"""
import argparse, sys
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

P0 = 16.


def causal_norm_subject(Xs, mu0, sd0):
    """Row t uses rows < t only: running mean/var with prior (mu0, sd0, pseudo-count P0)."""
    out = np.zeros_like(Xs); s1 = np.zeros(Xs.shape[1]); s2 = np.zeros(Xs.shape[1])
    for t in range(len(Xs)):
        n = t + P0
        mu = (P0 * mu0 + s1) / n; var = (P0 * (sd0 ** 2 + mu0 ** 2) + s2) / n - mu ** 2
        out[t] = (Xs[t] - mu) / np.sqrt(np.maximum(var, 1e-12))
        s1 = s1 + Xs[t]; s2 = s2 + Xs[t] ** 2
    return out


def sessnorm(X, g):
    Xo = X.copy()
    for s in np.unique(g):
        q = g == s; Xo[q] = (X[q] - X[q].mean(0)) / (X[q].std(0) + 1e-9)
    return Xo


def lda_score(X, y, tr, te):
    sc = StandardScaler().fit(X[tr])
    m = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(sc.transform(X[tr]), y[tr])
    return C.sig(m.decision_function(sc.transform(X[te])))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); args = ap.parse_args()
    d, stem, tee = C.start_log('theta_negamp', args.smoke)
    D = C.setup(); R = C.raw_features(D); y, g, subs = D['y'], D['g'], D['subs']
    folds = [0] if args.smoke else list(range(5)); nb = 100 if args.smoke else 1000
    sseeds = (0,) if args.smoke else C.STREAM_SEEDS
    TH = R['theta']; base = C.strict_baseline(D, folds)
    ev = np.isin(D['fold'], folds); ev_lit = ev & np.isin(D['lit_fold'], folds)
    res = {'meta': {'folds': folds, 'stream_seeds': list(sseeds), 'smoke': args.smoke,
                    'features': ['log_mean_neg_trough', 'log_median_neg_trough', 'log_rel_neg_trough'],
                    'applicability': 'single channel, not frontal-selected; marker as defined on Fz/FCz in the paper'},
           'checks': {'strict_fold_equality': 'ok'}, 'methods': {}, 'single_feature_auc': {}}
    S = {}
    for cols, nm in (([0, 1, 2], 'lda3'), ([0], 'lda1')):
        for proto in ('strict', 'lit_kfold', 'sessnorm'):
            fk = D['lit_fold'] if proto == 'lit_kfold' else D['fold']
            X = sessnorm(TH, g) if proto == 'sessnorm' else TH
            p = np.full(len(y), np.nan)
            for k in folds:
                tr, te = np.flatnonzero(fk != k), np.flatnonzero(fk == k)
                p[te] = lda_score(X[:, cols], y, tr, te)
            S[f'{nm}__{proto}'] = p
        for sd in sseeds:
            order, _ = C.stream_order(g, subs, sd); p = np.full(len(y), np.nan)
            for k in folds:
                tr, te = np.flatnonzero(D['fold'] != k), np.flatnonzero(D['fold'] == k)
                X = np.zeros((len(y), len(cols)))
                mu0, sd0 = TH[tr][:, cols].mean(0), TH[tr][:, cols].std(0)
                for loc in order: X[loc] = causal_norm_subject(TH[loc][:, cols], mu0, sd0)
                p[te] = lda_score(X, y, tr, te)
            S[f'{nm}__causal_norm_s{sd}'] = p
        print(f'[{nm}] scores done {C.el()}', flush=True)
    S['lr889__strict'] = base; S['lr889__lit_kfold'] = C.lit_baseline(D, folds)
    for name, p in S.items():
        res['methods'][name] = C.report(name, p, D, ev_lit if '__lit' in name else ev, base, nb)
    # single-feature AUC: pooled (orientation from training rows of each fold) + median within-subject AUC
    for j, fn in enumerate(res['meta']['features']):
        for proto, X in (('raw', TH), ('sessnorm', sessnorm(TH, g))):
            p = np.full(len(y), np.nan)
            for k in folds:
                tr, te = np.flatnonzero(D['fold'] != k), np.flatnonzero(D['fold'] == k)
                sgn = 1. if roc_auc_score(y[tr], X[tr, j]) >= .5 else -1.
                p[te] = sgn * X[te, j]
            ws = [roc_auc_score(y[g == s], p[g == s]) for s in np.unique(g[ev])]
            res['single_feature_auc'][f'{fn}__{proto}'] = {
                'pooled_AUC': float(roc_auc_score(y[ev], p[ev])), 'median_within_subject_AUC': float(np.median(ws)),
                'frac_subjects_AUC_gt_.5': float(np.mean(np.array(ws) > .5))}
            print(f'[single] {fn:24s} {proto:8s} pooledAUC={roc_auc_score(y[ev], p[ev]):.4f} '
                  f'median within-subj AUC={np.median(ws):.4f} frac>.5={np.mean(np.array(ws) > .5):.2f}', flush=True)
    # checks
    tr, te = np.flatnonzero(D['fold'] != 0), np.flatnonzero(D['fold'] == 0)
    res['checks']['test_label_shuffle'] = C.check_label_shuffle(lambda yy: lda_score(TH, yy, tr, te), D, 0)
    loc = C.stream_order(g, subs, 0)[0][subs.index(np.unique(g[te])[0])]
    res['checks']['prefix_causal_norm'] = C.check_prefix(lambda Xs: causal_norm_subject(Xs, TH[tr].mean(0), TH[tr].std(0))[:, 0], TH[loc])
    print(f'[check] fold equality, test-label shuffle, prefix invariance OK {res["checks"]}', flush=True)
    C.finish(d, stem, tee, res)


if __name__ == '__main__':
    main()
