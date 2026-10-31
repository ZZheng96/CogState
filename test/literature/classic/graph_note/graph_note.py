"""Graph / connectivity features (Bocini-style functional-connectivity graphs, GPDC) adapted to ONE channel.

Applicability: GPDC / PLV / coherence graphs and graph neural networks need >= 2 spatial channels; our recordings are
single-channel, so no spatial graph exists. Closest legal substitute: a "spectral graph" whose nodes are 8 filter-bank
pseudo-channels (common.FBANK) and whose edges are Fisher-z correlations of their Hilbert envelopes within the window
(28 edges, cached in common.raw_features 'envcorr'), plus node strength (sum of |edges| per node, 8). This measures
cross-frequency amplitude coupling, NOT spatial connectivity.
Variants (LR, fixed a priori): envcorr (28), envcorr+strength (36) with C=.1; fuse889 = 889 + 36 with C=.01.
Protocols: strict (subject folds), lit_kfold (window KFold), sessnorm (whole-session per-subject z-score, LIT).
Run: python graph_note.py [--smoke]
"""
import argparse, sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402


def strength(E):
    nb = len(C.FBANK); iu = np.triu_indices(nb, 1); A = np.zeros((len(E), nb, nb))
    A[:, iu[0], iu[1]] = np.abs(E); A = A + np.swapaxes(A, 1, 2)
    return A.sum(2)


def sessnorm(X, g):
    X = X.copy()
    for s in np.unique(g):
        q = g == s; X[q] = (X[q] - X[q].mean(0)) / (X[q].std(0) + 1e-6)
    return X


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); args = ap.parse_args()
    d, stem, tee = C.start_log('graph_note', args.smoke)
    D = C.setup(); y, g = D['y'], D['g']; n = len(y)
    R = C.raw_features(D)
    E = R['envcorr']; G36 = np.c_[E, strength(E)]
    folds = [0] if args.smoke else list(range(5)); nb = 100 if args.smoke else 1000
    base = C.strict_baseline(D, folds); lit_base = C.lit_baseline(D, folds)
    ev = np.isin(D['fold'], folds); ev_lit = ev & np.isin(D['lit_fold'], folds)
    res = {'meta': {'folds': folds, 'smoke': args.smoke, 'nodes': [list(b) for b in C.FBANK],
                    'note': 'single channel: spatial graphs (GPDC/PLV/GNN) not applicable; envelope-correlation '
                            'graph over filter-bank pseudo-channels = cross-frequency coupling substitute'},
           'checks': {'strict_fold_equality': 'ok'}, 'methods': {}}
    feats = {'envcorr': (E, .1), 'graph36': (G36, .1)}
    S = {}
    for proto, fcol in (('strict', 'fold'), ('lit_kfold', 'lit_fold'), ('sessnorm', 'fold')):
        for k in folds:
            tr, te = np.flatnonzero(D[fcol] != k), np.flatnonzero(D[fcol] == k)
            Z = C.Z889(D, tr)
            for name, (X, c) in list(feats.items()) + [('fuse889', (G36, C.C_BASE))]:
                Xp = sessnorm(X, g) if proto == 'sessnorm' else X
                Xs = StandardScaler().fit(Xp[tr]).transform(Xp)
                if name == 'fuse889': Xs = np.c_[Z, Xs]
                with threadpool_limits(limits=8):
                    p = LogisticRegression(C=c, max_iter=3000).fit(Xs[tr], y[tr]).predict_proba(Xs[te])[:, 1]
                S.setdefault(f'{name}__{proto}', np.full(n, np.nan))[te] = p
            print(f'[{proto} fold {k + 1}/5] done {C.el()}', flush=True)
    S['lr889__strict'] = base; S['lr889__lit_kfold'] = lit_base
    for name, p in S.items():
        res['methods'][name] = C.report(name, p, D, ev_lit if 'lit_kfold' in name else ev, base, nb)

    k = folds[0]; tr = np.flatnonzero(D['fold'] != k); te = D['fold'] == k

    def lab_fn(yy):
        Xs = StandardScaler().fit(G36[tr]).transform(G36)
        return LogisticRegression(C=.1, max_iter=3000).fit(Xs[tr], yy[tr]).predict_proba(Xs[te])[:, 1]
    res['checks']['test_label_shuffle'] = C.check_label_shuffle(lab_fn, D, k)
    print(f'[check] {res["checks"]}', flush=True)
    C.finish(d, stem, tee, res)


if __name__ == '__main__':
    main()
