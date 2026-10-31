"""SVM on band-power / spectral features (arXiv 2408.13930 style) + SVM on our 889-dim features.

Features: band = log abs power delta/theta/alpha/beta/gamma (Welch 2 s, 50 %), log relative power, log total (11 dims);
psd = band + 44 log 1-Hz bins 1-45 Hz (55 dims); f889 = 858 + 31 features.
Classifiers: SVC(RBF, C=1, gamma='scale'), LinearSVC(C=.01); score = sigmoid(decision) (threshold .5 == margin 0).
Protocols: strict (subject folds, train-fold StandardScaler), lit_kfold (window-level KFold(5)),
           sessnorm (strict folds but each subject z-scored with its whole session = literature normalisation).
Run: python svm_spectral.py [--smoke]
"""
import argparse, sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402


class SqHingeSVM:
    """Linear SVM, LinearSVC objective (L2 + C * sum squared hinge, with intercept), solved by L-BFGS: liblinear
    did not converge in 5000 iterations / 145 s per fold on the collinear 889-dim set."""

    def __init__(self, C=.01): self.C = C

    def fit(self, X, y):
        from scipy.optimize import minimize
        s = 2. * y - 1.

        def f(wb):
            w, b = wb[:-1], wb[-1]; m = np.maximum(0., 1 - s * (X @ w + b))
            gr = -2 * self.C * (s * m)
            return .5 * w @ w + self.C * (m ** 2).sum(), np.r_[w + X.T @ gr, gr.sum()]
        r = minimize(f, np.zeros(X.shape[1] + 1), jac=True, method='L-BFGS-B', options={'maxiter': 2000})
        self.w, self.b = r.x[:-1], r.x[-1]; return self

    def decision_function(self, X): return X @ self.w + self.b


def clf(kind):
    return SVC(C=1., kernel='rbf', gamma='scale', cache_size=2000) if kind == 'rbf' else SqHingeSVM(.01)


def sessnorm(X, g):
    Xo = X.copy()
    for s in np.unique(g):
        q = g == s; Xo[q] = (X[q] - X[q].mean(0)) / (X[q].std(0) + 1e-9)
    return Xo


def fit_score(X, y, tr, te, kind):
    sc = StandardScaler().fit(X[tr])
    with threadpool_limits(limits=8):
        m = clf(kind).fit(sc.transform(X[tr]), y[tr])
    return C.sig(m.decision_function(sc.transform(X[te])))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); args = ap.parse_args()
    d, stem, tee = C.start_log('svm_spectral', args.smoke)
    D = C.setup(); R = C.raw_features(D); y, g = D['y'], D['g']
    folds = [0] if args.smoke else list(range(5)); nb = 100 if args.smoke else 1000
    feats = {'band': R['band'], 'psd': np.hstack([R['band'], R['psd']])}
    base = C.strict_baseline(D, folds)
    res = {'meta': {'folds': folds, 'smoke': args.smoke, 'ref_pipeline': C.REF_PIPELINE}, 'checks': {}, 'methods': {}}
    S = {}
    jobs = []
    for f in ('band', 'psd', 'f889'):
        for kind in ('rbf', 'lin'):
            for proto in ('strict', 'lit_kfold') + (('sessnorm',) if f != 'f889' else ()):
                if args.smoke and f == 'f889' and kind == 'rbf': continue
                jobs.append((f, kind, proto))
    for j, (f, kind, proto) in enumerate(jobs):
        name = f'svm_{kind}_{f}__{proto}'; p = np.full(len(y), np.nan)
        fk = D['lit_fold'] if proto == 'lit_kfold' else D['fold']
        for k in folds:
            tr, te = np.flatnonzero(fk != k), np.flatnonzero(fk == k)
            if f == 'f889': X = C.Z889(D, tr)
            else: X = sessnorm(feats[f], g) if proto == 'sessnorm' else feats[f]
            p[te] = fit_score(X, y, tr, te, kind)
            print(f'  [{j + 1}/{len(jobs)}] {name} fold {k + 1}/5 {C.el()}', flush=True)
        S[name] = p
    S['lr889__strict'] = base
    S['lr889__lit_kfold'] = C.lit_baseline(D, folds)
    ev = np.isin(D['fold'], folds); ev_lit = ev & np.isin(D['lit_fold'], folds)
    for name, p in S.items():
        res['methods'][name] = C.report(name, p, D, ev_lit if '__lit' in name else ev, base, nb)
    # check: test-label shuffle invariance (strict, RBF on bands, fold 0)
    tr, te = np.flatnonzero(D['fold'] != 0), np.flatnonzero(D['fold'] == 0)
    res['checks']['test_label_shuffle'] = C.check_label_shuffle(lambda yy: fit_score(feats['band'], yy, tr, te, 'rbf'), D, 0)
    res['checks']['strict_fold_equality'] = 'ok'
    print(f'[check] fold equality + test-label shuffle invariance OK', flush=True)
    C.finish(d, stem, tee, res)


if __name__ == '__main__':
    main()
