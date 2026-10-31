"""Euclidean Alignment (He & Wu 2019) / Riemannian re-centring (Zanini 2018) + tangent space on filter-bank
pseudo-channels of our single channel.

Pseudo-multichannel: 8 zero-phase Butterworth-4 band-passes 1-4,4-6,6-8,8-10,10-13,13-20,20-30,30-45 Hz -> 8x8
covariance per window (uV^2, + ridge). Tangent vector = upper triangle of logm(M^-1/2 C M^-1/2) (36 dims).
Variants:
  ts_noalign          tangent space at the training-subject Riemannian mean M_tr (no per-subject alignment)
  ea_causal           EA: C_t whitened by the arithmetic mean of the subject's PAST windows (+ prior = train grand
                      arithmetic mean, pseudo-count 16), projected at I. STRICT: past only, stream order.
  ea_session          EA with the whole-session arithmetic mean (LIT).
  rpa_causal          Zanini re-centring: reference = online Riemannian mean of PAST windows (geodesic incremental
                      update, prior M_tr, pseudo-count 16); tangent at I after whitening. STRICT.
  rpa_session         reference = whole-session Riemannian mean (LIT, the original Zanini 2018 offline re-centring).
  ts_noalign__lit_kfold  window-level KFold(5).
Training subjects are aligned the same way as test subjects (causal variants: causally in the same stream seed).
Heads: LR(C=1) on standardised tangent vectors; SVC-RBF(C=1) for ts_noalign / ea_causal / rpa_causal;
fusion: LR(C=.01) on [889 features, tangent].
Causal variants: stream seeds 0,1,2 -> per-seed results + mean over seeds.
Run: python ea_tangent.py [--smoke]
"""
import argparse, sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

P0 = 16


def ea_causal_subject(Cs, prior):
    out = np.zeros((len(Cs), 36)); S = P0 * prior
    for t in range(len(Cs)):
        out[t] = C.tangent(Cs[t], S / (P0 + t)); S = S + Cs[t]
    return out


def rpa_causal_subject(Cs, prior):
    out = np.zeros((len(Cs), 36)); M = prior.copy()
    for t in range(len(Cs)):
        out[t] = C.tangent(Cs[t], M)
        M = C.geodesic(M, Cs[t], 1. / (P0 + t + 1))
    return out


def per_subject(Cv, g, order_locs, fn):
    T = np.zeros((len(Cv), 36))
    for loc in order_locs: T[loc] = fn(Cv[loc])
    return T


def head(X, y, tr, te, kind, c=1.):
    sc = StandardScaler().fit(X[tr]); A, B = sc.transform(X[tr]), sc.transform(X[te])
    with threadpool_limits(limits=8):
        if kind == 'svm': return C.sig(SVC(C=1., gamma='scale', cache_size=2000).fit(A, y[tr]).decision_function(B))
        return LogisticRegression(C=c, max_iter=3000).fit(A, y[tr]).predict_proba(B)[:, 1]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); args = ap.parse_args()
    d, stem, tee = C.start_log('ea_tangent', args.smoke)
    D = C.setup(); R = C.raw_features(D); y, g, subs = D['y'], D['g'], D['subs']
    Cv = C.covs(R); n = len(y)
    folds = [0] if args.smoke else list(range(5)); nb = 100 if args.smoke else 1000
    sseeds = (0,) if args.smoke else C.STREAM_SEEDS
    base = C.strict_baseline(D, folds); ev = np.isin(D['fold'], folds); ev_lit = ev & np.isin(D['lit_fold'], folds)
    res = {'meta': {'folds': folds, 'stream_seeds': list(sseeds), 'smoke': args.smoke, 'filter_bank': C.FBANK,
                    'pseudo_count': P0}, 'checks': {'strict_fold_equality': 'ok'}, 'methods': {}}
    S = {}
    by_sub = [np.flatnonzero(g == s) for s in subs]
    pre = {}
    for k in folds:
        tr, te = np.flatnonzero(D['fold'] != k), np.flatnonzero(D['fold'] == k)
        Mtr = C.riem_mean(Cv[tr]); Atr = Cv[tr].mean(0)
        T_no = C.tangent(Cv, Mtr)
        T_eas = np.zeros((n, 36)); T_rps = np.zeros((n, 36))
        for loc in by_sub:
            T_eas[loc] = C.tangent(Cv[loc], Cv[loc].mean(0)); T_rps[loc] = C.tangent(Cv[loc], C.riem_mean(Cv[loc]))
        pre[k] = (Mtr, Atr, tr, te)
        Z = C.Z889(D, tr)
        for nm, T in (('ts_noalign', T_no), ('ea_session', T_eas), ('rpa_session', T_rps)):
            S.setdefault(f'{nm}__lr', np.full(n, np.nan))[te] = head(T, y, tr, te, 'lr')
            S.setdefault(f'fuse889_{nm}__lr', np.full(n, np.nan))[te] = head(np.hstack([Z, T]), y, tr, te, 'lr', .01)
        S.setdefault('ts_noalign__svm', np.full(n, np.nan))[te] = head(T_no, y, tr, te, 'svm')
        print(f'[fold {k + 1}/5] order-free variants done {C.el()}', flush=True)
    # LIT window-level KFold on non-aligned tangent space
    p = np.full(n, np.nan)
    for k in folds:
        tr, te = np.flatnonzero(D['lit_fold'] != k), np.flatnonzero(D['lit_fold'] == k)
        p[te] = head(C.tangent(Cv, C.riem_mean(Cv[tr])), y, tr, te, 'lr')
    S['ts_noalign__lr__lit_kfold'] = p
    seed_scores = {}
    for sd in sseeds:
        order, _ = C.stream_order(g, subs, sd)
        for k in folds:
            Mtr, Atr, tr, te = pre[k]; Z = C.Z889(D, tr)
            for nm, fn in (('ea_causal', lambda Cs: ea_causal_subject(Cs, Atr)),
                           ('rpa_causal', lambda Cs: rpa_causal_subject(Cs, Mtr))):
                T = per_subject(Cv, g, order, fn)
                for hk, X, c in (('lr', T, 1.), ('svm', T, 1.), ('fuse889', np.hstack([Z, T]), .01)):
                    key = f'{nm}__{hk}' if hk != 'fuse889' else f'fuse889_{nm}__lr'
                    seed_scores.setdefault((key, sd), np.full(n, np.nan))[te] = head(X, y, tr, te, 'svm' if hk == 'svm' else 'lr', c)
                print(f'  [stream {sd} fold {k + 1}/5] {nm} done {C.el()}', flush=True)
    keys = sorted({k for k, _ in seed_scores})
    for key in keys:
        for sd in sseeds: S[f'{key}__s{sd}'] = seed_scores[(key, sd)]
    S['lr889__strict'] = base; S['lr889__lit_kfold'] = C.lit_baseline(D, folds)
    for name, p in S.items():
        res['methods'][name] = C.report(name, p, D, ev_lit if '__lit' in name else ev, base, nb)
    for key in keys:
        r = [res['methods'][f'{key}__s{sd}'] for sd in sseeds]
        res['methods'][f'{key}__seedmean'] = {q: float(np.mean([x[q] for x in r])) for q in
                                              ('AUC', 'subject_macro_bACC', 'n_subj_bacc_ge_0.9')}
        print(f'[seedmean] {key:30s} ' + ' '.join(f'{q}={v:.4f}' for q, v in res['methods'][f'{key}__seedmean'].items()), flush=True)
    # checks
    Mtr, Atr, tr, te = pre[folds[0]]; order, _ = C.stream_order(g, subs, 0)
    s0 = subs.index(np.unique(g[te])[0]); loc = order[s0]
    res['checks']['prefix_ea_causal'] = C.check_prefix(lambda Cs: ea_causal_subject(Cs, Atr), Cv[loc])
    res['checks']['prefix_rpa_causal'] = C.check_prefix(lambda Cs: rpa_causal_subject(Cs, Mtr), Cv[loc])
    T = per_subject(Cv, g, order, lambda Cs: rpa_causal_subject(Cs, Mtr))
    res['checks']['test_label_shuffle'] = C.check_label_shuffle(lambda yy: head(T, yy, tr, te, 'lr'), D, folds[0])
    print(f'[check] fold equality, prefix invariance (EA/RPA causal), test-label shuffle OK {res["checks"]}', flush=True)
    C.finish(d, stem, tee, res)


if __name__ == '__main__':
    main()
