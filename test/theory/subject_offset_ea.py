"""D3: per-subject causal diagonal alignment ("EA") of 1 Hz log-PSD (legal, strict protocol).

Input P: per-window log Welch PSD, 44 bins 1..44 Hz (nperseg 500, raw uV signal minus window mean), from
test/literature/classic/_cache_rawfeat.npz 'psd' (y / groups asserted == canonical).
Outer folds SGKF(5, shuffle, rs0) (== npz). Per fold, prior mu0 = mean of P over fold-train rows, v0 = mean
within-subject variance over fold-train subjects (label-free). Every subject (train and test alike) is centred causally in
its own stream order: window t uses only that subject's windows < t plus the prior with pseudo-count 16.
Variants (features -> train-row StandardScaler -> LR(C=.01)):
  psd_raw          P (no subject centring; reference)
  ea_mean          P_t - m_t,  m_t = (16 mu0 + sum_{i<t} P_i) / (16 + t)
  ea_mean_var      (P_t - m_t) / sqrt(v_t), running second moment with the same prior (v_t >= .1 v0)
  ea_median        P_t - median([mu0 x16, P_0..P_{t-1}])
  fooof_raw        [R, offset, exponent]: per window np.polyfit(log10 f, P, 1) over 2-40 Hz, R = P - fit (fooof pkg absent)
  fooof_ea_mean / fooof_ea_mean_var / fooof_ea_median  same centring on [R, offset, exponent]
  fuse889_<v>      hstack(889 Z, variant Z) -> LR(C=.01)
Baseline: LR(C=.01) on 889 (strict). Orders: stream seeds 0,1,2 (shuffled, project protocol) + 'real' (recovered real
recording order, extra). Paired subject bootstrap vs baseline (1000). Checks: fold == npz; test-label shuffle (fold 0);
prefix invariance of the full centring -> score pipeline for one test subject per variant.
Run: python test/theory/subject_offset_ea.py [--smoke]   (smoke: fold 0, stream 0, boot 100)
"""
import argparse, sys, time

import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from _common import GROUPS, STREAM_SEEDS, TEST, boot, el, impute_scale, load, metrics, open_log, save

PRIOR = 16.
F_HZ = np.arange(1, 45) + .5
BASE = ('psd_raw', 'ea_mean', 'ea_mean_var', 'ea_median')
VARIANTS = BASE + tuple('fooof_' + v.replace('psd_', '') for v in BASE)


def lr():
    return LogisticRegression(C=.01, max_iter=2000)


def stream_order(g, subs, seed):  # == depositioned_resnet_history_delta.stream_order
    return [np.flatnonzero(g == s)[np.random.default_rng([seed, j]).permutation(int((g == s).sum()))]
            for j, s in enumerate(subs)]


def fooof(P):
    sel = (F_HZ >= 2) & (F_HZ <= 40); lf = np.log10(F_HZ)
    b, a = np.polyfit(lf[sel], P[:, sel].T, 1)  # slope, offset per window
    return np.c_[P - (a[:, None] + b[:, None] * lf[None]), a, -b]


def center(X, mu0, v0, mode):
    """X rows in stream order of ONE subject -> causally centred rows."""
    n = len(X); t = np.arange(n)[:, None]
    if mode == 'median':
        out = np.empty_like(X); pri = np.repeat(mu0[None], int(PRIOR), 0)
        for i in range(n): out[i] = X[i] - np.median(np.vstack([pri, X[:i]]), 0)
        return out
    cs = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(X, 0)[:-1]])
    m = (PRIOR * mu0 + cs) / (PRIOR + t)
    if mode == 'mean':
        return X - m
    cs2 = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(X ** 2, 0)[:-1]])
    v = (PRIOR * (v0 + mu0 ** 2) + cs2) / (PRIOR + t) - m ** 2
    return (X - m) / np.sqrt(np.maximum(v, .1 * v0))


def prior(X, tr, g):
    mu0 = X[tr].mean(0)
    v0 = np.mean([X[tr][g[tr] == s].var(0) for s in np.unique(g[tr])], 0)
    return mu0, v0


def variant_X(name, P, PF, tr, g, order):
    X = PF if name.startswith('fooof_') else P
    mode = name.replace('fooof_', '').replace('psd_', '')
    if mode == 'raw': return X
    mu0, v0 = prior(X, tr, g); out = np.empty_like(X)
    cm = {'ea_mean': 'mean', 'ea_mean_var': 'var', 'ea_median': 'median'}[mode]
    for loc in order: out[loc] = center(X[loc], mu0, v0, cm)
    return out


def scale(X, tr):
    mu, sd = X[tr].mean(0), X[tr].std(0); return (X - mu) / np.where(sd > 1e-12, sd, 1.)


def fit_scores(X, y, tr, te):
    with threadpool_limits(limits=8):
        m = lr().fit(X[tr], y[tr])
    return m.predict_proba(X[te])[:, 1], m


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); args = ap.parse_args()
    stem = 'subject_offset_ea' + ('_smoke' if args.smoke else ''); tee = open_log(stem)
    D = load(); y, g, fold, subs, grp_of = D['y'], D['g'], D['fold'], D['subs'], D['grp_of']; n = len(y)
    C = np.load(TEST / 'literature' / 'classic' / '_cache_rawfeat.npz')
    assert np.array_equal(C['y'], y) and np.array_equal(C['g'], g)
    P = C['psd'].astype(float); PF = fooof(P)
    print(f'[psd] {P.shape} fooof-style features {PF.shape} (np.polyfit, fooof package not installed) {el()}', flush=True)
    run_folds = [0] if args.smoke else list(range(5)); seeds = STREAM_SEEDS[:1] if args.smoke else STREAM_SEEDS
    nb = 100 if args.smoke else 1000
    ev = np.isin(fold, run_folds)
    orders = {f's{sd}': stream_order(g, subs, sd) for sd in seeds}
    orders['real'] = [np.flatnonzero(g == s)[np.argsort(D['t'][g == s], kind='stable')] for s in subs]
    res = {'meta': {'prior_pseudocount': PRIOR, 'psd': '44 x 1 Hz log Welch bins 1-44 Hz', 'fooof': 'np.polyfit 2-40 Hz',
                    'C': .01, 'folds': run_folds, 'orders': list(orders), 'bootstrap_n': nb, 'smoke': args.smoke,
                    'note_real_order': 'real = recovered recording order (extra; the project protocol is s0/s1/s2)'},
           'checks': {}, 'methods': {}}
    # baseline 889
    Z889 = {}; pb = np.full(n, np.nan)
    for k in run_folds:
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        tfo, tfn = impute_scale(D['Fc'], tr), impute_scale(D['Fn'], tr)
        Z889[k] = np.hstack([tfo(D['Fc']), tfn(D['Fn'])]); pb[te] = fit_scores(Z889[k], y, tr, te)[0]
    S = {'strict_LR889': pb}
    for oname, order in orders.items():
        for v in VARIANTS:
            to = time.perf_counter(); p, pf = np.full(n, np.nan), np.full(n, np.nan)
            for k in run_folds:
                tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
                X = scale(variant_X(v, P, PF, tr, g, order), tr)
                p[te], m = fit_scores(X, y, tr, te)
                pf[te], mf = fit_scores(np.hstack([Z889[k], X]), y, tr, te)
                if k == run_folds[0] and oname == list(orders)[0]:
                    y2 = y.copy(); y2[te] = np.random.default_rng(k).permutation(y2[te])
                    assert np.array_equal(p[te], fit_scores(X, y2, tr, te)[0]), ('label shuffle', v)
                    res['checks'][f'{v}_fold{k}_test_label_shuffle'] = 'ok'
                    if v != 'psd_raw' and v != 'fooof_raw':  # prefix invariance of centring -> scaler -> model
                        s0 = [s for s in subs if D['fold_of'][s] == k][0]; loc = order[subs.index(s0)]
                        src = PF if v.startswith('fooof_') else P
                        mu0, v0 = prior(src, tr, g); Xs = scale(variant_X(v, P, PF, tr, g, order), tr)
                        sc_mu = variant_X(v, P, PF, tr, g, order)[tr].mean(0); sc_sd = variant_X(v, P, PF, tr, g, order)[tr].std(0)
                        cm = {'ea_mean': 'mean', 'ea_mean_var': 'var', 'ea_median': 'median'}[v.replace('fooof_', '')]

                        def fn(Xraw):
                            Zc = (center(Xraw, mu0, v0, cm) - sc_mu) / np.where(sc_sd > 1e-12, sc_sd, 1.)
                            return m.decision_function(Zc)
                        assert np.allclose(fn(src[loc]), m.decision_function(Xs[loc]), atol=1e-10)
                        cut = len(loc) // 2; a = fn(src[loc]); Xp = src[loc].copy(); Xp[cut:] = Xp[cut:][::-1] * 1.7 + .3
                        b = fn(Xp); dpre, dfut = float(np.abs(a[:cut] - b[:cut]).max()), float(np.abs(a[cut:] - b[cut:]).max())
                        assert dpre == 0. and dfut > 0., ('prefix', v, dpre, dfut)
                        res['checks'][f'{v}_prefix_invariance_{s0}'] = {'max_abs_prefix_change': dpre,
                                                                        'max_abs_future_change': dfut}
            S[f'{oname}__{v}'] = p; S[f'{oname}__fuse889_{v}'] = pf
            print(f'[{oname} {v}] AUC={metrics(y, g, p, ev)["AUC"]:.4f} fuse AUC={metrics(y, g, pf, ev)["AUC"]:.4f} '
                  f'variant_s={time.perf_counter() - to:.1f} {el()}', flush=True)
    for name, p in S.items():
        m = metrics(y, g, p, ev); per = {}
        for s in np.unique(g[ev]):
            q = metrics(y, g, p, ev & (g == s))['subject_macro_bACC']
            if q is not None: per[s] = q
        m['groups_macro_bACC'] = {G: float(np.mean([v for s, v in per.items() if grp_of[s] == G])) for G in GROUPS}
        if name != 'strict_LR889':
            m['boot_vs_strict_LR889'] = boot(p, pb, y, g, ev, nb)
        res['methods'][name] = m; b = m.get('boot_vs_strict_LR889')
        print(f'[result] {name:30s} AUC={m["AUC"]:.4f} mac={m["subject_macro_bACC"]:.4f} ' +
              ' '.join(f'{G}={v:.3f}' for G, v in m['groups_macro_bACC'].items()) +
              (f' dAUC={b["dAUC"]["mean"]:+.4f}[{b["dAUC"]["p2.5"]:+.4f},{b["dAUC"]["p97.5"]:+.4f}] '
               f'dmac={b["dmacro_bACC"]["mean"]:+.4f}[{b["dmacro_bACC"]["p2.5"]:+.4f},{b["dmacro_bACC"]["p97.5"]:+.4f}]'
               if b else ''), flush=True)
    # seed-mean summary (project protocol streams only)
    res['seed_mean'] = {}
    for v in VARIANTS + tuple(f'fuse889_{v}' for v in VARIANTS):
        ks = [f's{sd}__{v}' for sd in seeds]
        res['seed_mean'][v] = {q: float(np.mean([res['methods'][k][q] for k in ks])) for q in ('AUC', 'subject_macro_bACC')}
        res['seed_mean'][v]['dAUC'] = float(np.mean([res['methods'][k]['boot_vs_strict_LR889']['dAUC']['mean'] for k in ks]))
        res['seed_mean'][v]['dmac'] = float(np.mean([res['methods'][k]['boot_vs_strict_LR889']['dmacro_bACC']['mean']
                                                     for k in ks]))
        print(f'[seed-mean] {v:24s} ' + ' '.join(f'{q}={x:+.4f}' for q, x in res['seed_mean'][v].items()), flush=True)
    save(stem, res, tee)


if __name__ == '__main__':
    main()
