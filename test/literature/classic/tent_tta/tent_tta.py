"""TENT-style test-time adaptation (Wang et al. 2021) on the 889-feature linear model.

Model: LR(C=.01) on train-fold-standardised 889 features, rewritten as BN-affine + Linear:
  logit = w . (gamma * (z - mu) / sd + beta) + b, gamma=1, beta=0, mu=0, sd=1 at the end of training (== plain LR).
TENT adapts (gamma, beta) by minimising prediction entropy on unlabelled test windows; BN statistics (mu, sd) can be
re-estimated on test data. Analytic gradient: dH/dlogit = -logit * p(1-p).
STRICT (per test subject, stream order, seeds 0,1,2): window t is scored with parameters/statistics computed from
windows < t only; then (gamma, beta) take one SGD step on the entropy of the last B=8 windows (<= t) -> used from t+1.
  none          plain LR (no adaptation)
  stats_causal  BN statistics re-estimated causally (running mean/var, prior N(0,1), pseudo-count 16), no affine update
  tent_causal_lr{.1,1}         stats_causal + entropy SGD on gamma/beta (lr fixed a priori)
  tent_affine_lr{.1,1}         train statistics kept, entropy SGD on gamma/beta only
LIT (whole session, future windows used): tent_session = session statistics + 20 full-batch entropy steps (lr 1).
Run: python tent_tta.py [--smoke]
"""
import argparse, sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

P0, B = 16., 8
LRS = (.1, 1.)


def tent_stream(Zs, w, b, lr, stats, affine):
    """Zs: one subject's rows in stream order. Returns logits; row t uses only rows < t."""
    n, d = Zs.shape
    gam, bet = np.ones(d), np.zeros(d)
    s1, s2 = np.zeros(d), np.zeros(d)
    out = np.zeros(n)
    for t in range(n):
        if stats:
            mu = s1 / (P0 + t); var = (P0 + s2) / (P0 + t) - mu ** 2
            sd = np.sqrt(np.maximum(var, 1e-6))
        else:
            mu, sd = 0., 1.
        out[t] = w @ (gam * (Zs[t] - mu) / sd + bet) + b
        s1 += Zs[t]; s2 += Zs[t] ** 2
        if affine:
            lo = max(0, t + 1 - B); zh = (Zs[lo:t + 1] - mu) / sd
            l = (zh * gam + bet) @ w + b; p = C.sig(l)
            gl = -l * p * (1 - p)  # dH/dlogit
            gam -= lr * (gl[:, None] * w[None] * zh).mean(0)
            bet -= lr * (gl[:, None] * w[None]).mean(0)
    return out


def tent_session(Zs, w, b, lr=1., steps=20):
    zh = (Zs - Zs.mean(0)) / (Zs.std(0) + 1e-6); d = Zs.shape[1]
    gam, bet = np.ones(d), np.zeros(d)
    for _ in range(steps):
        l = (zh * gam + bet) @ w + b; p = C.sig(l); gl = -l * p * (1 - p)
        gam -= lr * (gl[:, None] * w[None] * zh).mean(0); bet -= lr * (gl[:, None] * w[None]).mean(0)
    return (zh * gam + bet) @ w + b


VARS = [('stats_causal', None, True, False)] + [(f'tent_causal_lr{lr:g}', lr, True, True) for lr in LRS] + \
       [(f'tent_affine_lr{lr:g}', lr, False, True) for lr in LRS]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); args = ap.parse_args()
    d, stem, tee = C.start_log('tent_tta', args.smoke)
    D = C.setup(); y, g, subs = D['y'], D['g'], D['subs']; n = len(y)
    folds = [0] if args.smoke else list(range(5)); nb = 100 if args.smoke else 1000
    sseeds = (0,) if args.smoke else C.STREAM_SEEDS
    base = C.strict_baseline(D, folds); ev = np.isin(D['fold'], folds)
    res = {'meta': {'folds': folds, 'stream_seeds': list(sseeds), 'smoke': args.smoke, 'pseudo_count': P0,
                    'buffer': B, 'lrs': list(LRS)}, 'checks': {'strict_fold_equality': 'ok'}, 'methods': {}}
    S = {}
    for k in folds:
        tr, te = np.flatnonzero(D['fold'] != k), np.flatnonzero(D['fold'] == k)
        Z = C.Z889(D, tr)
        with threadpool_limits(limits=8):
            m = LogisticRegression(C=C.C_BASE, max_iter=2000).fit(Z[tr], y[tr])
        w, b = m.coef_.ravel(), float(m.intercept_[0])
        lt = Z[te] @ w + b
        S.setdefault('none__strict', np.full(n, np.nan))[te] = C.sig(lt)
        assert np.allclose(C.sig(lt), base[te], atol=1e-8), 'BN-affine rewrite != plain LR'
        ses = np.full(n, np.nan)
        for s in np.unique(g[te]):
            q = np.flatnonzero(g == s); ses[q] = tent_session(Z[q], w, b)
        S.setdefault('tent_session__lit', np.full(n, np.nan))[te] = C.sig(ses[te])
        for sd in sseeds:
            order, _ = C.stream_order(g, subs, sd)
            te_subs = [j for j, s in enumerate(subs) if D['fold'][order[j][0]] == k]
            for name, lr, st, af in VARS:
                key = f'{name}__s{sd}'; S.setdefault(key, np.full(n, np.nan))
                for j in te_subs:
                    loc = order[j]; S[key][loc] = C.sig(tent_stream(Z[loc], w, b, lr, st, af))
            print(f'[fold {k + 1}/5 stream {sd}] {len(te_subs)} test subjects x {len(VARS)} variants {C.el()}', flush=True)
        if k == folds[0]:
            loc = order[te_subs[0]]
            res['checks']['prefix_tent_causal'] = C.check_prefix(lambda X: tent_stream(X, w, b, 1., True, True), Z[loc])
            res['checks']['prefix_tent_affine'] = C.check_prefix(lambda X: tent_stream(X, w, b, 1., False, True), Z[loc])

            def lab_fn(yy):
                with threadpool_limits(limits=8):
                    mm = LogisticRegression(C=C.C_BASE, max_iter=2000).fit(Z[tr], yy[tr])
                return tent_stream(Z[loc], mm.coef_.ravel(), float(mm.intercept_[0]), 1., True, True)
            res['checks']['test_label_shuffle'] = C.check_label_shuffle(lab_fn, D, k)
            print(f'[check] {res["checks"]}', flush=True)
    for name, p in S.items():
        res['methods'][name] = C.report(name, p, D, ev, base, nb)
    for name, *_ in VARS:
        r = [res['methods'][f'{name}__s{sd}'] for sd in sseeds]
        res['methods'][f'{name}__seedmean'] = {q: float(np.mean([x[q] for x in r])) for q in ('AUC', 'subject_macro_bACC', 'n_subj_bacc_ge_0.9')}
        print(f'[seedmean] {name:20s} ' + ' '.join(f'{q}={x:.4f}' for q, x in res['methods'][f'{name}__seedmean'].items()), flush=True)
    C.finish(d, stem, tee, res)


if __name__ == '__main__':
    main()
