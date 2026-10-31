"""Conditional transfer gating (Wu-style 'only adapt when beneficial', label-free criterion on past windows).

Two experts per outer fold (889 features, LR C=.01):
  base   cross-subject LR on train-fold-standardised features (no adaptation)
  adapt  subject-adapted LR: every subject (train and test) causally re-standardised with its own PAST windows
         (running mean/var, prior N(0,1) pseudo-count 16) in stream order; LR trained on adapted training rows.
Label-free diagnostics for window t of a subject, from rows < t only (running mean, prior 0 / pseudo-count 8):
  d1 = mean(|logit_adapt|) - mean(|logit_base|) (confidence gap), d2 = agreement rate of the two experts,
  d3 = |mean sigmoid(adapt) - .5| (class-balance deviation), d4 = log(1 + t).
Gates:
  gate_conf     fixed rule: use adapt iff d1 > 0 (adapted expert more confident on the past windows)
  gate_learned  LR on (d1..d4) trained on TRAINING subjects only: inner GroupKFold(4) OOF expert scores; target =
                'adapt bACC > base bACC' for that training subject. hard switch at .5 and soft mix (p * adapt +
                (1-p) * base logits).
  gate_oracle   (leaky ceiling) per test subject pick the expert with the higher true bACC.
LIT counterparts: sessnorm (whole-session per-subject z-score) expert, gate_conf_session (gap over whole session).
Causal parts: stream seeds 0,1,2. Run: python conditional_tl.py [--smoke]
"""
import argparse, sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

P0, PD = 16., 8.


def causal_mean(X, prior, p0):
    """Row t: (p0 * prior + sum_{j<t} X_j) / (p0 + t)."""
    cs = np.vstack([np.zeros((1,) + X.shape[1:]), np.cumsum(X, 0)[:-1]])
    t = np.arange(len(X)).reshape((-1,) + (1,) * (X.ndim - 1))
    return (p0 * prior + cs) / (p0 + t)


def causal_norm(Zs):
    mu = causal_mean(Zs, 0., P0); m2 = causal_mean(Zs ** 2, 1., P0)
    return (Zs - mu) / np.sqrt(np.maximum(m2 - mu ** 2, 1e-6))


def diags(lb, la):
    q = np.c_[np.abs(la) - np.abs(lb), (np.sign(la) == np.sign(lb)).astype(float), C.sig(la)]
    m = causal_mean(q, np.array([0., 1., .5]), PD)
    return np.c_[m[:, 0], m[:, 1], np.abs(m[:, 2] - .5), np.log1p(np.arange(len(lb)))]


def lr(c=C.C_BASE):
    return LogisticRegression(C=c, max_iter=2000)


def adapt_all(Z, order):
    A = np.zeros_like(Z)
    for loc in order: A[loc] = causal_norm(Z[loc])
    return A


def bacc(y, l): return balanced_accuracy_score(y, l > 0) if np.unique(y).size == 2 else np.nan


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); args = ap.parse_args()
    d, stem, tee = C.start_log('conditional_tl', args.smoke)
    D = C.setup(); y, g, subs = D['y'], D['g'], D['subs']; n = len(y)
    folds = [0] if args.smoke else list(range(5)); nb = 100 if args.smoke else 1000
    sseeds = (0,) if args.smoke else C.STREAM_SEEDS
    base = C.strict_baseline(D, folds); ev = np.isin(D['fold'], folds)
    res = {'meta': {'folds': folds, 'stream_seeds': list(sseeds), 'smoke': args.smoke, 'pseudo_count_norm': P0,
                    'pseudo_count_diag': PD, 'leaky_rows': ['gate_oracle']},
           'checks': {'strict_fold_equality': 'ok'}, 'methods': {}, 'gate_usage': {}}
    S = {}
    Zs, inner_base = {}, {}
    for k in folds:
        tr, te = np.flatnonzero(D['fold'] != k), np.flatnonzero(D['fold'] == k)
        Z = C.Z889(D, tr); Zs[k] = Z
        # LIT expert: whole-session per-subject z-score
        ZS = Z.copy()
        for s in subs:
            q = g == s; ZS[q] = (Z[q] - Z[q].mean(0)) / (Z[q].std(0) + 1e-6)
        with threadpool_limits(limits=8):
            m_base = lr().fit(Z[tr], y[tr]); lbs = m_base.decision_function(Z)
            ls = lr().fit(ZS[tr], y[tr]).decision_function(ZS)
            ib = np.zeros(n)
            inner = list(GroupKFold(4).split(tr, y[tr], g[tr]))
            for a, b in inner: ib[tr[b]] = lr().fit(Z[tr[a]], y[tr[a]]).decision_function(Z[tr[b]])
        inner_base[k] = (ib, inner, lbs, m_base)
        S.setdefault('sessnorm_expert__lit', np.full(n, np.nan))[te] = C.sig(ls[te])
        gs = np.full(n, np.nan)
        for s in np.unique(g[te]):
            q = g == s; gs[q] = ls[q] if np.abs(ls[q]).mean() > np.abs(lbs[q]).mean() else lbs[q]
        S.setdefault('gate_conf_session__lit', np.full(n, np.nan))[te] = C.sig(gs[te])
        print(f'[fold {k + 1}/5] base/sessnorm/inner-base done {C.el()}', flush=True)
    usage = {}
    for sd in sseeds:
        order, _ = C.stream_order(g, subs, sd); loc_of = {s: order[j] for j, s in enumerate(subs)}
        for k in folds:
            tr, te = np.flatnonzero(D['fold'] != k), np.flatnonzero(D['fold'] == k)
            Z = Zs[k]; ib, inner, lbs, m_b = inner_base[k]; A = adapt_all(Z, order)
            with threadpool_limits(limits=8):
                m_ad = lr().fit(A[tr], y[tr]); la = m_ad.decision_function(A)
                ia = np.zeros(n)
                for a, b in inner: ia[tr[b]] = lr().fit(A[tr[a]], y[tr[a]]).decision_function(A[tr[b]])
            # gate training rows (training subjects only, inner OOF experts)
            G_X, G_y = [], []
            for s in np.unique(g[tr]):
                loc = loc_of[s]; G_X.append(diags(ib[loc], ia[loc]))
                G_y.append(np.full(len(loc), float(bacc(y[loc], ia[loc]) > bacc(y[loc], ib[loc]))))
            G_X, G_y = np.vstack(G_X), np.concatenate(G_y)
            gate = LogisticRegression(C=1., max_iter=2000).fit(G_X, G_y)
            out = {v: np.full(n, np.nan) for v in ('adapt_expert', 'gate_conf', 'gate_learned', 'gate_soft', 'gate_oracle')}
            for s in np.unique(g[te]):
                loc = loc_of[s]; lb_, la_ = lbs[loc], la[loc]; Dg = diags(lb_, la_)
                pg = gate.predict_proba(Dg)[:, 1]
                out['adapt_expert'][loc] = la_
                out['gate_conf'][loc] = np.where(Dg[:, 0] > 0, la_, lb_)
                out['gate_learned'][loc] = np.where(pg > .5, la_, lb_)
                out['gate_soft'][loc] = pg * la_ + (1 - pg) * lb_
                out['gate_oracle'][loc] = la_ if bacc(y[loc], la_) > bacc(y[loc], lb_) else lb_
                usage.setdefault(sd, []).append({'subject': s, 'frac_conf_adapt': float((Dg[:, 0] > 0).mean()),
                                                 'frac_learned_adapt': float((pg > .5).mean()),
                                                 'adapt_better_true': bool(bacc(y[loc], la_) > bacc(y[loc], lb_))})
            for v, p in out.items(): S.setdefault(f'{v}__s{sd}', np.full(n, np.nan))[te] = C.sig(p[te])
            print(f'  [stream {sd} fold {k + 1}/5] adapt expert + gates done; gate train pos-rate={G_y.mean():.2f} {C.el()}', flush=True)
            if sd == sseeds[0] and k == folds[0]:
                s0 = np.unique(g[te])[0]; loc = loc_of[s0]
                def pipe(Zsub):
                    lb_, la_ = m_b.decision_function(Zsub), m_ad.decision_function(causal_norm(Zsub))
                    Dg = diags(lb_, la_); pg = gate.predict_proba(Dg)[:, 1]
                    return np.c_[np.where(Dg[:, 0] > 0, la_, lb_), pg * la_ + (1 - pg) * lb_]
                dmax = float(np.abs(pipe(Z[loc])[:, 1] - out['gate_soft'][loc]).max())
                assert dmax < 1e-8, ('pipeline != pooled', dmax)
                res['checks']['prefix_gate_pipeline'] = C.check_prefix(lambda Zsub: pipe(Zsub)[:, 1], Z[loc])

                def lab_fn(yy):
                    G_y2 = np.concatenate([np.full(len(loc_of[s]), float(bacc(yy[loc_of[s]], ia[loc_of[s]]) >
                                                                           bacc(yy[loc_of[s]], ib[loc_of[s]])))
                                           for s in np.unique(g[tr])])
                    gt = LogisticRegression(C=1., max_iter=2000).fit(G_X, G_y2)
                    return gt.predict_proba(diags(lbs[te], la[te]))[:, 1] * la[te]
                res['checks']['test_label_shuffle_gate'] = C.check_label_shuffle(lab_fn, D, k)
                print(f'[check] prefix invariance + test-label shuffle OK {res["checks"]}', flush=True)
    res['gate_usage'] = {f's{sd}': u for sd, u in usage.items()}
    S['base_lr889__strict'] = base
    for name, p in S.items():
        res['methods'][name] = C.report(name, p, D, ev, base, nb)
    for v in ('adapt_expert', 'gate_conf', 'gate_learned', 'gate_soft', 'gate_oracle'):
        r = [res['methods'][f'{v}__s{sd}'] for sd in sseeds]
        res['methods'][f'{v}__seedmean'] = {q: float(np.mean([x[q] for x in r])) for q in ('AUC', 'subject_macro_bACC', 'n_subj_bacc_ge_0.9')}
        print(f'[seedmean] {v:14s} ' + ' '.join(f'{q}={x:.4f}' for q, x in res['methods'][f'{v}__seedmean'].items()), flush=True)
    u = res['gate_usage'][f's{sseeds[0]}']
    agree = np.mean([(x['frac_learned_adapt'] > .5) == x['adapt_better_true'] for x in u])
    res['gate_decision_accuracy_vs_truth_s0'] = float(agree)
    print(f'[gate] learned-gate majority decision matches true better expert for {agree:.2f} of test subjects '
          f'(true adapt-better rate {np.mean([x["adapt_better_true"] for x in u]):.2f})', flush=True)
    C.finish(d, stem, tee, res)


if __name__ == '__main__':
    main()
