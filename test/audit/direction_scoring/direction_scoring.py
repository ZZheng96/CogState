"""Direction scoring audit: does the cross-subject direction w_xs rank a held-out subject's windows like its own
direction w_s?  (replaces weight-cosine as evidence of direction disagreement)

Per outer fold k (SGKF(5, shuffle, rs0), asserted == npz fold), ZZ = 889 features (train-median impute + StandardScaler
on fold-train rows). w_xs = LR(C=.01) on all training-fold rows. For each held-out subject s:
  own_in   LR(C=.01) on s's own labelled windows, scored in-sample (LEAKY ceiling)
  own_cv   5-fold StratifiedKFold(shuffle, rs0) within s, out-of-fold scores (LEAKY: uses s's labels, but no row
           scores itself)
  xs       w_xs decision function on s (legal)
Reported per subject and per group (high/mid/low; low is post-hoc defined from pipeline performance -> circular):
  AUC of each, AUC_xs oriented = .5 + |AUC_xs - .5| (needs s's labels to orient), Spearman(xs, own_in),
  Spearman(xs, own_cv), cos(w_s, w_xs) vs random baseline 1/sqrt(889) and an empirical null cos(w_perm, w_xs) with
  w_perm = LR on s's windows with permuted labels (N_NULL permutations, seeds [k, subject index, i]).
Checks (assert): fold == npz; test-label shuffle leaves xs scores bit-identical; determinism of own_in / own_cv.
Run: python test/audit/direction_scoring/direction_scoring.py [--smoke]   (smoke: fold 0, 4 null permutations)
"""
import argparse, sys, time, warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import GROUPS, el, load_all, open_log, save, zz_fold  # noqa: E402

C_REG, N_JOBS = .01, 8


def lr():
    return LogisticRegression(C=C_REG, max_iter=1000)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def subject_job(Zs, ys, w_xs, b_xs, n_null, seed):
    with threadpool_limits(limits=1):
        xs = Zs @ w_xs + b_xs
        m = lr().fit(Zs, ys); w_s = m.coef_[0]; own_in = m.decision_function(Zs)
        own_cv = np.full(len(ys), np.nan)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Zs, ys):
            own_cv[te] = lr().fit(Zs[tr], ys[tr]).decision_function(Zs[te])
        null = []
        for i in range(n_null):
            yp = np.random.default_rng(seed + [i]).permutation(ys)
            null.append(cos(lr().fit(Zs, yp).coef_[0], w_xs))
    a_xs = roc_auc_score(ys, xs)
    return {'n': int(len(ys)), 'auc_xs': float(a_xs), 'auc_xs_oriented': float(.5 + abs(a_xs - .5)),
            'auc_own_in': float(roc_auc_score(ys, own_in)), 'auc_own_cv': float(roc_auc_score(ys, own_cv)),
            'spearman_xs_own_in': float(spearmanr(xs, own_in)[0]), 'spearman_xs_own_cv': float(spearmanr(xs, own_cv)[0]),
            'cos_ws_wxs': cos(w_s, w_xs), 'cos_null_mean': float(np.mean(null)),
            'cos_null_absmean': float(np.mean(np.abs(null))), 'cos_null_sd': float(np.std(null)),
            'cos_null_p_abs': float((1 + np.sum(np.abs(null) >= abs(cos(w_s, w_xs)))) / (1 + len(null))),
            '_own_in': own_in, '_own_cv': own_cv, '_xs': xs}


def summarize(vals, rng, nb):
    v = np.asarray(vals, float)
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(nb)]
    return {'n': int(len(v)), 'mean': float(v.mean()), 'median': float(np.median(v)),
            'ci95': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def main():
    warnings.filterwarnings('ignore')
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); args = ap.parse_args()
    stem = 'direction_scoring' + ('_smoke' if args.smoke else '')
    tee = open_log(HERE, stem)
    D = load_all(); y, g, fold, subs, grp_of = D['y'], D['g'], D['fold'], D['subs'], D['grp_of']
    print('[check] fold == depositioned_aperiodic_emg_oof.npz fold', flush=True)
    run_folds = [0] if args.smoke else list(range(5)); n_null = 4 if args.smoke else 50
    nb = 200 if args.smoke else 2000
    sidx = {s: j for j, s in enumerate(subs)}
    print(f'[start] rows={len(y)} subjects={len(subs)} folds={run_folds} n_null={n_null} {el()}', flush=True)
    res = {'meta': {'C': C_REG, 'folds': run_folds, 'n_null': n_null, 'smoke': args.smoke,
                    'random_cos_baseline_sd': 1 / np.sqrt(889), 'random_cos_baseline_abs_mean': np.sqrt(2 / (np.pi * 889)),
                    'leaky': ['own_in', 'own_cv', 'auc_xs_oriented', 'cos_*'],
                    'note_groups': 'low group defined post-hoc from pipeline performance -> group split is circular'},
           'checks': {}, 'subjects': {}, 'groups': {}}
    for k in run_folds:
        tk = time.perf_counter()
        tr = np.flatnonzero(fold != k); te_subs = [s for s in subs if D['fold_of'][s] == k]
        Z = zz_fold(D, tr)
        with threadpool_limits(limits=4):
            mx = lr().fit(Z[tr], y[tr])
        w_xs, b_xs = mx.coef_[0], float(mx.intercept_[0])
        # check: test-label shuffle leaves legal xs scores unchanged
        te = np.flatnonzero(fold == k); y2 = y.copy(); y2[te] = np.random.default_rng(k).permutation(y2[te])
        with threadpool_limits(limits=4):
            m2 = lr().fit(Z[tr], y2[tr])
        assert np.array_equal(mx.decision_function(Z[te]), m2.decision_function(Z[te])), 'label shuffle'
        res['checks'][f'fold{k}_test_label_shuffle'] = 'ok'
        out = Parallel(n_jobs=N_JOBS, backend='loky')(
            delayed(subject_job)(Z[g == s], y[g == s], w_xs, b_xs, n_null, [k, sidx[s]]) for s in te_subs)
        for s, r in zip(te_subs, out):
            if s == te_subs[0]:
                r2 = subject_job(Z[g == s], y[g == s], w_xs, b_xs, 0, [k, sidx[s]])
                assert np.array_equal(r['_own_in'], r2['_own_in']) and np.array_equal(r['_own_cv'], r2['_own_cv'])
                assert np.allclose(r['_xs'], mx.decision_function(Z[g == s]), atol=1e-9)
                res['checks'][f'fold{k}_determinism_{s}'] = 'ok'
            res['subjects'][s] = {'group': grp_of[s], 'fold': k, **{q: v for q, v in r.items() if not q.startswith('_')}}
        print(f'[fold {k + 1}/{len(run_folds) if args.smoke else 5}] subjects={len(te_subs)} '
              f'fold_s={time.perf_counter() - tk:.1f} {el()}', flush=True)
        for s in te_subs:
            q = res['subjects'][s]
            print(f'   {s} {q["group"]:4s} AUC xs={q["auc_xs"]:.3f} own_cv={q["auc_own_cv"]:.3f} own_in={q["auc_own_in"]:.3f}'
                  f' rho(xs,own_cv)={q["spearman_xs_own_cv"]:+.3f} cos={q["cos_ws_wxs"]:+.3f} '
                  f'null|cos|={q["cos_null_absmean"]:.3f}', flush=True)
    rng = np.random.default_rng(0)
    keys = ('auc_xs', 'auc_xs_oriented', 'auc_own_in', 'auc_own_cv', 'spearman_xs_own_in', 'spearman_xs_own_cv',
            'cos_ws_wxs', 'cos_null_absmean')
    for G in ('all',) + GROUPS:
        ss = [s for s in res['subjects'] if G == 'all' or res['subjects'][s]['group'] == G]
        if not ss: continue
        res['groups'][G] = {kx: summarize([res['subjects'][s][kx] for s in ss], rng, nb) for kx in keys}
        res['groups'][G]['frac_auc_xs_below_.5'] = float(np.mean([res['subjects'][s]['auc_xs'] < .5 for s in ss]))
        res['groups'][G]['frac_spearman_cv_negative'] = float(np.mean([res['subjects'][s]['spearman_xs_own_cv'] < 0 for s in ss]))
        res['groups'][G]['frac_cos_p_lt_.05'] = float(np.mean([res['subjects'][s]['cos_null_p_abs'] < .05 for s in ss]))
        q = res['groups'][G]
        print(f'[group {G}] n={len(ss)} ' + ' '.join(f'{kx}={q[kx]["mean"]:+.3f}[{q[kx]["ci95"][0]:+.3f},{q[kx]["ci95"][1]:+.3f}]'
                                                  for kx in keys) +
              f' frac(AUCxs<.5)={q["frac_auc_xs_below_.5"]:.2f} frac(rho_cv<0)={q["frac_spearman_cv_negative"]:.2f}'
              f' frac(cos p<.05)={q["frac_cos_p_lt_.05"]:.2f}', flush=True)
    res['seconds'] = time.perf_counter() - __import__('common').T0
    save(HERE, stem, res); print(f'[done] {el()} -> {stem}.json', flush=True)
    sys.stdout = tee.stream; tee.f.close()


if __name__ == '__main__':
    main()
