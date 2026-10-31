"""D1: is the within-subject ceiling (own_cv AUC .823) inflated by temporally adjacent windows?

Per subject s (889 features, median-impute + StandardScaler fitted on the within-subject TRAIN rows only), models
lr = LR(C=.01) and hgb = HGB(200, depth 3, lr .06, l2 1, rs0). Out-of-fold scores under within-subject splits:
  random        StratifiedKFold(5, shuffle, rs0) over s's windows (== audit own_cv split)
  lobo          leave-one-block-out (real recording blocks = trial id, ~6 alternating 5-min relax/task blocks)
  gap{G}        random split, but train rows within G windows (|dt| <= G*6 s, real time) of ANY test row are dropped
  gap{G}_ctrl   random split, train randomly subsampled to the same size as gap{G} (separates size from leakage)
  seg10_gap5    contiguous 10-window segments (real time order) randomly assigned to 5 folds, +-5-window gap
  seg10_gap5_ctrl  same test folds, train subsampled to the seg10_gap5 size without the gap
Folds whose train set lacks a class (or < 5 rows per class) are skipped; coverage reported.
Per subject: AUC of pooled OOF scores, bACC at p>=.5. Groups high/mid/low (low post-hoc -> circular).
Run: python test/theory/own_cv_block.py [--smoke]
"""
import argparse, time

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from _common import GROUPS, bacc, el, gsummary, load, open_log, save, subj_impute_scale

GAPS = (1, 2, 5)
WIN = 6.


def make(name):
    if name == 'hgb':
        return HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=.06, l2_regularization=1.,
                                              random_state=0)
    return LogisticRegression(C=.01, max_iter=2000)


def splits(y, t, block, seed):
    """-> {scheme: list of (train_idx, test_idx)}; all indices into the subject's canonical rows."""
    rng = np.random.default_rng(seed); n = len(y); S = {}
    rnd = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(n), y))
    S['random'] = rnd
    S['lobo'] = [(np.flatnonzero(block != b), np.flatnonzero(block == b)) for b in np.unique(block)]

    def gapped(tr, te, G):
        d = np.abs(t[tr][:, None] - t[te][None, :]).min(1)
        return tr[d > G * WIN + 1e-3]
    for G in GAPS:
        S[f'gap{G}'] = [(gapped(tr, te, G), te) for tr, te in rnd]
        S[f'gap{G}_ctrl'] = [(np.sort(rng.choice(tr, len(gt), replace=False)), te)
                             for (tr, te), (gt, _) in zip(rnd, S[f'gap{G}'])]
    o = np.argsort(t, kind='stable'); seg = np.empty(n, int); seg[o] = np.arange(n) // 10
    fold_of_seg = rng.permutation(seg.max() + 1) % 5
    sf = fold_of_seg[seg]
    base = [(np.flatnonzero(sf != k), np.flatnonzero(sf == k)) for k in range(5)]
    S['seg10_gap5'] = [(gapped(tr, te, 5), te) for tr, te in base]
    S['seg10_gap5_ctrl'] = [(np.sort(rng.choice(tr, len(gt), replace=False)), te)
                            for (tr, te), (gt, _) in zip(base, S['seg10_gap5'])]
    return S


def run_subject(F, y, t, block, seed, models):
    out = {}
    with threadpool_limits(limits=1):
        for scheme, sp in splits(y, t, block, seed).items():
            ntr = [len(tr) for tr, _ in sp]
            for m in models:
                p = np.full(len(y), np.nan)
                for tr, te in sp:
                    if len(te) == 0 or min(np.sum(y[tr] == 0), np.sum(y[tr] == 1)) < 5: continue
                    Z = subj_impute_scale(F, tr)
                    p[te] = make(m).fit(Z[tr], y[tr]).predict_proba(Z[te])[:, 1]
                ok = np.isfinite(p)
                r = {'coverage': float(ok.mean()), 'mean_train_n': float(np.mean(ntr))}
                if ok.sum() and np.unique(y[ok]).size == 2:
                    r['auc'] = float(roc_auc_score(y[ok], p[ok])); r['bacc'] = float(bacc(y[ok], p[ok]))
                else:
                    r['auc'] = r['bacc'] = None
                out[f'{scheme}__{m}'] = r
                out[f'_p_{scheme}__{m}'] = p
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); args = ap.parse_args()
    stem = 'own_cv_block' + ('_smoke' if args.smoke else ''); tee = open_log(stem)
    D = load(); y, g, subs, grp_of = D['y'], D['g'], D['subs'], D['grp_of']
    run_subs = subs if not args.smoke else [next(s for s in subs if grp_of[s] == G) for G in GROUPS]
    models = ('lr', 'hgb'); sidx = {s: j for j, s in enumerate(subs)}
    print(f'[start] subjects={len(run_subs)} models={models} gaps={GAPS} {el()}', flush=True)
    res = {'meta': {'models': {'lr': 'LR C=.01, within-train standardisation', 'hgb': 'HGB 200/3/.06/l2=1'},
                    'schemes': ['random', 'lobo'] + [f'gap{G}{c}' for G in GAPS for c in ('', '_ctrl')] +
                               ['seg10_gap5', 'seg10_gap5_ctrl'], 'win_s': WIN, 'smoke': args.smoke,
                    'leaky': 'all schemes use the subject\'s own labels (ceiling analysis, not a deployable protocol)',
                    'note_groups': 'low group defined post-hoc from pipeline performance -> circular'},
           'checks': {}, 'subjects': {}, 'groups': {}}
    tk = time.perf_counter(); job = lambda s: run_subject(D['F'][g == s], y[g == s], D['t'][g == s],
                                                           D['block'][g == s], [7, sidx[s]], models)
    outs = []
    with Parallel(n_jobs=14, backend='loky', return_as='generator') as par:
        for i, (s, r) in enumerate(zip(run_subs, par(delayed(run_subject)(
                D['F'][g == s], y[g == s], D['t'][g == s], D['block'][g == s], [7, sidx[s]], models)
                for s in run_subs)), 1):
            outs.append((s, r))
            fa = lambda v: 'NA' if v is None else f'{v:.3f}'
            print(f'[subject {i}/{len(run_subs)}] {s} {grp_of[s]:4s} n={int((g == s).sum())} blocks='
                  f'{np.unique(D["block"][g == s]).size} ' +
                  ' '.join(f'{k.replace("__", ":")}={r[k]["auc"]:.3f}' for k in r
                           if not k.startswith('_') and k.endswith('lr') and r[k]['auc'] is not None) +
                  f' | hgb random={fa(r["random__hgb"]["auc"])} lobo={fa(r["lobo__hgb"]["auc"])} {el()}', flush=True)
    s0, r0 = outs[0]; r0b = job(s0)
    assert all(np.array_equal(r0[k], r0b[k], equal_nan=True) for k in r0 if k.startswith('_p_')), 'determinism'
    res['checks']['determinism_first_subject'] = 'ok'
    keys = []
    for s, r in outs:
        res['subjects'][s] = {'group': grp_of[s], **{k: v for k, v in r.items() if not k.startswith('_')}}
        flat = {}
        for k, v in r.items():
            if k.startswith('_'): continue
            for q in ('auc', 'bacc', 'coverage', 'mean_train_n'): flat[f'{k}__{q}'] = v[q]
        res['subjects'][s]['_flat'] = flat; keys = list(flat)
    per = {s: res['subjects'][s].pop('_flat') for s in res['subjects']}
    res['groups'] = gsummary(per, grp_of, keys)
    for G, q in res['groups'].items():
        print(f'[group {G}] n={q["random__lr__auc"]["n"]}', flush=True)
        for sch in res['meta']['schemes']:
            print(f'   {sch:16s} ' + ' '.join(
                f'{m}: AUC mean={fa(q[f"{sch}__{m}__auc"]["mean"])} med={fa(q[f"{sch}__{m}__auc"]["median"])} '
                f'bACC={fa(q[f"{sch}__{m}__bacc"]["mean"])} n={q[f"{sch}__{m}__auc"]["n"]}' for m in models) +
                f' train_n={q[f"{sch}__lr__mean_train_n"]["mean"]:.0f} cov={q[f"{sch}__lr__coverage"]["mean"]:.2f}',
                flush=True)
    res['meta']['run_seconds'] = time.perf_counter() - tk
    save(stem, res, tee)


if __name__ == '__main__':
    main()
