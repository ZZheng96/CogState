"""D2: which feature families drive each subject's own discriminative direction w_s, and do they disagree with w_xs?

Per outer fold k (SGKF(5, shuffle, rs0) == npz fold): Z = 889 features, median-impute + StandardScaler fitted on fold-train
rows (same space as audit/direction_scoring). w_xs = LR(C=.01) on all fold-train rows (other subjects). For every held-out
subject s: w_s = LR(C=.01) on all of s's windows in the same Z space (LEAKY ceiling, uses s's labels).
Families: _common.family (name-based; dims listed in results). Per subject and family f:
  energy_share   sum(w_s[f]^2) / sum(w_s^2);  enrichment = energy_share / (dims_f / 889)
  energy_share_xs  same for w_xs
  cos_f          cos(w_s[f], w_xs[f])
  auc_xs_f       AUC on s of the family-restricted cross-subject score Z[:, f] @ w_xs[f]  (< .5 = reversed on s)
  auc_own_f_{random,lobo,seg10_gap5}  within-subject AUC of LR(C=.01) trained on family f only, own_cv_block splits,
                 within-train standardisation
Artifact families: emg_hf, delta_ocular. Groups high/mid/low (low post-hoc -> circular).
Run: python test/theory/own_direction_loading.py [--smoke]   (smoke: fold 0, 6 subjects)
"""
import argparse, time

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from _common import ARTIFACT, FAMILIES, GROUPS, el, families, gsummary, load, open_log, save, subj_impute_scale
from _common import impute_scale
from own_cv_block import splits

SCHEMES = ('random', 'lobo', 'seg10_gap5')


def lr():
    return LogisticRegression(C=.01, max_iter=2000)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def subject_job(Zs, Fs, ys, t, block, w_xs, fam, fams, seed):
    r = {}
    with threadpool_limits(limits=1):
        w_s = lr().fit(Zs, ys).coef_[0]
        E, Ex = (w_s ** 2).sum(), (w_xs ** 2).sum()
        r['cos_all'] = cos(w_s, w_xs); r['auc_xs_all'] = float(roc_auc_score(ys, Zs @ w_xs))
        sp = {k: v for k, v in splits(ys, t, block, seed).items() if k in SCHEMES}
        for f in fams:
            m = fam == f
            r[f'{f}__energy_share'] = float((w_s[m] ** 2).sum() / E)
            r[f'{f}__enrichment'] = r[f'{f}__energy_share'] / (m.sum() / len(fam))
            r[f'{f}__energy_share_xs'] = float((w_xs[m] ** 2).sum() / Ex)
            r[f'{f}__cos'] = cos(w_s[m], w_xs[m])
            r[f'{f}__auc_xs'] = float(roc_auc_score(ys, Zs[:, m] @ w_xs[m]))
            for sch in SCHEMES:
                p = np.full(len(ys), np.nan)
                for tr, te in sp[sch]:
                    if min(np.sum(ys[tr] == 0), np.sum(ys[tr] == 1)) < 5: continue
                    A = subj_impute_scale(Fs[:, m], tr)
                    p[te] = lr().fit(A[tr], ys[tr]).decision_function(A[te])
                ok = np.isfinite(p)
                r[f'{f}__auc_own_{sch}'] = float(roc_auc_score(ys[ok], p[ok])) if np.unique(ys[ok]).size == 2 else None
        a = np.isin(fam, ARTIFACT)
        r['artifact__energy_share'] = float((w_s[a] ** 2).sum() / E)
        r['artifact__energy_share_xs'] = float((w_xs[a] ** 2).sum() / Ex)
        r['artifact__cos'] = cos(w_s[a], w_xs[a]); r['nonartifact__cos'] = cos(w_s[~a], w_xs[~a])
        r['artifact__auc_xs'] = float(roc_auc_score(ys, Zs[:, a] @ w_xs[a]))
        r['nonartifact__auc_xs'] = float(roc_auc_score(ys, Zs[:, ~a] @ w_xs[~a]))
    return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); args = ap.parse_args()
    stem = 'own_direction_loading' + ('_smoke' if args.smoke else ''); tee = open_log(stem)
    D = load(); y, g, fold, subs, grp_of = D['y'], D['g'], D['fold'], D['subs'], D['grp_of']
    fam, dims = families(D['names']); fams = [f for f in FAMILIES if dims[f] > 0]
    print(f'[families] dims={dims}', flush=True)
    sidx = {s: j for j, s in enumerate(subs)}; run_folds = [0] if args.smoke else list(range(5))
    res = {'meta': {'C': .01, 'folds': run_folds, 'smoke': args.smoke, 'family_dims': dims,
                    'artifact_families': list(ARTIFACT), 'artifact_dim_share': float(np.isin(fam, ARTIFACT).mean()),
                    'family_rules': '_common.family; BANDS 0..4 = delta(->delta_ocular) theta alpha beta gamma(->emg_hf); '
                                    'DWT A5 D5 D4 D3 D2/D1; quality amplitude/ptp/drift/spike -> delta_ocular, '
                                    'high_freq_ratio/diff_energy/raw Hjorth/zero-crossing/ac1/hf_* -> emg_hf',
                    'leaky': 'w_s and auc_own_* use the subject\'s labels (ceiling analysis)',
                    'note_groups': 'low group defined post-hoc -> circular'},
           'feature_family': dict(zip(D['names'], fam.tolist())), 'checks': {}, 'subjects': {}, 'groups': {},
           'w_xs_energy_share_by_fold': {}}
    for k in run_folds:
        tk = time.perf_counter()
        tr = np.flatnonzero(fold != k); te_subs = [s for s in subs if D['fold_of'][s] == k]
        if args.smoke: te_subs = te_subs[:6]
        tfo, tfn = impute_scale(D['Fc'], tr), impute_scale(D['Fn'], tr); Z = np.hstack([tfo(D['Fc']), tfn(D['Fn'])])
        with threadpool_limits(limits=8):
            w_xs = lr().fit(Z[tr], y[tr]).coef_[0]
            te = np.flatnonzero(fold == k); y2 = y.copy(); y2[te] = np.random.default_rng(k).permutation(y2[te])
            assert np.array_equal(w_xs, lr().fit(Z[tr], y2[tr]).coef_[0]), 'test-label shuffle changed w_xs'
        res['checks'][f'fold{k}_w_xs_independent_of_test_labels'] = 'ok'
        res['w_xs_energy_share_by_fold'][k] = {f: float((w_xs[fam == f] ** 2).sum() / (w_xs ** 2).sum()) for f in fams}
        out = Parallel(n_jobs=14, backend='loky')(
            delayed(subject_job)(Z[g == s], D['F'][g == s], y[g == s], D['t'][g == s], D['block'][g == s], w_xs, fam,
                                 fams, [7, sidx[s]]) for s in te_subs)
        for s, r in zip(te_subs, out):
            res['subjects'][s] = {'group': grp_of[s], 'fold': k, **r}
            print(f'   {s} {grp_of[s]:4s} cos={r["cos_all"]:+.3f} aucxs={r["auc_xs_all"]:.3f} '
                  f'art_share={r["artifact__energy_share"]:.2f} art_cos={r["artifact__cos"]:+.3f} '
                  f'nonart_cos={r["nonartifact__cos"]:+.3f} | ' +
                  ' '.join(f'{f}:{r[f + "__energy_share"]:.2f}/{r[f + "__cos"]:+.2f}' for f in fams), flush=True)
        print(f'[fold {k + 1}/5] subjects={len(te_subs)} fold_s={time.perf_counter() - tk:.1f} {el()}', flush=True)
    keys = [q for q in next(iter(res['subjects'].values())) if q not in ('group', 'fold')]
    res['groups'] = gsummary(res['subjects'], grp_of, keys)
    for G, q in res['groups'].items():
        print(f'[group {G}] n={q["cos_all"]["n"]} cos_all={q["cos_all"]["mean"]:+.3f} auc_xs={q["auc_xs_all"]["mean"]:.3f} '
              f'artifact share={q["artifact__energy_share"]["mean"]:.3f} (dims {res["meta"]["artifact_dim_share"]:.3f}, '
              f'w_xs {q["artifact__energy_share_xs"]["mean"]:.3f}) art_cos={q["artifact__cos"]["mean"]:+.3f} '
              f'nonart_cos={q["nonartifact__cos"]["mean"]:+.3f} art_aucxs={q["artifact__auc_xs"]["mean"]:.3f} '
              f'nonart_aucxs={q["nonartifact__auc_xs"]["mean"]:.3f}', flush=True)
        for f in fams:
            print(f'   {f:15s} dims={dims[f]:3d} share={q[f + "__energy_share"]["mean"]:.3f} '
                  f'enrich={q[f + "__enrichment"]["mean"]:.2f} share_xs={q[f + "__energy_share_xs"]["mean"]:.3f} '
                  f'cos={q[f + "__cos"]["mean"]:+.3f} auc_xs_f={q[f + "__auc_xs"]["mean"]:.3f} ' +
                  ' '.join(f'own_{sch}={q[f"{f}__auc_own_{sch}"]["mean"]:.3f}' for sch in SCHEMES), flush=True)
    save(stem, res, tee)


if __name__ == '__main__':
    main()
