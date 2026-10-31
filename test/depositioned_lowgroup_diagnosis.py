"""Diagnosis only: which feature families carry within-subject separability of low-group subjects, and do they transfer?

Data / preprocessing as depositioned_subject_diagnosis.py: load_db -> Fc (858 features), global median impute + global
StandardScaler -> Z; within-subject analyses additionally StandardScaler per subject. Groups (high/mid/low) are read from
depositioned_subject_diagnosis.json (fusionM_prank mean bACC; >=.8 / [.6,.8) / <.6).
Feature families from manifest feature_names (12 families, see FAMILY_RULES).
1. Per subject x family: within-subject AUC = LR(C=.05, max_iter=2000), StratifiedKFold(5, shuffle, rs0), family only;
   leave-one-family-out AUC (all minus family); full AUC; HGB(200, depth3, lr .06) full AUC as control.
2. Cohen's d per feature (pooled sd, on Z); top-50 |d| Jaccard (between group-mean d, within-group pairwise median);
   corr of group-mean d vectors, overall and per family.
3. Transfer: direction consistency per family = corr(d_s[family], mean d of other subjects[family]);
   cross-subject AUC: StratifiedGroupKFold(5, shuffle, rs0) by subject, LR(C=.05) on global Z (family only), per test
   subject AUC. Also the same with per-subject z-scored inputs (subject-standardized) to separate offset from direction.
4. Low-group subjects with within AUC >= .85: top-10 |d| features, sign, family, sign vs. mean d of other subjects.
No hyperparameter is tuned on results.
"""
import argparse, json, re, sys, time
from itertools import combinations
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from depositioned_history_delta import load_db  # noqa: E402
from switch_dual_memory import Tee  # noqa: E402

T0 = time.perf_counter()
TOPK, HI_WITHIN = 50, .85
FAMILY_RULES = [  # (family, regex) first match wins
    ('stft_abs_band', r'^stft_log_power_'), ('stft_rel_band', r'^stft_relative_'),
    ('stft_ms_log', r'^stft_\d\.\d_log_'), ('stft_ms_rel', r'^stft_\d\.\d_relative_'),
    ('dwt_energy', r'^dwt_(log_energy|relative|entropy|log_ratio)_'), ('dwt_stats', r'^dwt_(db4|sym5|coif2)_'),
    ('hilbert_env', r'^hilbert_envelope_'), ('hilbert_freq', r'^hilbert_frequency_'),
    ('hjorth', r'^(filtered|raw)_(activity_normalized|mobility|complexity)$'),
    ('entropy_zc', r'^(filtered|raw)_(sample_entropy|permutation_entropy|zero_crossing)$'),
    ('autocorr', r'^(filtered|raw)_ac\d+$'),
    ('time_qc', r'^(amplitude_max|peak_to_peak|rms|flat_fraction|diff_energy|spike_fraction|low_freq_drift|'
                r'high_freq_ratio|nan_fraction)$')]


def el():
    return f'elapsed={time.perf_counter() - T0:.1f}s'


def families(names):
    fam = {}
    for i, n in enumerate(names):
        f = next((k for k, r in FAMILY_RULES if re.match(r, n)), 'other')
        fam.setdefault(f, []).append(i)
    return {k: np.array(v) for k, v in fam.items()}


def lr():
    return LogisticRegression(C=.05, max_iter=2000)


def cv_auc(X, y, model):
    q = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        m = model().fit(X[tr], y[tr])
        q[te] = m.decision_function(X[te]) if hasattr(m, 'decision_function') else m.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, q))


def within_subject(zs, ys, fam, hgb):
    allc = np.arange(zs.shape[1]); r = {'full': cv_auc(zs, ys, lr), 'fam': {}, 'lofo': {}}
    for f, idx in fam.items():
        r['fam'][f] = cv_auc(zs[:, idx], ys, lr)
        r['lofo'][f] = cv_auc(zs[:, np.setdiff1d(allc, idx)], ys, lr)
    if hgb:
        r['hgb_full'] = cv_auc(zs, ys, lambda: HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=.06,
                                                                               random_state=0))
    return r


def jacc(a, b):
    a, b = set(a), set(b); return len(a & b) / len(a | b)


def top(d, k=TOPK):
    return np.argsort(-np.abs(d))[:k]


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float('nan')


def med(v):
    v = np.asarray(v, float); return float(np.nanmedian(v)) if v.size else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); args = ap.parse_args()
    stem = 'depositioned_lowgroup_diagnosis' + ('_smoke' if args.smoke else '')
    tee = Tee(sys.stdout, OUT / (stem + '.log')); sys.stdout = tee
    man, subs, data = load_db(); names = man['feature_names']; assert len(names) == 858
    prev = json.loads((OUT / 'depositioned_subject_diagnosis.json').read_text(encoding='utf-8'))['diagnosis']
    grp_of = {s: G for G, v in prev['groups'].items() for s in v['subjects']}
    prev_within = {r['subject']: r['within_AUC'] for r in prev['subjects']}
    if args.smoke:
        subs = [next(s for s in subs if grp_of[s] == G) for G in ('high', 'mid', 'low')] + \
               [s for s in subs if grp_of[s] == 'low'][1:4] + [s for s in subs if grp_of[s] == 'mid'][1:4]
    Fc = np.vstack([data[s][0] for s in subs]); y = np.concatenate([data[s][1] for s in subs])
    g = np.concatenate([np.full(len(data[s][1]), s) for s in subs])
    fam = families(names); FN = list(fam)
    print(f'[start] rows={len(y)} subjects={len(subs)} smoke={args.smoke}', flush=True)
    print('[families] ' + ' '.join(f'{f}={len(fam[f])}' for f in FN) + f' total={sum(len(v) for v in fam.values())}', flush=True)
    res = {'meta': {'smoke': args.smoke, 'topk': TOPK, 'hi_within': HI_WITHIN,
                    'families': {f: {'dim': len(fam[f]), 'names': [names[i] for i in fam[f]]} for f in FN}}}
    md = np.nanmedian(np.where(np.isfinite(Fc), Fc, np.nan), 0); md = np.where(np.isfinite(md), md, 0.)
    Z = StandardScaler().fit_transform(np.where(np.isfinite(Fc), Fc, md))
    ZS = np.zeros_like(Z); D = {}
    for s in subs:
        m = g == s; ZS[m] = StandardScaler().fit_transform(Z[m]); a, b = Z[m & (y == 0)], Z[m & (y == 1)]
        sd = np.sqrt(((len(a) - 1) * a.var(0, ddof=1) + (len(b) - 1) * b.var(0, ddof=1)) / (len(a) + len(b) - 2))
        D[s] = np.where(sd > 1e-9, (b.mean(0) - a.mean(0)) / np.where(sd > 1e-9, sd, 1.), 0.)
    Dm = np.array([D[s] for s in subs])

    # 1. within-subject per family / LOFO / full / HGB
    with threadpool_limits(limits=1):
        jobs = Parallel(n_jobs=8, backend='loky', return_as='generator')(
            delayed(within_subject)(ZS[g == s], y[g == s], fam, True) for s in subs)
        W = {}
        for j, (s, r) in enumerate(zip(subs, jobs)):
            W[s] = r; ch = abs(r['full'] - prev_within[s])
            best = max(r['fam'], key=r['fam'].get)
            print(f'[within {j + 1:2d}/{len(subs)}] {s} grp={grp_of[s]:4s} full={r["full"]:.3f} (prev {prev_within[s]:.3f}) '
                  f'hgb={r["hgb_full"]:.3f} best_fam={best}:{r["fam"][best]:.3f} {el()}', flush=True)
            assert ch < 1e-6, (s, ch)

    # 3a. cross-subject SGKF transfer, global Z and subject-z
    cross = {k: {s: {} for s in subs} for k in ('globalZ', 'subjZ')}
    sets = [('full', np.arange(858))] + [(f, fam[f]) for f in FN]
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Z, y, g))
    with threadpool_limits(limits=8):
        for key, X in (('globalZ', Z), ('subjZ', ZS)):
            for f, idx in sets:
                q = np.zeros(len(y))
                for tr, te in folds:
                    q[te] = lr().fit(X[tr][:, idx], y[tr]).decision_function(X[te][:, idx])
                for s in subs:
                    m = g == s; cross[key][s][f] = float(roc_auc_score(y[m], q[m]))
                print(f'[cross {key}] {f:14s} median AUC={med([cross[key][s][f] for s in subs]):.3f} {el()}', flush=True)

    # 3b. direction consistency per family
    dirc = {s: {} for s in subs}
    for j, s in enumerate(subs):
        oth = np.delete(Dm, j, 0).mean(0); dirc[s]['full'] = corr(D[s], oth)
        for f in FN: dirc[s][f] = corr(D[s][fam[f]], oth[fam[f]])

    GR = [G for G in ('high', 'mid', 'low') if any(grp_of[s] == G for s in subs)]
    gs = {G: [s for s in subs if grp_of[s] == G] for G in GR}
    tab = {}
    print('\n[table] per group medians: within(fam only) | drop=full-LOFO | cross globalZ | cross subjZ | dir_consistency', flush=True)
    for G in GR:
        S = gs[G]; tab[G] = {'n': len(S), 'within_full': med([W[s]['full'] for s in S]),
                             'within_hgb_full': med([W[s]['hgb_full'] for s in S]),
                             'cross_full_globalZ': med([cross['globalZ'][s]['full'] for s in S]),
                             'cross_full_subjZ': med([cross['subjZ'][s]['full'] for s in S]),
                             'dir_full': med([dirc[s]['full'] for s in S]), 'family': {}}
        print(f'== {G} n={len(S)} within_full={tab[G]["within_full"]:.3f} hgb={tab[G]["within_hgb_full"]:.3f} '
              f'cross_gZ={tab[G]["cross_full_globalZ"]:.3f} cross_sZ={tab[G]["cross_full_subjZ"]:.3f} dir={tab[G]["dir_full"]:+.3f}', flush=True)
        for f in FN:
            t = tab[G]['family'][f] = {
                'within': med([W[s]['fam'][f] for s in S]), 'drop': med([W[s]['full'] - W[s]['lofo'][f] for s in S]),
                'cross_globalZ': med([cross['globalZ'][s][f] for s in S]), 'cross_subjZ': med([cross['subjZ'][s][f] for s in S]),
                'dir': med([dirc[s][f] for s in S]),
                'n_best_family': int(sum(max(W[s]['fam'], key=W[s]['fam'].get) == f for s in S))}
            print(f'  {f:14s} dim={len(fam[f]):3d} within={t["within"]:.3f} drop={t["drop"]:+.4f} cross_gZ={t["cross_globalZ"]:.3f} '
                  f'cross_sZ={t["cross_subjZ"]:.3f} dir={t["dir"]:+.3f} n_best={t["n_best_family"]}', flush=True)

    # 2. Cohen's d overlaps / correlations
    gmean = {G: Dm[[subs.index(s) for s in gs[G]]].mean(0) for G in GR}
    ov = {'top50_jaccard_groupmean': {}, 'corr_groupmean_d': {}, 'corr_groupmean_d_by_family': {},
          'top50_jaccard_within_group_pairwise_median': {}, 'top50_family_share_groupmean': {}}
    for A, B in combinations(GR, 2):
        k = f'{A}_vs_{B}'
        ov['top50_jaccard_groupmean'][k] = jacc(top(gmean[A]), top(gmean[B]))
        ov['corr_groupmean_d'][k] = corr(gmean[A], gmean[B])
        ov['corr_groupmean_d_by_family'][k] = {f: corr(gmean[A][fam[f]], gmean[B][fam[f]]) for f in FN}
    for G in GR:
        ov['top50_jaccard_within_group_pairwise_median'][G] = med([jacc(top(D[a]), top(D[b])) for a, b in combinations(gs[G], 2)]) \
            if len(gs[G]) > 1 else None
        fi = {i: f for f in FN for i in fam[f]}; tp = top(gmean[G])
        ov['top50_family_share_groupmean'][G] = {f: int(sum(fi[i] == f for i in tp)) for f in FN}
    ov['top50_jaccard_cross_group_pairwise_median'] = {f'{A}_vs_{B}': med([jacc(top(D[a]), top(D[b])) for a in gs[A] for b in gs[B]])
                                                      for A, B in combinations(GR, 2)}
    print('\n[cohen d] ' + json.dumps({k: v for k, v in ov.items() if k != 'corr_groupmean_d_by_family'}, default=float), flush=True)
    for k, v in ov['corr_groupmean_d_by_family'].items():
        print(f'[cohen d corr by family] {k}: ' + ' '.join(f'{f}={c:+.2f}' for f, c in v.items()), flush=True)

    # 4. low-group subjects with high within AUC
    fi = {i: f for f in FN for i in fam[f]}; hl = {}
    for s in gs.get('low', []):
        if W[s]['full'] < HI_WITHIN: continue
        j = subs.index(s); oth = np.delete(Dm, j, 0).mean(0); tp = top(D[s], 10)
        feats = [{'name': names[i], 'family': fi[i], 'd': float(D[s][i]), 'mean_d_others': float(oth[i]),
                  'opposite_sign': bool(np.sign(D[s][i]) != np.sign(oth[i]))} for i in tp]
        fams = {}
        for x in feats: fams[x['family']] = fams.get(x['family'], 0) + 1
        hl[s] = {'within_full': W[s]['full'], 'cross_full_globalZ': cross['globalZ'][s]['full'], 'dir_full': dirc[s]['full'],
                 'top10': feats, 'top10_family_counts': fams, 'n_opposite_sign_top10': int(sum(x['opposite_sign'] for x in feats)),
                 'best_family': max(W[s]['fam'], key=W[s]['fam'].get), 'fam_within': W[s]['fam']}
        print(f'[low-hi] {s} within={W[s]["full"]:.3f} cross={cross["globalZ"][s]["full"]:.3f} dir={dirc[s]["full"]:+.2f} '
              f'opp_sign={hl[s]["n_opposite_sign_top10"]}/10 fams={fams}', flush=True)
        for x in feats:
            print(f'    {x["name"]:32s} {x["family"]:14s} d={x["d"]:+.2f} others={x["mean_d_others"]:+.2f}'
                  f'{" OPP" if x["opposite_sign"] else ""}', flush=True)

    # 5. summary
    summ = {}
    if 'low' in tab:
        L = tab['low']['family']; rank = sorted(FN, key=lambda f: -L[f]['within'])
        summ['low_top_within_families'] = [(f, L[f]['within'], L[f]['cross_globalZ'], L[f]['dir']) for f in rank[:4]]
        summ['low_untransferable'] = [f for f in FN if L[f]['within'] >= .7 and (L[f]['cross_globalZ'] < .6 or L[f]['dir'] < .2)]
        summ['low_largest_lofo_drop'] = sorted(FN, key=lambda f: -L[f]['drop'])[:3]
        allm = {f: med([cross['globalZ'][s][f] for s in subs]) for f in FN}
        summ['most_transferable_families_all'] = sorted(FN, key=lambda f: -allm[f])[:4]
        summ['low_n_hi_within'] = len(hl)
        summ['low_hi_within_frac_opposite_sign_top10'] = med([v['n_opposite_sign_top10'] / 10 for v in hl.values()]) if hl else None
        summ['subjZ_minus_globalZ_cross_full_low'] = tab['low']['cross_full_subjZ'] - tab['low']['cross_full_globalZ']
        print('\n[summary]', flush=True)
        print(f'  low group: within_full={tab["low"]["within_full"]:.3f} vs cross_full={tab["low"]["cross_full_globalZ"]:.3f} '
              f'(subjZ {tab["low"]["cross_full_subjZ"]:.3f}); dir_full={tab["low"]["dir_full"]:+.3f}', flush=True)
        print('  low: top within families (within, cross_gZ, dir): ' +
              '; '.join(f'{f} {a:.3f}/{b:.3f}/{c:+.2f}' for f, a, b, c in summ['low_top_within_families']), flush=True)
        print(f'  low: within>=.7 but cross<.6 or dir<.2: {summ["low_untransferable"]}', flush=True)
        print(f'  low: largest LOFO drop: {summ["low_largest_lofo_drop"]} (small drops => redundancy across families)', flush=True)
        print(f'  most transferable families (all subjects, cross_gZ): {summ["most_transferable_families_all"]}', flush=True)
        print(f'  low hi-within subjects n={len(hl)}, median frac top10 opposite sign={summ["low_hi_within_frac_opposite_sign_top10"]}', flush=True)
        print(f'  subject z-scoring gain on low cross AUC: {summ["subjZ_minus_globalZ_cross_full_low"]:+.3f}', flush=True)
    res.update({'subjects': {s: {'group': grp_of[s], 'within': W[s], 'cross': {k: cross[k][s] for k in cross}, 'dir': dirc[s]}
                             for s in subs},
                'groups': tab, 'cohen_d': ov, 'low_hi_within': hl, 'summary': summ, 'seconds': time.perf_counter() - T0})
    (OUT / (stem + '.json')).write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding='utf-8')
    print(f'[done] {el()} -> {stem}.json', flush=True)
    sys.stdout = tee.stream; tee.f.close()


if __name__ == '__main__':
    main()
