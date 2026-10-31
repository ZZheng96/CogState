"""Score-level merge of the two best legal pipelines (strict causal, 6 s window, test labels never used).

Pipeline 1 = depositioned_within_subject_adapt_v2.py, best legal variant pseudo_lr_full:
  per stream sd, L = mean(logit(A['stream{sd}__all_prank']), logit(B['stream{sd}__resnet_whiten_logit_delta']),
  logit(B['stream{sd}__resnet_fbank_logit_delta'])); per test subject in stream order, refit every R=16 windows starting at
  t0=48 on rows [0, t0) only; representation = StandardScaler(past ZZ of 889 dims, fold-train imputation); pseudo labels =
  top / bottom 30% of past L; LR(C=.01, max_iter 1000) -> z = clip((s - mean(past s)) / std(past s), -3, 3); f1 = L + z
  (z = 0 where undefined). Re-derived here; only the pseudo_lr_full branch is refit (no PCA / other variants), which does
  not change its numbers - asserted against the v2 json.

Pipeline 2 = 4.30 merge_best_window6.py, combo 'C4_bb_resnet_eq' (combo C four members + broadband LR + ResNet1d,
equal-weight z-logit with per-fold z from the training fold's inner OOF): per-window deployable, no test history.

Alignment (asserted, never guessed): the depositioned database windows are mapped 1:1 onto the highdim rows of
merge_best_window6_oof.npz by multiset (subject, label, md5(raw float32)); all 16067 windows matched, the 81 identical
saturated windows are consumed in file order (same procedure as merge_best_window6.py). Then per row: subject id, label,
window count per subject and fold (pipeline 2 fold_id == pipeline 1 fold + 1) must agree, and every mapped pair must carry
the same label.

FIXED merge rules (pre-registered, nothing tuned on test):
  - normalisation, chosen once and fixed: PRIMARY = m_plain, i.e. NO extra z, the two per-window logits are averaged as
    they are. Reason: both parents are already on a comparable logit scale and both are calibrated between subjects
    (std(L) = .60 so std(f1) ~ 1.1; std(s2) = .74; pipeline 1 inherits L's cross-subject calibration, pipeline 2's
    per-fold z is anchored on the training fold's inner OOF), so an equal-weight logit average is already scale-balanced
    to within ~1.5x, while it keeps the between-subject offset information that pooled AUC and subject-macro bACC are
    computed on. The alternative m_z re-centres every subject on its own running mean (causal past-window z: expanding
    mean/std over strictly earlier windows of the same subject in the same stream order, clipped +-3, with ZMINZ=16 past
    windows required - the same minimum past as thr_mean256; a row with fewer than ZMINZ past windows, i.e. only the first
    16 windows of a subject, has no causal z for either channel and falls back to m_plain) - that removes exactly the
    between-subject calibration both parents rely on; it is kept as the pre-registered alternative and reported on exactly
    the same rows as the parents (no NaN, same n).
  - equal weights .5 / .5, not tuned.
  - threshold: primary = thr_mean256 (depositioned_adaptive_threshold.apply: l_t - mean(l_{t-256..t-1}), no shift with
    fewer than 16 past rows), the strategy pipeline 1 validated; additionally fixed 0.5 on sig(merged) (= 0 on the logit
    scale, pipeline 2's native rule).
Variants (all pre-registered): m_plain (.5*f1 + .5*s2), m_z, m_pseudo_only (= pipeline 1, f1), m_c4_only (= pipeline 2,
s2), m_hwf_c4 (same merge rule applied to L instead of f1 = pipeline 1 without adaptation).
Evaluation (same as the parents): all / mature (t>=120) AUC, subject-macro bACC, micro bACC, #subjects bACC>=.9, max;
paired subject bootstrap (1000, smoke 100) of every merged variant vs BOTH parents (both thresholds); high/mid/low groups
from depositioned_subject_diagnosis.json (reporting only).
Checks (assert): alignment (md5 multiset 1:1, subject id / window count / labels / fold); pipeline 1 reproduces the v2
json (fuse_pseudo_lr_full_thr subject-macro bACC, 1e-9) and pipeline 2 reproduces 4.30 (AUC / bACC@0 from
merge_best_window6_results.json); past-only (perturbing a check subject's features and branch logits after a cut leaves
every merged score before the cut identical, later scores change); test-label shuffle leaves every legal variant
identical; subject isolation (alone == pooled parallel).
Parallel over subjects with joblib (loky, 8 jobs, 1 BLAS thread per job; deterministic).
Run: python test/depositioned_final_merge.py [--smoke]   (smoke: stream0, fold0, bootstrap 100)
"""
import argparse, hashlib, json, sys, time, warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
DB = OUT.parent / 'database'
sys.path.insert(0, str(OUT))
from depositioned_history_delta import arrange, boot, load_db  # noqa: E402
from depositioned_resnet_history_delta import stream_order, logit, sig  # noqa: E402
from depositioned_adaptive_threshold import apply as thr_apply  # noqa: E402
from depositioned_symmetric_contrast import impute_scale  # noqa: E402
from causal_kernel_state_mapping import metrics  # noqa: E402
from switch_dual_memory import Tee  # noqa: E402
from depositioned_within_subject_adapt import pseudo_labels, GROUPS, MATURE, STREAM_SEEDS, THR_K, R_STEP, MIN_PAST  # noqa: E402

CLIP, EPS, N_JOBS, ZMIN, ZMINZ = 3., 1e-6, 8, MIN_PAST, 16
W1, W2 = .5, .5
VARIANTS = ('m_plain', 'm_z', 'm_pseudo_only', 'm_c4_only', 'm_hwf_c4')
MERGED = ('m_plain', 'm_z', 'm_hwf_c4')
PARENTS = ('m_pseudo_only', 'm_c4_only')
P2_KEY = 'combo__C4_bb_resnet_eq'
T0 = time.perf_counter()


def el():
    return f'elapsed={time.perf_counter() - T0:.1f}s'


def zc(sp, sn):
    return np.clip((sn - sp.mean()) / max(sp.std(), EPS), -CLIP, CLIP)


def lr():
    return LogisticRegression(C=.01, max_iter=1000)


def p1_subject(ZZ, Bl, y=None):
    """Pipeline 1 for one subject in stream order -> (L, f1, z).

    ZZ (n, 889) is that subject's test-side zz, Bl (n, 3) the branch logits. y is accepted only so the label-shuffle check
    can demonstrate that this legal variant never reads test labels.
    """
    with threadpool_limits(limits=1):
        n = len(Bl); L = Bl.mean(1); z = np.full(n, np.nan)
        for t0 in range(ZMIN, n, R_STEP):
            t1 = min(n, t0 + R_STEP)
            sc = StandardScaler().fit(ZZ[:t0]); xp, xn = sc.transform(ZZ[:t0]), sc.transform(ZZ[t0:t1])
            i, yl = pseudo_labels(L[:t0])           # unlabeled pseudo labels from past L only
            m = lr().fit(xp[i], yl)
            z[t0:t1] = zc(m.decision_function(xp), m.decision_function(xn))
    return L, L + np.where(np.isfinite(z), z, 0.), z


def pz(v):
    """Causal expanding past-window z (clip +-3); rows with fewer than ZMINZ past windows -> NaN."""
    n = len(v); o = np.full(n, np.nan)
    for t in range(ZMINZ, n):
        P = v[:t]; sd = P.std()
        if sd > 0: o[t] = np.clip((v[t] - P.mean()) / sd, -CLIP, CLIP)
    return o


def pz_map(v, order_hd):
    """Apply pz to every subject's stream-ordered series (order_hd = row indices in stream order)."""
    o = np.full(len(v), np.nan)
    for loc in order_hd:
        o[loc] = pz(v[loc])
    return o


def merge_z(a, b, fallback):
    """Equal average of the two causal past-z channels; rows where neither is defined -> the plain average (fallback)."""
    c = np.isfinite(a).astype(float) + np.isfinite(b).astype(float)
    num = np.where(np.isfinite(a), a, 0.) + np.where(np.isfinite(b), b, 0.)
    return np.where(c > 0, num / np.maximum(c, 1), fallback)


def merge_subject(f1, l1, s2):
    """The five pre-registered scores for one subject's stream-ordered series (s2 finite everywhere)."""
    s2z = pz(s2); plain = W1 * f1 + W2 * s2
    return {'m_plain': plain, 'm_z': merge_z(pz(f1), s2z, plain), 'm_pseudo_only': f1, 'm_c4_only': s2,
            'm_hwf_c4': merge_z(pz(l1), s2z, plain)}


def main():
    warnings.filterwarnings('ignore', category=FutureWarning)
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); args = ap.parse_args()
    sfx = '_smoke' if args.smoke else ''
    stem = 'depositioned_final_merge' + sfx
    tee = Tee(sys.stdout, OUT / (stem + '.log')); sys.stdout = tee
    sseeds = STREAM_SEEDS[:1] if args.smoke else STREAM_SEEDS
    nb = 100 if args.smoke else 1000

    # ---------------- pipeline 1 data (depositioned order) ----------------
    man, subs, data = load_db()
    Fc = np.vstack([data[s][0] for s in subs]); y = np.concatenate([data[s][1] for s in subs])
    g = np.concatenate([np.full(len(data[s][1]), s) for s in subs])
    A_ = np.load(OUT / 'depositioned_aperiodic_emg_oof.npz'); B_ = np.load(OUT / 'depositioned_resnet_timefreq_oof.npz')
    assert np.array_equal(A_['y'], y) and np.array_equal(A_['groups'], g)
    for q in ('y', 'groups', 'fold'): assert np.array_equal(A_[q], B_[q]), q
    Fn = A_['new_features']; assert Fc.shape[1] == 858 and Fn.shape == (len(y), 31)
    fold = A_['fold']; n = len(y)

    # ---------------- pipeline 2 data (highdim order) ----------------
    P2 = np.load(OUT / 'merge_best_window6_oof.npz')
    yh, gh = P2['y'].astype(int), P2['groups'].astype(str)
    fold_h = P2['fold_id'].astype(int)
    s2_h = P2[P2_KEY]
    assert s2_h.shape == (len(yh),) and np.isfinite(s2_h).all()
    gsub_h = np.array(['sub-' + x for x in gh])

    # ---------------- alignment ----------------
    raw = np.load(OUT / 'highdim_deep_features.npz')['raw']
    assert raw.shape[0] == len(yh)
    key = defaultdict(list)
    for i in range(len(raw)):
        key[(gh[i], int(yh[i]), hashlib.md5(raw[i].tobytes()).hexdigest())].append(i)
    idx = np.empty(n, int); pos = 0
    for s in subs:
        for k in ('relax', 'task'):
            d = np.load(DB / 'windows' / k / f'{s}.npz')
            for x, l in zip(d['x'], d['label']):
                idx[pos] = key[(s[4:], int(l), hashlib.md5(x.astype(np.float32).tobytes()).hexdigest())].pop(0)
                pos += 1
    assert pos == n and len(np.unique(idx)) == n == len(yh) == len(gh)
    assert sum(len(v) for v in key.values()) == 0, 'unmatched highdim rows'
    assert np.array_equal(yh[idx], y), 'label mismatch after alignment'
    assert np.array_equal(gsub_h[idx], g), 'subject mismatch after alignment'
    assert np.array_equal(fold_h[idx], fold + 1), 'fold mismatch (pipeline 2 fold_id is 1-based)'
    assert all(int((g == s).sum()) == int((gsub_h == s).sum()) for s in subs), 'window count mismatch'
    print(f'[align] md5 multiset 1:1 n={n} subjects={len(subs)}; subject/label/window-count/fold agree {el()}', flush=True)

    def to_hd(v):
        o = np.full(n, np.nan); o[idx] = v; return o

    s2_c = s2_h[idx]                                # pipeline 2 in depositioned canonical order

    # ---------------- folds of pipeline 1 (asserted == npz) ----------------
    Xa, ya, ga, _ = arrange(subs, data, 0, 'independent'); fold_of = {}
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Xa, ya, ga)):
        for s in np.unique(ga[te]): fold_of[s] = k
    assert np.array_equal(np.array([fold_of[s] for s in g]), fold)
    run_folds = [0] if args.smoke else list(range(5))
    run_subs = [s for s in subs if fold_of[s] in run_folds]
    ev_c = np.isin(fold, run_folds); ev_h = np.isin(fold_h, np.array(run_folds) + 1)
    assert np.array_equal(ev_h[idx], ev_c)
    print(f'[check] pipeline-1 fold == depositioned_aperiodic_emg_oof.npz fold; folds={run_folds} '
          f'subjects={len(run_subs)} {el()}', flush=True)

    # ---------------- pipeline 1 test-side ZZ ----------------
    ZZt = np.full((n, 889), np.nan)
    for k in run_folds:
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        tfo, tfn = impute_scale(Fc, tr), impute_scale(Fn, tr)
        ZZt[te] = np.hstack([tfo(Fc[te]), tfn(Fn[te])])
        print(f'[fold {k}] test-side ZZ rows={len(te)} {el()}', flush=True)

    diag = json.loads((OUT / 'depositioned_subject_diagnosis.json').read_text(encoding='utf-8'))
    grp_of = {s: G for G in GROUPS for s in diag['diagnosis']['groups'][G]['subjects']}
    assert all(s in grp_of for s in subs)

    result = {'meta': {'pipeline1': 'depositioned_within_subject_adapt_v2.pseudo_lr_full (f1 = L + z)',
                       'pipeline2': f'merge_best_window6.{P2_KEY} (4.30 C4 + broadband LR + ResNet)',
                       'merge_weights': [W1, W2],
                       'primary_rule': 'm_plain (equal-weight logit average, no extra z; both parents already logit-scaled '
                                       'and between-subject calibrated)',
                       'alternative_rule': 'm_z (causal past-window z per pipeline, clipped +-3, then equal average)',
                       'refit_step': R_STEP, 'min_past': MIN_PAST, 'zmin_past': ZMINZ, 'clip': CLIP, 'eps': EPS,
                       'mature_index': MATURE, 'thr': {'k': THR_K, 'min_past': 16, 'mode': 'mean'},
                       'extra_threshold': .5, 'bootstrap_n': nb, 'stream_seeds': list(sseeds), 'smoke': args.smoke,
                       'variants': list(VARIANTS), 'merged': list(MERGED), 'parents': list(PARENTS)},
              'alignment': {'method': 'multiset (subject, label, md5(raw float32)) depositioned-canonical -> highdim',
                            'n_rows': int(n), 'n_subjects': len(subs), 'all_matched': True,
                            'pipeline2_fold_offset': 1},
              'checks': {}}
    checks = result['checks']; sidx = {s: j for j, s in enumerate(subs)}
    v2j = json.loads((OUT / 'depositioned_within_subject_adapt_v2.json').read_text(encoding='utf-8'))
    if not args.smoke:
        p2j = json.loads((OUT / 'merge_best_window6_results.json').read_text(encoding='utf-8'))['combos'][
            P2_KEY.replace('combo__', '')]
        m2 = metrics(yh, gsub_h, sig(s2_h), np.ones(len(yh), bool))
        assert abs(m2['AUC'] - p2j['oof_auc']) < 1e-9 and abs(m2['micro_bACC'] - p2j['bacc_at0']) < 1e-9
        checks['pipeline2_reproduced'] = {'AUC': p2j['oof_auc'], 'bACC_at0': p2j['bacc_at0']}
        print(f'[check] pipeline 2 reproduces 4.30 AUC={p2j["oof_auc"]:.4f} bACC@0={p2j["bacc_at0"]:.4f}', flush=True)

    for sd in sseeds:
        order, t_c = stream_order(g, subs, sd)
        order_hd = [idx[loc] for loc in order]
        order_of = {s: np.asarray(order[sidx[s]]) for s in run_subs}
        Bl = np.c_[logit(A_[f'stream{sd}__all_prank']), logit(B_[f'stream{sd}__resnet_whiten_logit_delta']),
                   logit(B_[f'stream{sd}__resnet_fbank_logit_delta'])]
        L = Bl.mean(1); assert np.isfinite(L).all()
        print(f'[s{sd}] std(L)={L[ev_c].std():.3f} mean(L)={L[ev_c].mean():+.3f}', flush=True)
        Lc = np.full(n, np.nan); f1c = np.full(n, np.nan)
        ts = time.perf_counter()
        gen = Parallel(n_jobs=N_JOBS, backend='loky', return_as='generator')(
            delayed(p1_subject)(ZZt[order_of[s]], Bl[order_of[s]], y[order_of[s]]) for s in run_subs)
        for i, (s, (Ls, f1s, _)) in enumerate(zip(run_subs, gen)):
            loc = order_of[s]; Lc[loc] = Ls; f1c[loc] = f1s
            if (i + 1) % 8 == 0 or i + 1 == len(run_subs):
                print(f'[s{sd}] subjects {i + 1}/{len(run_subs)} stream_s={time.perf_counter() - ts:.1f} {el()}',
                      flush=True)
        print(f'[s{sd}] pipeline 1 adapted {el()}', flush=True)

        # ---- merged scores in highdim row space
        f1_h, l_h = to_hd(f1c), to_hd(Lc)
        zf_h, zs_h = pz_map(f1_h, order_hd), pz_map(s2_h, order_hd)
        plain_h = W1 * f1_h + W2 * s2_h
        S = {'m_plain': plain_h, 'm_z': merge_z(zf_h, zs_h, plain_h), 'm_pseudo_only': f1_h,
             'm_c4_only': s2_h, 'm_hwf_c4': merge_z(pz_map(l_h, order_hd), zs_h, plain_h)}
        cov = float((np.isfinite(zf_h) & np.isfinite(zs_h))[ev_h].mean())
        print(f'[z s{sd}] coverage(both past-z channels available)={cov:.3f}', flush=True)

        # ---- checks on 2 subjects: isolation, label shuffle, past-only prefix
        for s in run_subs[:2]:
            loc = order_of[s]; Z0, B0, y0 = ZZt[loc], Bl[loc], y[loc]; ns = len(loc)
            L0, f0, _ = p1_subject(Z0, B0, y0)
            assert np.array_equal(f0, f1c[loc], equal_nan=True), ('isolation', s)
            assert np.array_equal(L0, Lc[loc], equal_nan=True), ('isolation L', s)
            f_sh = p1_subject(Z0, B0, np.random.default_rng(1).permutation(y0))[1]
            assert np.array_equal(f0, f_sh, equal_nan=True), ('label_shuffle', s)
            cut = ZMIN + R_STEP * ((ns - ZMIN) // (2 * R_STEP)) + 5
            rng = np.random.default_rng(int(loc[0])); Z1, B1 = Z0.copy(), B0.copy()
            Z1[cut:] += rng.normal(0, 2, Z1[cut:].shape); B1[cut:] += rng.normal(0, 2, B1[cut:].shape)
            _, f1p, _ = p1_subject(Z1, B1, y0)
            v0, v1 = merge_subject(f0, L0, s2_c[loc]), merge_subject(f1p, B1.mean(1), s2_c[loc])
            for nm in ('m_plain', 'm_z', 'm_pseudo_only', 'm_hwf_c4'):
                assert np.array_equal(v0[nm][:cut], v1[nm][:cut], equal_nan=True), ('prefix', s, nm)
                d = np.abs(v0[nm][cut:] - v1[nm][cut:]); fin = np.isfinite(d)
                assert fin.any() and d[fin].max() > 0, ('prefix-unchanged', s, nm)
            assert np.array_equal(v0['m_c4_only'], v1['m_c4_only'], equal_nan=True)
            assert np.array_equal(v0['m_z'], S['m_z'][idx[loc]], equal_nan=True), ('pooled', s)
            checks[f's{sd}_{s}'] = {'isolation': 'ok', 'label_shuffle': 'ok', 'past_only_prefix': 'ok', 'cut': int(cut)}
            print(f'[check s{sd}] {s}: isolation / label shuffle / past-only prefix OK (cut={cut})', flush=True)

        if not args.smoke:
            p_thr = sig(thr_apply(np.where(ev_c, f1c, np.nan), order, THR_K, 'mean'))
            mac = metrics(y, g, p_thr, ev_c)['subject_macro_bACC']
            ref = v2j[f'stream{sd}']['methods']['fuse_pseudo_lr_full_thr']['all']['subject_macro_bACC']
            assert abs(mac - ref) < 1e-9, ('v2 reproduction', mac, ref)
            checks[f's{sd}_v2_reproduced'] = ref
            print(f'[check s{sd}] fuse_pseudo_lr_full_thr reproduces v2 mac={ref:.4f}', flush=True)

        # ---- scores at fixed 0.5 and at thr_mean256
        P = {}
        for v in VARIANTS:
            l = np.where(ev_h, S[v], np.nan)
            P[v] = sig(l); P[f'{v}_thr'] = sig(thr_apply(l, order_hd, THR_K, 'mean'))
        assert all(np.isfinite(P[nm][ev_h]).all() for nm in P), 'NaN inside the evaluated rows'
        M = {'all': ev_h, 'mature': ev_h & (t_c[idx] >= MATURE)}
        R = result[f'stream{sd}'] = {'methods': {}, 'bootstrap': {}, 'subjects': {}, 'groups': {}, 'coverage_m_z': cov}
        ev_subs = list(np.unique(g[ev_c])); PS = {}
        for nm, p in P.items():
            r = R['methods'][nm] = {sg: metrics(yh, gsub_h, p, M[sg]) for sg in M}
            PS[nm] = {s: metrics(yh, gsub_h, p, ev_h & (gsub_h == s))['subject_macro_bACC'] for s in ev_subs}
            ba = np.array([q for q in PS[nm].values() if q is not None])
            R['subjects'][nm] = {'n_ge_0.9': int((ba >= .9).sum()), 'max': float(ba.max()), 'n': int(len(ba))}
            print(f'[result stream {sd}] {nm:20s} ' + ' | '.join(
                f'{sg}: AUC={r[sg]["AUC"]:.4f} mac={r[sg]["subject_macro_bACC"]:.4f} mic={r[sg]["micro_bACC"]:.4f}'
                for sg in M) + f' | subj>=.9={R["subjects"][nm]["n_ge_0.9"]} max={R["subjects"][nm]["max"]:.3f}',
                flush=True)
        for nm in P:
            R['groups'][nm] = {}
            for G in GROUPS:
                ss = [s for s in ev_subs if grp_of.get(s) == G and PS[nm][s] is not None]
                if not ss: continue
                a = np.array([PS[nm][s] for s in ss])
                row = {'n': len(ss), 'mean': float(a.mean()), 'median': float(np.median(a))}
                for par in PARENTS:
                    b = np.array([PS[par][s] for s in ss])
                    row[f'dmean_vs_{par}'] = float(a.mean() - b.mean())
                    row[f'dmed_vs_{par}'] = float(np.median(a) - np.median(b))
                R['groups'][nm][G] = row
            print(f'[group stream {sd}] {nm:20s} ' + ' | '.join(
                f'{G}(n={q["n"]}): mean={q["mean"]:.4f} dmean_vs_p1={q["dmean_vs_m_pseudo_only"]:+.4f} '
                f'dmean_vs_p2={q["dmean_vs_m_c4_only"]:+.4f}' for G, q in R['groups'][nm].items()), flush=True)
        pairs = [(a_, b_) for a_ in MERGED for b_ in PARENTS] + [('m_plain', 'm_z')]
        for a_, b_ in pairs:
            for sg in M:
                for suff in ('', '_thr'):
                    bb = R['bootstrap'][f'{a_}{suff}__vs__{b_}{suff}__{sg}'] = \
                        boot(P[f'{a_}{suff}'], P[f'{b_}{suff}'], yh, gsub_h, M[sg], nb)
                    print(f'[boot stream {sd}] {a_ + suff:22s} vs {b_ + suff:22s} {sg:6s} '
                          f'dAUC={bb["dAUC"]["mean"]:+.4f}[{bb["dAUC"]["p2.5"]:+.4f},{bb["dAUC"]["p97.5"]:+.4f}] '
                          f'dmac={bb["dmacro_bACC"]["mean"]:+.4f}[{bb["dmacro_bACC"]["p2.5"]:+.4f},{bb["dmacro_bACC"]["p97.5"]:+.4f}] '
                          f'dmic={bb["dmicro_bACC"]["mean"]:+.4f}[{bb["dmicro_bACC"]["p2.5"]:+.4f},{bb["dmicro_bACC"]["p97.5"]:+.4f}]',
                          flush=True)
        print(f'[s{sd}] done {el()}', flush=True)
        (OUT / (stem + '.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')

    result['seconds'] = time.perf_counter() - T0
    (OUT / (stem + '.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[done] all checks passed {el()} -> {stem}.json', flush=True)
    sys.stdout = tee.stream; tee.f.close()


if __name__ == '__main__':
    main()
