"""Idea: causal WITHIN-subject adaptation on top of fuseH_wf (oracle upper bound + pseudo labels + clustering).

Base score per stream sd: L = mean(logit(A['stream{sd}__all_prank']), logit(B['stream{sd}__resnet_whiten_logit_delta']),
logit(B['stream{sd}__resnet_fbank_logit_delta'])) (A = depositioned_aperiodic_emg_oof.npz, B =
depositioned_resnet_timefreq_oof.npz); fuseH_wf = sig(L). No cross-subject retraining.
Protocol: stream order = stream_order(g, subs, sd); outer folds = SGKF(5, shuffle, rs0) on seed-0 'independent'
arrangement, re-derived and asserted == npz fold. Per fold k: train-median imputation + StandardScaler(train rows) for the
858 features and the 31 new_features -> ZZ (889); within-subject adaptation runs for the test subjects of fold k.

Within-subject (per test subject, in stream order; row t only uses the subject's rows < t):
  Refit points t0 = 48, 64, 80, ... (R=16, MIN=48): model fitted on rows [0, t0) scores rows t0..t0+15. Rows t < 48 (or
  a refit whose requirements fail) -> NaN -> fallback (fusion = L, alone = sig(L)).
  Representation: StandardScaler(past rows of ZZ) -> PCA(16, svd_solver='full') fitted on past rows.
  Variants (fixed hyper-parameters):
    oracle_lda     LEAKY: TRUE labels of the subject's past rows; LDA(solver lsqr, shrinkage 'auto') on PCA16.
    oracle_lr      LEAKY: same labels; LogisticRegression(C=.1, lbfgs, max_iter 1000) on PCA16.
    oracle_lr_full LEAKY: same labels; LogisticRegression(C=.01) on past-scaled ZZ (889).
                   oracle_* require both classes with >= 8 past rows each, else NaN.
    pseudo_lda     legal: pseudo labels = top / bottom 30% of past L (stable argsort; top=1); LDA as above on PCA16.
    km2            legal: KMeans(2, n_init=5, random_state=0) on past PCA16; score = d(c0) - d(c1), oriented so the
                   cluster with higher mean past L is positive ("task").
    gmm_dir        LEAKY: same KMeans, oriented by TRUE past labels (cluster with higher past task fraction positive).
  Score -> logit-like: z = (s - mean(s_past)) / std(s_past), s_past = the same model's scores on its training (past)
  rows (NaN if std 0).
  Outputs: X (alone) = sig(z_X); fuse_X = sig(L + z_X) (equal weight; std of L printed for info);
  fuse_X_thr = thr_mean256 of the fused logit (apply(l, order, 256, 'mean') of depositioned_adaptive_threshold.py).
  NaN z -> fallback to L in all three. References: fuseH_wf, fuseH_wf_thr.
Evaluation: all / mature (t>=120); AUC, subject-macro bACC, micro bACC; #subjects bACC>=.9 and max; paired subject
bootstrap (1000; smoke 100) of every variant vs fuseH_wf (and *_thr vs fuseH_wf_thr); group (high/mid/low from
depositioned_subject_diagnosis.json, reporting only) macro bACC mean and change vs fuseH_wf.
Checks (assert): fold == npz; prefix invariance (future ZZ rows and L perturbed -> earlier outputs identical, later
differ) for every variant on 2 subjects; test-label shuffle -> pseudo_lda, km2 identical; subject isolation (subject
alone == pooled); naive reference (manual scaler/PCA/LDA/KMeans at one refit point).
Run: python test/depositioned_within_subject_adapt.py [--smoke]   (smoke: stream0, fold0, bootstrap 100)
"""
import argparse, json, sys, time, warnings
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from depositioned_history_delta import arrange, boot, load_db  # noqa: E402
from depositioned_resnet_history_delta import stream_order, logit, sig  # noqa: E402
from depositioned_adaptive_threshold import apply as thr_apply  # noqa: E402
from depositioned_symmetric_contrast import impute_scale  # noqa: E402
from causal_kernel_state_mapping import metrics  # noqa: E402
from switch_dual_memory import Tee  # noqa: E402

MATURE, STREAM_SEEDS = 120, (0, 1, 2)
THR_K, R_STEP, MIN_PAST, MIN_CLS, NPC, PSEUDO_Q = 256, 16, 48, 8, 16, .3
GROUPS = ('high', 'mid', 'low')
VARIANTS = ('oracle_lda', 'oracle_lr', 'oracle_lr_full', 'pseudo_lda', 'km2', 'gmm_dir')
LEAKY = ('oracle_lda', 'oracle_lr', 'oracle_lr_full', 'gmm_dir')
LEGAL = ('pseudo_lda', 'km2')
T0 = time.perf_counter()


def el():
    return f'elapsed={time.perf_counter() - T0:.1f}s'


def lda():
    return LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')


def zfit(s_past, s_new):
    sd = s_past.std()
    return (s_new - s_past.mean()) / sd if sd > 0 else np.full(len(s_new), np.nan)


def pseudo_labels(lp):
    n = len(lp); q = int(np.floor(PSEUDO_Q * n)); o = np.argsort(lp, kind='stable')
    return np.r_[o[:q], o[n - q:]], np.r_[np.zeros(q, int), np.ones(q, int)]


def refit(zp, lp, yp, zn):
    """One refit point: past rows (zp, lp, yp) -> z-scores for new rows zn, per variant."""
    out = {}
    sc = StandardScaler().fit(zp); xp, xn = sc.transform(zp), sc.transform(zn)
    pca = PCA(NPC, svd_solver='full').fit(xp); pp, pn = pca.transform(xp), pca.transform(xn)
    ok = min((yp == 0).sum(), (yp == 1).sum()) >= MIN_CLS
    for n, (Xp, Xn, C) in (('oracle_lda', (pp, pn, None)), ('oracle_lr', (pp, pn, .1)), ('oracle_lr_full', (xp, xn, .01))):
        if not ok: out[n] = np.full(len(zn), np.nan); continue
        m = lda() if C is None else LogisticRegression(C=C, max_iter=1000)
        m.fit(Xp, yp); out[n] = zfit(m.decision_function(Xp), m.decision_function(Xn))
    idx, yl = pseudo_labels(lp)
    m = lda().fit(pp[idx], yl); out['pseudo_lda'] = zfit(m.decision_function(pp), m.decision_function(pn))
    km = KMeans(2, n_init=5, random_state=0).fit(pp)
    dp, dn = km.transform(pp), km.transform(pn); sp, sn = dp[:, 0] - dp[:, 1], dn[:, 0] - dn[:, 1]
    lab = km.labels_
    mL = [lp[lab == c].mean() if (lab == c).any() else -np.inf for c in (0, 1)]
    sgn = 1. if mL[1] >= mL[0] else -1.  # score positive when closer to cluster1; flip if cluster0 has higher L
    out['km2'] = zfit(sgn * sp, sgn * sn)
    mY = [yp[lab == c].mean() if (lab == c).any() else -np.inf for c in (0, 1)]
    sgn = 1. if mY[1] >= mY[0] else -1.
    out['gmm_dir'] = zfit(sgn * sp, sgn * sn)
    return out


def adapt_subject(Z, L, y):
    """One subject in stream order: Z (n, 889), L (n,), y (n,) -> {variant: z (n,)} (NaN = fallback)."""
    n = len(L); res = {v: np.full(n, np.nan) for v in VARIANTS}
    for t0 in range(MIN_PAST, n, R_STEP):
        t1 = min(n, t0 + R_STEP)
        o = refit(Z[:t0], L[:t0], y[:t0], Z[t0:t1])
        for v in VARIANTS: res[v][t0:t1] = o[v]
    return res


def main():
    warnings.filterwarnings('ignore', category=FutureWarning)
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); args = ap.parse_args()
    stem = 'depositioned_within_subject_adapt' + ('_smoke' if args.smoke else '')
    tee = Tee(sys.stdout, OUT / (stem + '.log')); sys.stdout = tee
    sseeds = STREAM_SEEDS[:1] if args.smoke else STREAM_SEEDS
    nb = 100 if args.smoke else 1000
    man, subs, data = load_db()
    Fc = np.vstack([data[s][0] for s in subs]); y = np.concatenate([data[s][1] for s in subs])
    g = np.concatenate([np.full(len(data[s][1]), s) for s in subs])
    A_ = np.load(OUT / 'depositioned_aperiodic_emg_oof.npz'); B_ = np.load(OUT / 'depositioned_resnet_timefreq_oof.npz')
    assert np.array_equal(A_['y'], y) and np.array_equal(A_['groups'], g)
    for q in ('y', 'groups', 'fold'): assert np.array_equal(A_[q], B_[q]), q
    Fn = A_['new_features']; assert Fc.shape[1] == 858 and Fn.shape == (len(y), 31)
    diag = json.loads((OUT / 'depositioned_subject_diagnosis.json').read_text(encoding='utf-8'))
    grp_of = {s: G for G in GROUPS for s in diag['diagnosis']['groups'][G]['subjects']}
    assert all(s in grp_of for s in subs)
    print(f'[start] rows={len(y)} subjects={len(subs)} dims={Fc.shape[1]}+{Fn.shape[1]} smoke={args.smoke} '
          f'sseeds={sseeds} R={R_STEP} MIN={MIN_PAST} PCA={NPC} (oracle_* / gmm_dir LEAKY)', flush=True)

    Xa, ya, ga, _ = arrange(subs, data, 0, 'independent'); fold_of = {}
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Xa, ya, ga)):
        for s in np.unique(ga[te]): fold_of[s] = k
    fold = np.array([fold_of[s] for s in g]); assert np.array_equal(fold, A_['fold'])
    print('[check] fold == depositioned_aperiodic_emg_oof.npz fold', flush=True)
    run_folds = [0] if args.smoke else list(range(5))

    # ZZ rows of each fold's test subjects (scaled on that fold's train rows); independent of the stream order
    ZZt = np.full((len(y), 889), np.nan)
    for k in run_folds:
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        tfo, tfn = impute_scale(Fc, tr), impute_scale(Fn, tr)
        ZZt[te] = np.hstack([tfo(Fc[te]), tfn(Fn[te])])
    print(f'[zz] test-side ZZ built for folds {run_folds} {el()}', flush=True)

    result = {'meta': {'leaky_variants': list(LEAKY), 'legal_variants': list(LEGAL), 'refit_step': R_STEP,
                       'min_past': MIN_PAST, 'min_per_class_oracle': MIN_CLS, 'pca': NPC, 'pseudo_q': PSEUDO_Q,
                       'mature_index': MATURE, 'threshold': .5, 'thr': {'k': THR_K, 'min_past': 16, 'mode': 'mean'},
                       'bootstrap_n': nb, 'stream_seeds': list(sseeds), 'smoke': args.smoke}, 'checks': {}}
    checks = result['checks']
    sidx = {s: j for j, s in enumerate(subs)}
    run_subs = [s for s in subs if fold_of[s] in run_folds]
    with threadpool_limits(limits=1):
        for sd in sseeds:
            order, t = stream_order(g, subs, sd)
            L = np.mean([logit(A_[f'stream{sd}__all_prank']), logit(B_[f'stream{sd}__resnet_whiten_logit_delta']),
                         logit(B_[f'stream{sd}__resnet_fbank_logit_delta'])], 0)
            assert np.isfinite(L).all()
            ev = np.isin(fold, run_folds)
            print(f'[s{sd}] std(L)={L[ev].std():.3f} mean(L)={L[ev].mean():+.3f}', flush=True)
            Zs = {v: np.full(len(y), np.nan) for v in VARIANTS}
            for k in run_folds:
                tk = time.perf_counter(); ks = [s for s in run_subs if fold_of[s] == k]
                for i, s in enumerate(ks):
                    loc = order[sidx[s]]
                    r = adapt_subject(ZZt[loc], L[loc], y[loc])
                    for v in VARIANTS: Zs[v][loc] = r[v]
                    if (i + 1) % 4 == 0 or i + 1 == len(ks):
                        print(f'[s{sd} fold {k + 1}/5] subjects {i + 1}/{len(ks)} fold_s={time.perf_counter() - tk:.1f} '
                              f'{el()}', flush=True)
            # ---- checks (first two run subjects of this stream)
            for s in run_subs[:2]:
                loc = order[sidx[s]]; Z0, L0, y0 = ZZt[loc], L[loc], y[loc]; n = len(loc)
                r0 = adapt_subject(Z0, L0, y0)
                for v in VARIANTS:  # subject isolation: alone == pooled
                    assert np.array_equal(r0[v], Zs[v][loc], equal_nan=True), ('isolation', s, v)
                cut = MIN_PAST + R_STEP * ((n - MIN_PAST) // (2 * R_STEP)) + 5  # mid-block cut
                rng = np.random.default_rng(sidx[s])
                Z1, L1 = Z0.copy(), L0.copy()
                Z1[cut:] += rng.normal(0, 2, Z1[cut:].shape); L1[cut:] += rng.normal(0, 2, n - cut)
                r1 = adapt_subject(Z1, L1, y0)
                for v in VARIANTS:  # prefix invariance (outputs for rows < cut use rows < cut only)
                    assert np.array_equal(r0[v][:cut], r1[v][:cut], equal_nan=True), ('prefix', s, v)
                    a, b = r0[v][cut:], r1[v][cut:]; f = np.isfinite(a) & np.isfinite(b)
                    assert f.any() and np.abs(a[f] - b[f]).max() > 0, ('prefix-changed', s, v)
                ys = np.random.default_rng(1).permutation(y0); r2 = adapt_subject(Z0, L0, ys)
                dsh = {}
                for v in VARIANTS:
                    eq = np.array_equal(r0[v], r2[v], equal_nan=True); dsh[v] = bool(eq)
                    if v in LEGAL: assert eq, ('label_shuffle', s, v)
                # naive reference at one refit point
                t0 = MIN_PAST + R_STEP * 2; t1 = min(n, t0 + R_STEP)
                sc = StandardScaler().fit(Z0[:t0]); pc = PCA(NPC, svd_solver='full').fit(sc.transform(Z0[:t0]))
                P, Pn = pc.transform(sc.transform(Z0[:t0])), pc.transform(sc.transform(Z0[t0:t1]))
                m = lda().fit(P, y0[:t0]); sp, sn = m.decision_function(P), m.decision_function(Pn)
                assert np.allclose(r0['oracle_lda'][t0:t1], (sn - sp.mean()) / sp.std(), atol=1e-10)
                o = np.argsort(L0[:t0], kind='stable'); q = int(PSEUDO_Q * t0)
                m = lda().fit(P[np.r_[o[:q], o[t0 - q:]]], np.r_[np.zeros(q), np.ones(q)])
                sp, sn = m.decision_function(P), m.decision_function(Pn)
                assert np.allclose(r0['pseudo_lda'][t0:t1], (sn - sp.mean()) / sp.std(), atol=1e-10)
                km = KMeans(2, n_init=5, random_state=0).fit(P); d, dn = km.transform(P), km.transform(Pn)
                sgn = 1. if L0[:t0][km.labels_ == 1].mean() >= L0[:t0][km.labels_ == 0].mean() else -1.
                sp, sn = sgn * (d[:, 0] - d[:, 1]), sgn * (dn[:, 0] - dn[:, 1])
                assert np.allclose(r0['km2'][t0:t1], (sn - sp.mean()) / sp.std(), atol=1e-10)
                assert np.isnan(r0['km2'][:MIN_PAST]).all()
                checks[f's{sd}_{s}'] = {'isolation': 'ok', 'prefix_invariance': 'ok', 'naive_ref': 'ok', 'cut': int(cut),
                                        'label_shuffle_identical': dsh}
                print(f'[check s{sd}] {s}: isolation / prefix invariance / naive ref OK; label shuffle identical: {dsh}',
                      flush=True)
            # ---- scores
            S = {'fuseH_wf': sig(L), 'fuseH_wf_thr': sig(thr_apply(np.where(ev, L, np.nan), order, THR_K, 'mean'))}
            cover = {}
            for v in VARIANTS:
                z = Zs[v]; f = np.isfinite(z); cover[v] = float(f[ev].mean())
                S[v] = sig(np.where(f, z, L))
                lf = np.where(f, L + z, L); lf = np.where(ev, lf, np.nan)
                S[f'fuse_{v}'] = sig(lf); S[f'fuse_{v}_thr'] = sig(thr_apply(lf, order, THR_K, 'mean'))
                print(f'[s{sd}] {v:14s} coverage={cover[v]:.3f} std(z)={np.nanstd(z[ev]):.3f}', flush=True)
            for n_ in S: S[n_] = np.where(ev, S[n_], np.nan)
            M = {'all': ev, 'mature': ev & (t >= MATURE)}
            R = result[f'stream{sd}'] = {'std_L': float(L[ev].std()), 'coverage': cover, 'methods': {},
                                         'bootstrap': {}, 'subjects': {}, 'groups': {}}
            ev_subs = list(np.unique(g[ev])); PS = {}
            for n_, p in S.items():
                r = R['methods'][n_] = {s: metrics(y, g, p, M[s]) for s in M}
                PS[n_] = {s: metrics(y, g, p, ev & (g == s))['subject_macro_bACC'] for s in ev_subs}
                ba = np.array([q for q in PS[n_].values() if q is not None])
                R['subjects'][n_] = {'n_ge_0.9': int((ba >= .9).sum()), 'max': float(ba.max()), 'n': int(len(ba))}
                tag = ' [LEAKY]' if any(v in n_ for v in LEAKY) else ''
                print(f'[result stream {sd}] {n_:22s} ' + ' | '.join(
                    f'{s}: AUC={r[s]["AUC"]:.4f} mac={r[s]["subject_macro_bACC"]:.4f} mic={r[s]["micro_bACC"]:.4f}'
                    for s in M) + f' | subj>=.9={R["subjects"][n_]["n_ge_0.9"]} max={R["subjects"][n_]["max"]:.3f}{tag}',
                    flush=True)
            for n_ in S:
                R['groups'][n_] = {}
                for G in GROUPS:
                    ss = [s for s in ev_subs if grp_of.get(s) == G and PS[n_][s] is not None]
                    if not ss: continue
                    a = np.array([PS[n_][s] for s in ss]); b = np.array([PS['fuseH_wf'][s] for s in ss])
                    R['groups'][n_][G] = {'n': len(ss), 'median': float(np.median(a)), 'mean': float(a.mean()),
                                          'd_median_vs_fuseH_wf': float(np.median(a) - np.median(b)),
                                          'd_mean_vs_fuseH_wf': float(a.mean() - b.mean())}
                print(f'[group stream {sd}] {n_:22s} ' + ' | '.join(
                    f'{G}(n={q["n"]}): mean={q["mean"]:.4f} dmean={q["d_mean_vs_fuseH_wf"]:+.4f} '
                    f'dmed={q["d_median_vs_fuseH_wf"]:+.4f}' for G, q in R['groups'][n_].items()), flush=True)
            pairs = [(n_, 'fuseH_wf') for n_ in S if n_ != 'fuseH_wf'] + \
                    [(f'fuse_{v}_thr', 'fuseH_wf_thr') for v in VARIANTS]
            for a_, b_ in pairs:
                for seg in M:
                    bb = R['bootstrap'][f'{a_}__vs__{b_}__{seg}'] = boot(S[a_], S[b_], y, g, M[seg], nb)
                    print(f'[boot stream {sd}] {a_:22s} vs {b_:12s} {seg:6s} '
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
