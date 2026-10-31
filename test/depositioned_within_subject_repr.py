"""Within-subject REPRESENTATIONS for causal pseudo-label adaptation (goal: better low-group pseudo labels / fuse bACC).

Protocol identical to depositioned_within_subject_adapt_v2.py (folds SGKF(5, shuffle, rs0) on seed-0 'independent'
arrangement asserted == npz fold; stream_order; L = mean(a, w, f) branch logits; fuseH_wf = sig(L); refit points t0 = 48,
64, ... (R=16, MIN_PAST=48): every model is fitted on the subject's rows [0, t0) only and scores rows t0..t0+15; rows < 48
-> NaN -> fallback to L). ZZ (889) = train-fold median imputation + StandardScaler(train-fold rows) of the 858 stored
features and the 31 new_features. ZP2 (858) = the same impute_scale of the P2 features (preproc_study_feat_P2.npz:
detrend + in-window zero-phase band-pass/notch; rows = canonical load_db order, asserted via the 11 'raw_' columns
847-857 which do not depend on preprocessing: exact equality with stored f).
Pseudo labels (all except gmm_orient) = top / bottom 30% of past L (stable argsort, top=1; v1 pseudo_labels).
Within-subject past scaling: StandardScaler fitted on past rows. LR = LogisticRegression(C=.01, lbfgs, max_iter 1000).
Variants (fixed before running):
  ref_lr_full  = v2 pseudo_lr_full: LR on past-scaled ZZ (asserted: fuse macro bACC == v2 json fuse_pseudo_lr_full).
  rank_lr      every ZZ column -> empirical CDF w.r.t. the past rows: r(x) = (#{past < x} + .5 #{past == x}) / n_past
               (past rows included in their own reference set), feature = sqrt(12) * (r - .5); LR on these.
  p2_lr        LR on past-scaled ZP2 (858; replaces ZZ entirely).
  cat_lr       LR on past-scaled [ZZ | ZP2] (1747).
  fam_vote     families of the 858 columns = FAMILY_RULES of depositioned_lowgroup_diagnosis (manifest feature_names;
               'other' kept as a family if any), the 31 new_features = one family 'aper_emg'. Per family LR on the
               past-scaled ZZ columns of that family; per-family score standardised by its past mean/std (unclipped,
               std floor 1e-6); score = mean over families.
  spread       PCA16 (svd full) of past-scaled ZZ; LabelSpreading(kernel='knn', n_neighbors=10, alpha=.2, default
               max_iter 30 / tol 1e-3) with pseudo-label seeds, other past rows unlabelled (-1); LR on past-scaled ZZ
               fitted on ALL past rows with the transduced labels (one transduced class only -> NaN -> fallback).
  gmm_orient   GaussianMixture(2, covariance_type='full', random_state=0) on the same past PCA16; s = p(c1) - p(c0);
               sign flipped iff Spearman(s_past, L_past) < 0 (NaN -> no flip).
Every variant: z = clip((s - mean(s_past)) / max(std(s_past), 1e-6), -3, 3) with s_past = the model's scores on its own
past rows; fuse_X = sig(L + z_X), fuse_X_thr = thr_mean256 of the fused logit (depositioned_adaptive_threshold.apply).
Diagnosis only (uses labels, never feeds back): per refit point accuracy vs TRUE labels of the final training labels
(pseudo seeds for ref/rank/p2/cat/fam_vote = 'pseudo'; transduced past labels for spread; oriented hard GMM labels s>0
for gmm_orient) -> mean per subject -> overall / high / mid / low; per refit point within-subject AUC of every variant's
past scores vs true past labels (both classes present) -> same aggregation.
Evaluation as v2: all / mature (t>=120); AUC, subject-macro bACC, micro bACC, #subjects bACC>=.9, max; groups
high/mid/low (depositioned_subject_diagnosis.json); paired subject bootstrap (1000; smoke 100) of fuse_X vs fuse_ref_lr_full
and fuse_X_thr vs fuse_ref_lr_full_thr (plus ref vs fuseH_wf(_thr)).
Checks (assert): fold == npz; P2 row alignment; ref reproduction of v2; prefix invariance (future ZZ / ZP2 / branch logits
perturbed) for every variant on 2 subjects per stream; test-label shuffle -> all variants identical; subject isolation
(alone == parallel pooled); naive re-implementation at t0 = 80 of ref_lr_full, rank_lr (per-column searchsorted) and spread
(closed-form label spreading on a manual kNN graph).
Parallel over subjects: joblib loky, 8 jobs, 1 BLAS thread each (deterministic).
Run: python test/depositioned_within_subject_repr.py [--smoke]   (smoke: stream0, fold0, bootstrap 100)
"""
import argparse, json, sys, time, warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from depositioned_history_delta import DB, arrange, boot, load_db  # noqa: E402
from depositioned_resnet_history_delta import stream_order, logit, sig  # noqa: E402
from depositioned_adaptive_threshold import apply as thr_apply  # noqa: E402
from depositioned_symmetric_contrast import impute_scale  # noqa: E402
from depositioned_lowgroup_diagnosis import families  # noqa: E402
from causal_kernel_state_mapping import metrics  # noqa: E402
from switch_dual_memory import Tee  # noqa: E402
from depositioned_within_subject_adapt import pseudo_labels, GROUPS, MATURE, STREAM_SEEDS, THR_K, R_STEP, \
    MIN_PAST, NPC, PSEUDO_Q  # noqa: E402

CLIP, EPS, N_JOBS, KNN, ALPHA = 3., 1e-6, 8, 10, .2
VARIANTS = ('ref_lr_full', 'rank_lr', 'p2_lr', 'cat_lr', 'fam_vote', 'spread', 'gmm_orient')
ACC_KEYS = ('pseudo', 'spread', 'gmm_orient')
REF = 'ref_lr_full'
T0 = time.perf_counter()


def el():
    return f'elapsed={time.perf_counter() - T0:.1f}s'


def zc(sp, sn):
    return np.clip((sn - sp.mean()) / max(sp.std(), EPS), -CLIP, CLIP)


def lr():
    return LogisticRegression(C=.01, max_iter=1000)


def rank_tf(zp, zn):
    n = len(zp); rp = (rankdata(zp, axis=0) - .5) / n
    rn = ((zn[:, None, :] > zp[None]).sum(1) + .5 * (zn[:, None, :] == zp[None]).sum(1)) / n
    return np.sqrt(12.) * (rp - .5), np.sqrt(12.) * (rn - .5)


def fit_z(X, y, Xp, Xn):
    m = lr().fit(X, y); sp = m.decision_function(Xp); return zc(sp, m.decision_function(Xn)), sp


def refit(zp, pp2, Bp, yp, zn, pn2, fam):
    out, dg, sps = {}, {}, {}
    lp = Bp.mean(1)
    sc = StandardScaler().fit(zp); xp, xn = sc.transform(zp), sc.transform(zn)
    s2 = StandardScaler().fit(pp2); qp, qn = s2.transform(pp2), s2.transform(pn2)
    idx, yl = pseudo_labels(lp); dg['pseudo'] = float((yp[idx] == yl).mean())
    out['ref_lr_full'], sps['ref_lr_full'] = fit_z(xp[idx], yl, xp, xn)
    rp, rn = rank_tf(zp, zn)
    out['rank_lr'], sps['rank_lr'] = fit_z(rp[idx], yl, rp, rn)
    out['p2_lr'], sps['p2_lr'] = fit_z(qp[idx], yl, qp, qn)
    cp, cn = np.hstack([xp, qp]), np.hstack([xn, qn])
    out['cat_lr'], sps['cat_lr'] = fit_z(cp[idx], yl, cp, cn)
    ap, an = np.zeros(len(xp)), np.zeros(len(xn))
    for cols in fam:
        m = lr().fit(xp[idx][:, cols], yl); a, b = m.decision_function(xp[:, cols]), m.decision_function(xn[:, cols])
        mu, sd = a.mean(), max(a.std(), EPS); ap += (a - mu) / sd; an += (b - mu) / sd
    ap /= len(fam); an /= len(fam)
    out['fam_vote'], sps['fam_vote'] = zc(ap, an), ap
    pca = PCA(NPC, svd_solver='full').fit(xp); pp, pn = pca.transform(xp), pca.transform(xn)
    ys = np.full(len(xp), -1); ys[idx] = yl
    tr = LabelSpreading(kernel='knn', n_neighbors=KNN, alpha=ALPHA).fit(pp, ys).transduction_
    dg['spread'] = float((tr == yp).mean()); dg['spread_flip_vs_seed'] = float((tr[idx] != yl).mean())
    if len(np.unique(tr)) == 2: out['spread'], sps['spread'] = fit_z(xp, tr, xp, xn)
    else: out['spread'] = np.full(len(xn), np.nan); dg['spread_degenerate'] = True
    gm = GaussianMixture(2, covariance_type='full', random_state=0).fit(pp)
    a = gm.predict_proba(pp); b = gm.predict_proba(pn); a, b = a[:, 1] - a[:, 0], b[:, 1] - b[:, 0]
    rho = spearmanr(a, lp)[0]
    if rho < 0: a, b = -a, -b
    out['gmm_orient'], sps['gmm_orient'] = zc(a, b), a
    dg['gmm_orient'] = float(((a > 0).astype(int) == yp).mean())
    if min((yp == 0).sum(), (yp == 1).sum()) > 0:
        dg['past_auc'] = {v: float(roc_auc_score(yp, s)) for v, s in sps.items() if np.std(s) > 0}
    return out, dg


def adapt_subject(Z, P2, Bl, y, fam):
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        n = len(y); res = {v: np.full(n, np.nan) for v in VARIANTS}; diags = []
        for t0 in range(MIN_PAST, n, R_STEP):
            t1 = min(n, t0 + R_STEP)
            o, dg = refit(Z[:t0], P2[:t0], Bl[:t0], y[:t0], Z[t0:t1], P2[t0:t1], fam); diags.append(dg)
            for v in VARIANTS: res[v][t0:t1] = o[v]
    return res, diags


def naive_check(Z0, B0, y0, r0):
    """Independent re-implementation of ref_lr_full, rank_lr and spread at refit point t0 = MIN_PAST + 2R."""
    n = len(y0); t0 = MIN_PAST + R_STEP * 2; t1 = min(n, t0 + R_STEP)
    mu, sd = Z0[:t0].mean(0), Z0[:t0].std(0); sd[sd == 0] = 1.
    Xp, Xn = (Z0[:t0] - mu) / sd, (Z0[t0:t1] - mu) / sd
    L0 = B0[:t0].mean(1); q = int(PSEUDO_Q * t0); o = np.argsort(L0, kind='stable')
    i, yl = np.r_[o[:q], o[t0 - q:]], np.r_[np.zeros(q), np.ones(q)]

    def z(X, yy, A, An):
        m = lr().fit(X, yy); a, b = m.decision_function(A), m.decision_function(An)
        return np.clip((b - a.mean()) / max(a.std(), 1e-6), -3, 3)

    d = {'ref_lr_full': np.abs(r0['ref_lr_full'][t0:t1] - z(Xp[i], yl, Xp, Xn)).max()}
    Rp, Rn = np.empty_like(Z0[:t0]), np.empty_like(Z0[t0:t1])
    for j in range(Z0.shape[1]):
        s = np.sort(Z0[:t0, j])
        for M, src in ((Rp, Z0[:t0, j]), (Rn, Z0[t0:t1, j])):
            lo, hi = np.searchsorted(s, src, 'left'), np.searchsorted(s, src, 'right')
            M[:, j] = (lo + .5 * (hi - lo)) / t0
    Rp, Rn = np.sqrt(12.) * (Rp - .5), np.sqrt(12.) * (Rn - .5)
    rp, rn = rank_tf(Z0[:t0], Z0[t0:t1]); assert np.allclose(rp, Rp, atol=1e-12) and np.allclose(rn, Rn, atol=1e-12)
    d['rank_lr'] = np.abs(r0['rank_lr'][t0:t1] - z(Rp[i], yl, Rp, Rn)).max()
    P = PCA(NPC, svd_solver='full').fit(Xp).transform(Xp)
    D2 = ((P[:, None] - P[None]) ** 2).sum(-1); nb = np.argsort(D2, 1, kind='stable')[:, :KNN]
    A = np.zeros((t0, t0)); A[np.repeat(np.arange(t0), KNN), nb.ravel()] = 1.; np.fill_diagonal(A, 0.)
    dg = np.sqrt(A.sum(0)); dg[dg == 0] = 1.; S = A / dg[None] / dg[:, None]
    Y = np.zeros((t0, 2)); Y[i, yl.astype(int)] = 1.
    F = np.linalg.solve(np.eye(t0) - ALPHA * S, (1 - ALPHA) * Y); tr = F.argmax(1)
    m_ = lr().fit(Xp, tr); a = m_.decision_function(Xp)
    d['spread'] = np.abs(r0['spread'][t0:t1] - np.clip((m_.decision_function(Xn) - a.mean()) / max(a.std(), 1e-6),
                                                       -3, 3)).max()
    print(f'[naive] t0={t0} max|diff| ' + ' '.join(f'{k}={v:.2e}' for k, v in d.items()), flush=True)
    for k, v in d.items(): assert v < 1e-2, (k, v)
    for v in VARIANTS: assert np.isnan(r0[v][:MIN_PAST]).all()
    return {k: float(v) for k, v in d.items()}


def agg(per, grp_of):
    r = {'all': float(np.mean(list(per.values()))), 'n_subjects': len(per)}
    for G in GROUPS:
        a = [per[s] for s in per if grp_of[s] == G]
        if a: r[G] = float(np.mean(a))
    return r


def main():
    warnings.filterwarnings('ignore', category=FutureWarning)
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); args = ap.parse_args()
    sfx = '_smoke' if args.smoke else ''
    stem = 'depositioned_within_subject_repr' + sfx
    tee = Tee(sys.stdout, OUT / (stem + '.log')); sys.stdout = tee
    sseeds = STREAM_SEEDS[:1] if args.smoke else STREAM_SEEDS
    nb = 100 if args.smoke else 1000
    v2 = json.loads((OUT / f'depositioned_within_subject_adapt_v2{sfx}.json').read_text(encoding='utf-8'))
    man, subs, data = load_db()
    Fc = np.vstack([data[s][0] for s in subs]); y = np.concatenate([data[s][1] for s in subs])
    g = np.concatenate([np.full(len(data[s][1]), s) for s in subs])
    A_ = np.load(OUT / 'depositioned_aperiodic_emg_oof.npz'); B_ = np.load(OUT / 'depositioned_resnet_timefreq_oof.npz')
    assert np.array_equal(A_['y'], y) and np.array_equal(A_['groups'], g)
    for q in ('y', 'groups', 'fold'): assert np.array_equal(A_[q], B_[q]), q
    Fn = A_['new_features']; assert Fc.shape[1] == 858 and Fn.shape == (len(y), 31)
    P2z = np.load(OUT / 'preproc_study_feat_P2.npz'); assert P2z.files == ['F']
    F2 = P2z['F'].astype(float); assert F2.shape == Fc.shape
    names = json.loads((DB / 'manifest.json').read_text(encoding='utf-8'))['feature_names']; assert len(names) == 858
    raw_cols = [j for j, nm in enumerate(names) if nm.startswith('raw_')]; assert raw_cols == list(range(847, 858))
    assert np.array_equal(F2[:, raw_cols], Fc[:, raw_cols]), 'P2 row alignment'
    print(f'[check] P2 rows aligned with load_db order (11 raw_ columns identical, {len(y)} rows)', flush=True)
    fd = families(names); fam_names = sorted(fd) + ['aper_emg']
    fam = [fd[k] for k in sorted(fd)] + [np.arange(858, 889)]
    assert sorted(np.concatenate(fam).tolist()) == list(range(889))
    print('[fam] ' + ' '.join(f'{k}={len(c)}' for k, c in zip(fam_names, fam)), flush=True)
    diag = json.loads((OUT / 'depositioned_subject_diagnosis.json').read_text(encoding='utf-8'))
    grp_of = {s: G for G in GROUPS for s in diag['diagnosis']['groups'][G]['subjects']}
    assert all(s in grp_of for s in subs)
    print(f'[start] rows={len(y)} subjects={len(subs)} smoke={args.smoke} sseeds={sseeds} R={R_STEP} MIN={MIN_PAST} '
          f'PCA={NPC} clip=+-{CLIP} knn={KNN} alpha={ALPHA} jobs={N_JOBS}', flush=True)

    Xa, ya, ga, _ = arrange(subs, data, 0, 'independent'); fold_of = {}
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Xa, ya, ga)):
        for s in np.unique(ga[te]): fold_of[s] = k
    fold = np.array([fold_of[s] for s in g]); assert np.array_equal(fold, A_['fold'])
    print('[check] fold == depositioned_aperiodic_emg_oof.npz fold', flush=True)
    run_folds = [0] if args.smoke else list(range(5))
    ZZt = np.full((len(y), 889), np.nan); ZPt = np.full((len(y), 858), np.nan)
    for k in run_folds:
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        tfo, tfn, tf2 = impute_scale(Fc, tr), impute_scale(Fn, tr), impute_scale(F2, tr)
        ZZt[te] = np.hstack([tfo(Fc[te]), tfn(Fn[te])]); ZPt[te] = tf2(F2[te])
        print(f'[zz] fold {k} ZZ / ZP2 built ({len(te)} test rows) {el()}', flush=True)

    result = {'meta': {'variants': list(VARIANTS), 'reference': REF, 'refit_step': R_STEP, 'min_past': MIN_PAST,
                       'pca': NPC, 'pseudo_q': PSEUDO_Q, 'clip': CLIP, 'eps': EPS, 'knn': KNN, 'alpha': ALPHA,
                       'families': {k: int(len(c)) for k, c in zip(fam_names, fam)}, 'mature_index': MATURE,
                       'threshold': .5, 'thr': {'k': THR_K, 'min_past': 16, 'mode': 'mean'}, 'bootstrap_n': nb,
                       'stream_seeds': list(sseeds), 'smoke': args.smoke}, 'checks': {}}
    checks = result['checks']; sidx = {s: j for j, s in enumerate(subs)}
    run_subs = [s for s in subs if fold_of[s] in run_folds]
    for sd in sseeds:
        order, t = stream_order(g, subs, sd)
        Bl = np.c_[logit(A_[f'stream{sd}__all_prank']), logit(B_[f'stream{sd}__resnet_whiten_logit_delta']),
                   logit(B_[f'stream{sd}__resnet_fbank_logit_delta'])]
        L = Bl.mean(1); assert np.isfinite(L).all()
        ev = np.isin(fold, run_folds)
        Zs = {v: np.full(len(y), np.nan) for v in VARIANTS}; DG = {}
        ts = time.perf_counter()
        gen = Parallel(n_jobs=N_JOBS, backend='loky', return_as='generator')(
            delayed(adapt_subject)(ZZt[order[sidx[s]]], ZPt[order[sidx[s]]], Bl[order[sidx[s]]], y[order[sidx[s]]], fam)
            for s in run_subs)
        for i, (s, (r, dg)) in enumerate(zip(run_subs, gen)):
            loc = order[sidx[s]]; DG[s] = dg
            for v in VARIANTS: Zs[v][loc] = r[v]
            print(f'[s{sd}] subject {s} (fold {fold_of[s]}) {i + 1}/{len(run_subs)} refits={len(dg)} '
                  f'stream_s={time.perf_counter() - ts:.1f} {el()}', flush=True)
        # ---- checks
        for s in run_subs[:2]:
            loc = order[sidx[s]]; Z0, P0, B0, y0 = ZZt[loc], ZPt[loc], Bl[loc], y[loc]; n = len(loc)
            r0, _ = adapt_subject(Z0, P0, B0, y0, fam)
            for v in VARIANTS: assert np.array_equal(r0[v], Zs[v][loc], equal_nan=True), ('isolation', s, v)
            cut = MIN_PAST + R_STEP * ((n - MIN_PAST) // (2 * R_STEP)) + 5
            rng = np.random.default_rng(sidx[s]); Z1, P1, B1 = Z0.copy(), P0.copy(), B0.copy()
            Z1[cut:] += rng.normal(0, 2, Z1[cut:].shape); P1[cut:] += rng.normal(0, 2, P1[cut:].shape)
            B1[cut:] += rng.normal(0, 2, B1[cut:].shape)
            r1, _ = adapt_subject(Z1, P1, B1, y0, fam)
            for v in VARIANTS:
                assert np.array_equal(r0[v][:cut], r1[v][:cut], equal_nan=True), ('prefix', s, v)
                a, b = r0[v][cut:], r1[v][cut:]; f = np.isfinite(a) & np.isfinite(b)
                assert f.any() and np.abs(a[f] - b[f]).max() > 0, ('prefix-changed', s, v)
            r2, _ = adapt_subject(Z0, P0, B0, np.random.default_rng(1).permutation(y0), fam)
            for v in VARIANTS: assert np.array_equal(r0[v], r2[v], equal_nan=True), ('label_shuffle', s, v)
            nd = naive_check(Z0, B0, y0, r0)
            checks[f's{sd}_{s}'] = {'isolation': 'ok', 'prefix_invariance': 'ok', 'label_shuffle_identical': 'ok',
                                    'naive_ref': nd, 'cut': int(cut)}
            print(f'[check s{sd}] {s}: isolation / prefix / label shuffle / naive ref OK {el()}', flush=True)
        # ---- diagnosis
        acc = {kx: agg({s: float(np.mean(q)) for s in run_subs if (q := [d[kx] for d in DG[s] if kx in d])}, grp_of)
               for kx in ACC_KEYS}
        pauc = {}
        for v in VARIANTS:
            per = {s: float(np.mean(q)) for s in run_subs
                   if (q := [d['past_auc'][v] for d in DG[s] if v in d.get('past_auc', {})])}
            pauc[v] = agg(per, grp_of)
        nref = sum(len(DG[s]) for s in run_subs)
        ndeg = sum(d.get('spread_degenerate', False) for s in run_subs for d in DG[s])
        flip = float(np.mean([d['spread_flip_vs_seed'] for s in run_subs for d in DG[s]]))
        for kx, q in acc.items():
            print(f'[label-acc s{sd}] {kx:12s} all={q["all"]:.4f} ' +
                  ' '.join(f'{G}={q[G]:.4f}' for G in GROUPS if G in q) + f' (n_subj={q["n_subjects"]})', flush=True)
        print(f'[spread s{sd}] degenerate {ndeg}/{nref}; seed flip rate={flip:.4f}', flush=True)
        for v, q in pauc.items():
            print(f'[past-AUC s{sd}] {v:12s} all={q["all"]:.4f} ' +
                  ' '.join(f'{G}={q[G]:.4f}' for G in GROUPS if G in q), flush=True)
        # ---- scores
        S = {'fuseH_wf': sig(L), 'fuseH_wf_thr': sig(thr_apply(np.where(ev, L, np.nan), order, THR_K, 'mean'))}
        zinfo = {}
        for v in VARIANTS:
            z = Zs[v]; f = np.isfinite(z); zf = z[ev & f]
            zinfo[v] = {'coverage': float(f[ev].mean()), 'std': float(zf.std()),
                        'frac_clipped': float((np.abs(zf) >= CLIP).mean())}
            lf = np.where(ev, np.where(f, L + z, L), np.nan)
            S[f'fuse_{v}'] = sig(lf); S[f'fuse_{v}_thr'] = sig(thr_apply(lf, order, THR_K, 'mean'))
            print(f'[z s{sd}] {v:12s} coverage={zinfo[v]["coverage"]:.3f} std={zinfo[v]["std"]:.3g} '
                  f'clipped={zinfo[v]["frac_clipped"]:.3f}', flush=True)
        for n_ in S: S[n_] = np.where(ev, S[n_], np.nan)
        M = {'all': ev, 'mature': ev & (t >= MATURE)}
        R = result[f'stream{sd}'] = {'z_info': zinfo, 'label_acc': acc, 'past_auc': pauc,
                                     'spread': {'degenerate': int(ndeg), 'refits': nref, 'seed_flip_rate': flip},
                                     'methods': {}, 'bootstrap': {}, 'subjects': {}, 'groups': {}}
        ev_subs = list(np.unique(g[ev])); PS = {}
        for n_, p in S.items():
            r = R['methods'][n_] = {s: metrics(y, g, p, M[s]) for s in M}
            PS[n_] = {s: metrics(y, g, p, ev & (g == s))['subject_macro_bACC'] for s in ev_subs}
            ba = np.array([q for q in PS[n_].values() if q is not None])
            R['subjects'][n_] = {'n_ge_0.9': int((ba >= .9).sum()), 'max': float(ba.max()), 'n': int(len(ba))}
            print(f'[result stream {sd}] {n_:22s} ' + ' | '.join(
                f'{s}: AUC={r[s]["AUC"]:.4f} mac={r[s]["subject_macro_bACC"]:.4f} mic={r[s]["micro_bACC"]:.4f}'
                for s in M) + f' | subj>=.9={R["subjects"][n_]["n_ge_0.9"]} max={R["subjects"][n_]["max"]:.3f}',
                flush=True)
        for suf in ('', '_thr'):
            ref = v2[f'stream{sd}']['methods'][f'fuse_pseudo_lr_full{suf}']['all']['subject_macro_bACC']
            got = R['methods'][f'fuse_{REF}{suf}']['all']['subject_macro_bACC']
            assert abs(got - ref) < 1e-9, ('v2 reproduction', suf, got, ref)
            checks[f's{sd}_v2_reproduced{suf}'] = ref
        print(f'[check s{sd}] fuse_{REF}(_thr) reproduces v2 fuse_pseudo_lr_full(_thr) mac='
              f'{checks[f"s{sd}_v2_reproduced"]:.4f}/{checks[f"s{sd}_v2_reproduced_thr"]:.4f}', flush=True)
        for n_ in S:
            R['groups'][n_] = {}
            for G in GROUPS:
                ss = [s for s in ev_subs if grp_of.get(s) == G and PS[n_][s] is not None]
                if not ss: continue
                a = np.array([PS[n_][s] for s in ss]); b = np.array([PS[f'fuse_{REF}_thr'][s] for s in ss])
                R['groups'][n_][G] = {'n': len(ss), 'mean': float(a.mean()), 'median': float(np.median(a)),
                                      'd_mean_vs_ref_thr': float(a.mean() - b.mean())}
            print(f'[group stream {sd}] {n_:22s} ' + ' | '.join(
                f'{G}(n={q["n"]}): mean={q["mean"]:.4f} dmean_vs_ref_thr={q["d_mean_vs_ref_thr"]:+.4f}'
                for G, q in R['groups'][n_].items()), flush=True)
        pairs = [(f'fuse_{REF}', 'fuseH_wf'), (f'fuse_{REF}_thr', 'fuseH_wf_thr')]
        pairs += [(f'fuse_{v}{suf}', f'fuse_{REF}{suf}') for v in VARIANTS if v != REF for suf in ('', '_thr')]
        for a_, b_ in pairs:
            for seg in M:
                bb = R['bootstrap'][f'{a_}__vs__{b_}__{seg}'] = boot(S[a_], S[b_], y, g, M[seg], nb)
                print(f'[boot stream {sd}] {a_:22s} vs {b_:20s} {seg:6s} '
                      f'dAUC={bb["dAUC"]["mean"]:+.4f}[{bb["dAUC"]["p2.5"]:+.4f},{bb["dAUC"]["p97.5"]:+.4f}] '
                      f'dmac={bb["dmacro_bACC"]["mean"]:+.4f}[{bb["dmacro_bACC"]["p2.5"]:+.4f},{bb["dmacro_bACC"]["p97.5"]:+.4f}] '
                      f'dmic={bb["dmicro_bACC"]["mean"]:+.4f}[{bb["dmicro_bACC"]["p2.5"]:+.4f},{bb["dmicro_bACC"]["p97.5"]:+.4f}]',
                      flush=True)
        print(f'[s{sd}] done {el()}', flush=True)
        (OUT / (stem + '.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    # ---- summary (mean over streams)
    st = [result[f'stream{sd}'] for sd in sseeds]; summ = {}
    for v in VARIANTS:
        n_ = f'fuse_{v}_thr'; q = {'mac': float(np.mean([r['methods'][n_]['all']['subject_macro_bACC'] for r in st])),
                                   'AUC': float(np.mean([r['methods'][n_]['all']['AUC'] for r in st])),
                                   'mature_mac': float(np.mean([r['methods'][n_]['mature']['subject_macro_bACC'] for r in st])),
                                   'past_auc_low': float(np.mean([r['past_auc'][v].get('low', np.nan) for r in st]))}
        if v != REF:
            bk = f'{n_}__vs__fuse_{REF}_thr__all'
            for m_ in ('dmacro_bACC', 'dAUC'):
                q[m_] = [float(np.mean([r['bootstrap'][bk][m_][p] for r in st])) for p in ('mean', 'p2.5', 'p97.5')]
        summ[v] = q
        print(f'[summary] {n_:22s} mac={q["mac"]:.4f} AUC={q["AUC"]:.4f} mature_mac={q["mature_mac"]:.4f} '
              f'pastAUC_low={q["past_auc_low"]:.4f}' + (f' dmac={q["dmacro_bACC"]} dAUC={q["dAUC"]}' if v != REF else ''),
              flush=True)
    for kx in ACC_KEYS:
        summ[f'label_acc_{kx}'] = {G: float(np.mean([r['label_acc'][kx][G] for r in st if G in r['label_acc'][kx]]))
                                   for G in ('all',) + GROUPS if any(G in r['label_acc'][kx] for r in st)}
        print(f'[summary] label_acc {kx:12s} {summ[f"label_acc_{kx}"]}', flush=True)
    result['summary'] = summ; result['seconds'] = time.perf_counter() - T0
    (OUT / (stem + '.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[done] all checks passed {el()} -> {stem}.json', flush=True)
    sys.stdout = tee.stream; tee.f.close()


if __name__ == '__main__':
    main()
