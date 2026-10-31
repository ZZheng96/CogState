"""Idea v2: causal WITHIN-subject adaptation on top of fuseH_wf, with clipped z-scores and better pseudo-label sources.

Same protocol as depositioned_within_subject_adapt.py (v1; functions lda / pseudo_labels imported): base score per stream
sd: branches a = logit(A['stream{sd}__all_prank']), w = logit(B['stream{sd}__resnet_whiten_logit_delta']),
f = logit(B['stream{sd}__resnet_fbank_logit_delta']); L = mean(a, w, f); fuseH_wf = sig(L). Folds = SGKF(5, shuffle, rs0)
on seed-0 'independent' arrangement, asserted == npz fold; ZZ (889) = train-median imputation + StandardScaler(fold train
rows). Per test subject in stream order, refit points t0 = 48, 64, ... (R=16, MIN=48): models fitted on rows [0, t0) score
rows t0..t0+15; rows < 48 -> NaN -> fallback (fusion = L). Representation: StandardScaler(past ZZ) -> PCA(16, full).

Score normalisation (ALL variants): z = clip((s - mean(s_past)) / max(std(s_past), 1e-6), -3, 3), s_past = the same
model's scores on its training (past) rows. Fusion fuse_X = sig(L + z_X), threshold .5; fuse_X_thr = thr_mean256 of the
fused logit (apply(l, order, 256, 'mean') of depositioned_adaptive_threshold.py). X (alone) = sig(z_X).
Variants (fixed hyper-parameters):
  pseudo_lda_v1    reference: v1 pseudo_lda, UNCLIPPED z (recomputed; same model as pseudo_lda_c).
  pseudo_lda_c     pseudo labels = top / bottom 30% of past L (stable argsort, top=1); LDA(lsqr, shrinkage auto) on PCA16.
  pseudo_iter      round1 = pseudo_lda_c model -> z1 (clipped) on the past rows; new pseudo labels = top / bottom 30% of
                   (L + z1) on past rows; refit LDA -> z2 (clipped) = output.
  pseudo_agree     pseudo labels from past rows where a, w, f each lie in their own past top 40% (task) / each in bottom
                   40% (relax) (q = floor(.4 n), stable argsort); need >= 8 per class, else fall back to pseudo_lda_c labels
                   at that refit point (fallbacks counted). LDA on PCA16.
  pseudo_lr_full   pseudo_lda_c labels; LogisticRegression(C=.01, lbfgs, max_iter 1000) on past-scaled ZZ (889).
  oracle_lr_full_c LEAKY: TRUE past labels (both classes >= 8 else NaN); LogisticRegression(C=.01) on past-scaled ZZ.
Diagnosis only: pseudo-label accuracy vs TRUE labels per refit point -> mean per subject -> mean overall / per group.
Evaluation: all / mature (t>=120); AUC, subject-macro bACC, micro bACC; #subjects bACC>=.9, max; paired subject bootstrap
(1000; smoke 100) of every output vs fuseH_wf (and fuse_X_thr vs fuseH_wf_thr); groups high/mid/low from
depositioned_subject_diagnosis.json (reporting only).
Checks (assert): fold == npz; prefix invariance (future ZZ and branch logits perturbed) for every variant on 2 subjects;
test-label shuffle -> legal variants identical; subject isolation (alone == parallel pooled); naive reference (manual
re-implementation of every variant at one refit point); pseudo_lda_v1 fuse macro bACC == v1 json.
Parallel over subjects with joblib (loky, 8 jobs, 1 BLAS thread each; deterministic).
Run: python test/depositioned_within_subject_adapt_v2.py [--smoke]   (smoke: stream0, fold0, bootstrap 100)
"""
import argparse, json, sys, time, warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
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
from depositioned_within_subject_adapt import lda, pseudo_labels, GROUPS, MATURE, STREAM_SEEDS, THR_K, R_STEP, \
    MIN_PAST, MIN_CLS, NPC, PSEUDO_Q  # noqa: E402

CLIP, EPS, AGREE_Q, N_JOBS = 3., 1e-6, .4, 8
VARIANTS = ('pseudo_lda_v1', 'pseudo_lda_c', 'pseudo_iter', 'pseudo_agree', 'pseudo_lr_full', 'oracle_lr_full_c')
LEAKY = ('oracle_lr_full_c',)
LEGAL = tuple(v for v in VARIANTS if v not in LEAKY)
ACC_KEYS = ('pseudo_lda_c', 'pseudo_iter', 'pseudo_agree', 'pseudo_agree_nofb')
T0 = time.perf_counter()


def el():
    return f'elapsed={time.perf_counter() - T0:.1f}s'


def zc(sp, sn):
    return np.clip((sn - sp.mean()) / max(sp.std(), EPS), -CLIP, CLIP)


def zraw(sp, sn):
    sd = sp.std()
    return (sn - sp.mean()) / sd if sd > 0 else np.full(len(sn), np.nan)


def agree_labels(Bp):
    """Bp (n, 3) past branch logits -> (idx, labels) or None if < MIN_CLS per class."""
    n = len(Bp); q = int(np.floor(AGREE_Q * n))
    top = np.ones(n, bool); bot = np.ones(n, bool)
    for b in range(Bp.shape[1]):
        o = np.argsort(Bp[:, b], kind='stable'); m1 = np.zeros(n, bool); m0 = np.zeros(n, bool)
        m1[o[n - q:]] = True; m0[o[:q]] = True; top &= m1; bot &= m0
    i1, i0 = np.flatnonzero(top), np.flatnonzero(bot)
    if min(len(i1), len(i0)) < MIN_CLS: return None
    return np.r_[i0, i1], np.r_[np.zeros(len(i0), int), np.ones(len(i1), int)]


def lr():
    return LogisticRegression(C=.01, max_iter=1000)


def refit(zp, Bp, yp, zn):
    """One refit point: past ZZ, past branch logits (n,3), past true labels, new ZZ -> (z per variant, diag)."""
    out, dg = {}, {}
    lp = Bp.mean(1)
    sc = StandardScaler().fit(zp); xp, xn = sc.transform(zp), sc.transform(zn)
    pca = PCA(NPC, svd_solver='full').fit(xp); pp, pn = pca.transform(xp), pca.transform(xn)
    idx, yl = pseudo_labels(lp); dg['pseudo_lda_c'] = float((yp[idx] == yl).mean())
    m = lda().fit(pp[idx], yl); sp, sn = m.decision_function(pp), m.decision_function(pn)
    out['pseudo_lda_v1'] = zraw(sp, sn); out['pseudo_lda_c'] = zc(sp, sn)
    i2, y2 = pseudo_labels(lp + zc(sp, sp)); dg['pseudo_iter'] = float((yp[i2] == y2).mean())
    m2 = lda().fit(pp[i2], y2); out['pseudo_iter'] = zc(m2.decision_function(pp), m2.decision_function(pn))
    ag = agree_labels(Bp); dg['fallback'] = ag is None
    if ag is None: ia, ya = idx, yl
    else: ia, ya = ag; dg['pseudo_agree_nofb'] = float((yp[ia] == ya).mean())
    dg['pseudo_agree'] = float((yp[ia] == ya).mean()); dg['n_agree'] = int(len(ia))
    m3 = lda().fit(pp[ia], ya); out['pseudo_agree'] = zc(m3.decision_function(pp), m3.decision_function(pn))
    m4 = lr().fit(xp[idx], yl); out['pseudo_lr_full'] = zc(m4.decision_function(xp), m4.decision_function(xn))
    if min((yp == 0).sum(), (yp == 1).sum()) >= MIN_CLS:
        m5 = lr().fit(xp, yp); out['oracle_lr_full_c'] = zc(m5.decision_function(xp), m5.decision_function(xn))
    else: out['oracle_lr_full_c'] = np.full(len(zn), np.nan)
    return out, dg


def adapt_subject(Z, Bl, y):
    """One subject in stream order: Z (n, 889), Bl (n, 3) branch logits, y (n,) -> ({variant: z}, [diag per refit])."""
    with threadpool_limits(limits=1):
        n = len(y); res = {v: np.full(n, np.nan) for v in VARIANTS}; diags = []
        for t0 in range(MIN_PAST, n, R_STEP):
            t1 = min(n, t0 + R_STEP)
            o, dg = refit(Z[:t0], Bl[:t0], y[:t0], Z[t0:t1]); diags.append(dg)
            for v in VARIANTS: res[v][t0:t1] = o[v]
    return res, diags


def naive_check(Z0, B0, y0, r0):
    """Independent re-implementation of every variant at refit point t0 = MIN_PAST + 2R."""
    n = len(y0); t0 = MIN_PAST + R_STEP * 2; t1 = min(n, t0 + R_STEP)
    mu, sd = Z0[:t0].mean(0), Z0[:t0].std(0); sd[sd == 0] = 1.
    Xp, Xn = (Z0[:t0] - mu) / sd, (Z0[t0:t1] - mu) / sd
    pc = PCA(NPC, svd_solver='full').fit(Xp); P, Pn = pc.transform(Xp), pc.transform(Xn)
    L0 = B0[:t0].mean(1); q = int(PSEUDO_Q * t0)

    def top_bot(v):
        o = np.argsort(v, kind='stable'); return np.r_[o[:q], o[t0 - q:]], np.r_[np.zeros(q), np.ones(q)]

    def z(m, A, An):
        a, b = m.decision_function(A), m.decision_function(An)
        return np.clip((b - a.mean()) / max(a.std(), 1e-6), -3, 3), a

    i, yl = top_bot(L0); m = lda().fit(P[i], yl); zz, a = z(m, P, Pn)
    assert np.allclose(r0['pseudo_lda_c'][t0:t1], zz, atol=1e-8)
    assert np.allclose(r0['pseudo_lda_v1'][t0:t1], (m.decision_function(Pn) - a.mean()) / a.std(), atol=1e-8)
    z1 = np.clip((a - a.mean()) / max(a.std(), 1e-6), -3, 3); i2, y2 = top_bot(L0 + z1)
    assert np.allclose(r0['pseudo_iter'][t0:t1], z(lda().fit(P[i2], y2), P, Pn)[0], atol=1e-8)
    qa = int(.4 * t0); ranks = [np.argsort(np.argsort(B0[:t0, b], kind='stable'), kind='stable') for b in range(3)]
    tk = np.all([r >= t0 - qa for r in ranks], 0); rx = np.all([r < qa for r in ranks], 0)
    if min(tk.sum(), rx.sum()) >= 8: ia = np.r_[np.flatnonzero(rx), np.flatnonzero(tk)]; ya = np.r_[np.zeros(rx.sum()), np.ones(tk.sum())]
    else: ia, ya = i, yl
    assert np.allclose(r0['pseudo_agree'][t0:t1], z(lda().fit(P[ia], ya), P, Pn)[0], atol=1e-8)
    # lbfgs stops at tol 1e-4, so tiny scaling differences (manual vs StandardScaler) move the optimum slightly
    d4 = np.abs(r0['pseudo_lr_full'][t0:t1] - z(lr().fit(Xp[i], yl), Xp, Xn)[0]).max()
    d5 = np.abs(r0['oracle_lr_full_c'][t0:t1] - z(lr().fit(Xp, y0[:t0]), Xp, Xn)[0]).max()
    print(f'[naive] lr max|diff| pseudo_lr_full={d4:.2e} oracle_lr_full_c={d5:.2e}', flush=True)
    assert d4 < 1e-2 and d5 < 1e-2, (d4, d5)
    for v in VARIANTS: assert np.isnan(r0[v][:MIN_PAST]).all()


def main():
    warnings.filterwarnings('ignore', category=FutureWarning)
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); args = ap.parse_args()
    sfx = '_smoke' if args.smoke else ''
    stem = 'depositioned_within_subject_adapt_v2' + sfx
    tee = Tee(sys.stdout, OUT / (stem + '.log')); sys.stdout = tee
    sseeds = STREAM_SEEDS[:1] if args.smoke else STREAM_SEEDS
    nb = 100 if args.smoke else 1000
    v1 = json.loads((OUT / f'depositioned_within_subject_adapt{sfx}.json').read_text(encoding='utf-8'))
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
    print(f'[start] rows={len(y)} subjects={len(subs)} smoke={args.smoke} sseeds={sseeds} R={R_STEP} MIN={MIN_PAST} '
          f'PCA={NPC} clip=+-{CLIP} agree_q={AGREE_Q} jobs={N_JOBS} (oracle_lr_full_c LEAKY)', flush=True)

    Xa, ya, ga, _ = arrange(subs, data, 0, 'independent'); fold_of = {}
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Xa, ya, ga)):
        for s in np.unique(ga[te]): fold_of[s] = k
    fold = np.array([fold_of[s] for s in g]); assert np.array_equal(fold, A_['fold'])
    print('[check] fold == depositioned_aperiodic_emg_oof.npz fold', flush=True)
    run_folds = [0] if args.smoke else list(range(5))
    ZZt = np.full((len(y), 889), np.nan)
    for k in run_folds:
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        tfo, tfn = impute_scale(Fc, tr), impute_scale(Fn, tr)
        ZZt[te] = np.hstack([tfo(Fc[te]), tfn(Fn[te])])
    print(f'[zz] test-side ZZ built for folds {run_folds} {el()}', flush=True)

    result = {'meta': {'leaky_variants': list(LEAKY), 'legal_variants': list(LEGAL), 'refit_step': R_STEP,
                       'min_past': MIN_PAST, 'min_per_class': MIN_CLS, 'pca': NPC, 'pseudo_q': PSEUDO_Q,
                       'agree_q': AGREE_Q, 'clip': CLIP, 'eps': EPS, 'mature_index': MATURE, 'threshold': .5,
                       'thr': {'k': THR_K, 'min_past': 16, 'mode': 'mean'}, 'bootstrap_n': nb,
                       'stream_seeds': list(sseeds), 'smoke': args.smoke}, 'checks': {}}
    checks = result['checks']; sidx = {s: j for j, s in enumerate(subs)}
    run_subs = [s for s in subs if fold_of[s] in run_folds]
    for sd in sseeds:
        order, t = stream_order(g, subs, sd)
        Bl = np.c_[logit(A_[f'stream{sd}__all_prank']), logit(B_[f'stream{sd}__resnet_whiten_logit_delta']),
                   logit(B_[f'stream{sd}__resnet_fbank_logit_delta'])]
        L = Bl.mean(1); assert np.isfinite(L).all()
        ev = np.isin(fold, run_folds)
        print(f'[s{sd}] std(L)={L[ev].std():.3f} mean(L)={L[ev].mean():+.3f}', flush=True)
        Zs = {v: np.full(len(y), np.nan) for v in VARIANTS}; DG = {}
        ts = time.perf_counter()
        gen = Parallel(n_jobs=N_JOBS, backend='loky', return_as='generator')(
            delayed(adapt_subject)(ZZt[order[sidx[s]]], Bl[order[sidx[s]]], y[order[sidx[s]]]) for s in run_subs)
        for i, (s, (r, dg)) in enumerate(zip(run_subs, gen)):
            loc = order[sidx[s]]; DG[s] = dg
            for v in VARIANTS: Zs[v][loc] = r[v]
            if (i + 1) % 8 == 0 or i + 1 == len(run_subs):
                print(f'[s{sd}] subjects {i + 1}/{len(run_subs)} stream_s={time.perf_counter() - ts:.1f} {el()}', flush=True)
        # ---- checks
        for s in run_subs[:2]:
            loc = order[sidx[s]]; Z0, B0, y0 = ZZt[loc], Bl[loc], y[loc]; n = len(loc)
            r0, _ = adapt_subject(Z0, B0, y0)
            for v in VARIANTS: assert np.array_equal(r0[v], Zs[v][loc], equal_nan=True), ('isolation', s, v)
            cut = MIN_PAST + R_STEP * ((n - MIN_PAST) // (2 * R_STEP)) + 5
            rng = np.random.default_rng(sidx[s]); Z1, B1 = Z0.copy(), B0.copy()
            Z1[cut:] += rng.normal(0, 2, Z1[cut:].shape); B1[cut:] += rng.normal(0, 2, B1[cut:].shape)
            r1, _ = adapt_subject(Z1, B1, y0)
            for v in VARIANTS:
                assert np.array_equal(r0[v][:cut], r1[v][:cut], equal_nan=True), ('prefix', s, v)
                a, b = r0[v][cut:], r1[v][cut:]; f = np.isfinite(a) & np.isfinite(b)
                assert f.any() and np.abs(a[f] - b[f]).max() > 0, ('prefix-changed', s, v)
            r2, _ = adapt_subject(Z0, B0, np.random.default_rng(1).permutation(y0)); dsh = {}
            for v in VARIANTS:
                dsh[v] = bool(np.array_equal(r0[v], r2[v], equal_nan=True))
                if v in LEGAL: assert dsh[v], ('label_shuffle', s, v)
            naive_check(Z0, B0, y0, r0)
            checks[f's{sd}_{s}'] = {'isolation': 'ok', 'prefix_invariance': 'ok', 'naive_ref': 'ok', 'cut': int(cut),
                                    'label_shuffle_identical': dsh}
            print(f'[check s{sd}] {s}: isolation / prefix / naive ref OK; label shuffle identical: {dsh}', flush=True)
        # ---- pseudo-label accuracy (diagnosis) and fallback
        acc = {}
        for kx in ACC_KEYS:
            per = {s: float(np.mean(q)) for s in run_subs if (q := [d[kx] for d in DG[s] if kx in d])}
            acc[kx] = {'all': float(np.mean(list(per.values()))), 'n_subjects': len(per)}
            for G in GROUPS:
                a = [per[s] for s in per if grp_of[s] == G]
                if a: acc[kx][G] = float(np.mean(a))
        nref = sum(len(DG[s]) for s in run_subs); nfb = sum(d['fallback'] for s in run_subs for d in DG[s])
        nag = [d['n_agree'] for s in run_subs for d in DG[s] if not d['fallback']]
        fb = {'refits': nref, 'fallbacks': int(nfb), 'rate': nfb / nref,
              'subjects_all_fallback': int(sum(all(d['fallback'] for d in DG[s]) for s in run_subs)),
              'median_n_agree': float(np.median(nag)) if nag else None}
        for kx, q in acc.items():
            print(f'[pseudo-acc s{sd}] {kx:18s} all={q["all"]:.4f} ' +
                  ' '.join(f'{G}={q[G]:.4f}' for G in GROUPS if G in q) + f' (n_subj={q["n_subjects"]})', flush=True)
        print(f'[agree s{sd}] fallback {nfb}/{nref} ({fb["rate"]:.3f}); subjects always fallback='
              f'{fb["subjects_all_fallback"]}; median agreed rows={fb["median_n_agree"]}', flush=True)
        # ---- scores
        S = {'fuseH_wf': sig(L), 'fuseH_wf_thr': sig(thr_apply(np.where(ev, L, np.nan), order, THR_K, 'mean'))}
        zinfo = {}
        for v in VARIANTS:
            z = Zs[v]; f = np.isfinite(z); zf = z[ev & f]
            zinfo[v] = {'coverage': float(f[ev].mean()), 'std': float(zf.std()),
                        'pct': {p: float(np.percentile(zf, p)) for p in (1, 5, 50, 95, 99)},
                        'frac_clipped': float((np.abs(zf) >= CLIP).mean())}
            S[v] = sig(np.where(f, z, L))
            lf = np.where(ev, np.where(f, L + z, L), np.nan)
            S[f'fuse_{v}'] = sig(lf); S[f'fuse_{v}_thr'] = sig(thr_apply(lf, order, THR_K, 'mean'))
            zi = zinfo[v]
            print(f'[z s{sd}] {v:16s} coverage={zi["coverage"]:.3f} std={zi["std"]:.3g} p1/5/50/95/99=' +
                  '/'.join(f'{zi["pct"][p]:+.2f}' for p in (1, 5, 50, 95, 99)) + f' clipped={zi["frac_clipped"]:.3f}',
                  flush=True)
        for n_ in S: S[n_] = np.where(ev, S[n_], np.nan)
        M = {'all': ev, 'mature': ev & (t >= MATURE)}
        R = result[f'stream{sd}'] = {'std_L': float(L[ev].std()), 'z_info': zinfo, 'pseudo_acc': acc, 'agree_fallback': fb,
                                     'methods': {}, 'bootstrap': {}, 'subjects': {}, 'groups': {}}
        ev_subs = list(np.unique(g[ev])); PS = {}
        for n_, p in S.items():
            r = R['methods'][n_] = {s: metrics(y, g, p, M[s]) for s in M}
            PS[n_] = {s: metrics(y, g, p, ev & (g == s))['subject_macro_bACC'] for s in ev_subs}
            ba = np.array([q for q in PS[n_].values() if q is not None])
            R['subjects'][n_] = {'n_ge_0.9': int((ba >= .9).sum()), 'max': float(ba.max()), 'n': int(len(ba))}
            tag = ' [LEAKY]' if any(v in n_ for v in LEAKY) else ''
            print(f'[result stream {sd}] {n_:24s} ' + ' | '.join(
                f'{s}: AUC={r[s]["AUC"]:.4f} mac={r[s]["subject_macro_bACC"]:.4f} mic={r[s]["micro_bACC"]:.4f}'
                for s in M) + f' | subj>=.9={R["subjects"][n_]["n_ge_0.9"]} max={R["subjects"][n_]["max"]:.3f}{tag}',
                flush=True)
        ref = v1[f'stream{sd}']['methods']['fuse_pseudo_lda']['all']['subject_macro_bACC']
        assert abs(R['methods']['fuse_pseudo_lda_v1']['all']['subject_macro_bACC'] - ref) < 1e-9, 'v1 reproduction'
        checks[f's{sd}_v1_reproduced'] = ref
        print(f'[check s{sd}] fuse_pseudo_lda_v1 reproduces v1 fuse_pseudo_lda mac={ref:.4f}', flush=True)
        for n_ in S:
            R['groups'][n_] = {}
            for G in GROUPS:
                ss = [s for s in ev_subs if grp_of.get(s) == G and PS[n_][s] is not None]
                if not ss: continue
                a = np.array([PS[n_][s] for s in ss]); b = np.array([PS['fuseH_wf'][s] for s in ss])
                R['groups'][n_][G] = {'n': len(ss), 'median': float(np.median(a)), 'mean': float(a.mean()),
                                      'd_median_vs_fuseH_wf': float(np.median(a) - np.median(b)),
                                      'd_mean_vs_fuseH_wf': float(a.mean() - b.mean())}
            print(f'[group stream {sd}] {n_:24s} ' + ' | '.join(
                f'{G}(n={q["n"]}): mean={q["mean"]:.4f} dmean={q["d_mean_vs_fuseH_wf"]:+.4f} '
                f'dmed={q["d_median_vs_fuseH_wf"]:+.4f}' for G, q in R['groups'][n_].items()), flush=True)
        pairs = [(n_, 'fuseH_wf') for n_ in S if n_ != 'fuseH_wf'] + [(f'fuse_{v}_thr', 'fuseH_wf_thr') for v in VARIANTS]
        for a_, b_ in pairs:
            for seg in M:
                bb = R['bootstrap'][f'{a_}__vs__{b_}__{seg}'] = boot(S[a_], S[b_], y, g, M[seg], nb)
                print(f'[boot stream {sd}] {a_:24s} vs {b_:12s} {seg:6s} '
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
