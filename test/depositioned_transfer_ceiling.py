"""Step-1 diagnostic (NOT deployable): where does the low group lose its separability - features, decision
direction, or within-subject data/SNR?

Everything is measured with the artifacts of the current legal pipeline (depositioned_within_subject_adapt_v2
.pseudo_lr_full + thr_mean256 = fuse_pseudo_lr_full_thr, the best legal pipeline) on the de-positioned protocol
(per-subject random stream order, 3 stream seeds, 5 subject-grouped folds). Test labels are never used by any
legal score; the leaky rows below exist only as ceilings and are flagged as such.

Ladder measured per subject ("subject" = one evaluated subject of the OOF split):
  1. pipeline        LEGAL  f1 = L + z (pseudo labels from past L, LR C=.01 on past-scaled 889 zz, refit every
                     R=16 windows from t0=48, z clipped +-3), thr_mean256 -> per-subject bACC (threshold .5 after
                     the causal shift). This is the actual cross-subject pipeline.
  2. oracle_past     LEAKY  identical causal refit, but the LR is trained on the subject's TRUE past labels
                     (= v2 oracle_lr_full_c). This is "cross-subject training + the subject's own true past labels",
                     i.e. the adaptation ceiling that v2 reported as +.10 overall / +.15 low group.
  3. ws_rand         LEAKY  same feature family (fold-standardised 889 zz) and same classifier family
                     (LogisticRegression C=.01, class_weight balanced), trained on the subject's own windows with
                     true labels, evaluated by stratified 5-fold CV WITHIN the subject (leaky in time; measures how
                     much label information the features of that subject carry at all) -> AUC_ws_rand.
  3b. ws_block       LEAKY  identical models but leave-one-state-block-out (blocks = maximal runs of consecutive
                     same-label trials in the subject's real time order, from dense_window_protocol_cache.npz mapped
                     by md5, i.e. the block definition of per_subject_ceiling.py); the held-out block is scored by a
                     model trained on the other blocks, its decision scores z-normalised with that model's own
                     training scores (same normalisation as the causal adaptation), pooled over block-folds
                     -> AUC_ws_block = temporal transfer of the subject's own boundary.
  4. xs_lr           LEGAL (labels of OTHER subjects only)  cross-subject LogisticRegression C=.01 on the 889 zz,
                     trained per fold on all other subjects' true labels, no adaptation -> AUC_xs / bACC_xs.
  5. xs_lr_oriented  LEAKY  the same xs_lr score, per subject oriented by its true labels (max(AUC, 1-AUC)).
                     Measures how much of the xs->ws gap is a single per-subject sign mismatch.
  6. xs_lr_cz        LEGAL  xs_lr on causally per-subject z-scored 889 zz (row t standardised with the expanding
                     mean/std of the subject's strictly earlier rows, >=16 past rows required, earlier rows left as
                     they are). Tests whether a legal per-subject affine re-centering can repair the transfer.

Direction consistency: w_ws = LR(C=.01, unweighted, no class weights) fitted on all of the subject's windows with
true labels; w_xs = the fold's cross-subject LR coefficient vector (same C, same standardised space). Reported per
subject: cos(w_ws, w_xs), AUC of L alone (auc_L, the pipeline's raw separability), number of windows.

Pre-registered attribution rule (fixed before the full run; applied mechanically, no post-hoc switching; AUC_ws_rand
is the feature-information ceiling, AUC_ws_block is reported separately as a temporal-stability diagnostic and is NOT
part of the label):
  (c) within-subject data/SNR-limited   if median AUC_ws_rand < .70 in the group;
  else (b) decision-direction inconsistency  if the oriented transfer recovers >= 75% of the xs->ws_rand gap, i.e.
      (AUC_xs_or - AUC_xs) / (AUC_ws_rand - AUC_xs) >= .75, and (median cos(w_ws,w_xs) <= .30 or frac(cos<0) >= .25);
  else (a) feature-level non-transferability (the information is in the subject's features, but no cross-subject
      linear map recovers it even after per-subject orientation).

Checks (assert): the legal pipeline reproduces v2's fuse_pseudo_lr_full_thr subject-macro bACC (1e-9) and the group
means; the oracle reproduces v2's fuse_oracle_lr_full_c_thr (1e-9); future-row perturbation leaves every score before
the cut bit-identical (legal AND oracle); shuffling test labels leaves the legal score bit-identical (and changes the
oracle); subject isolation (subject alone == pooled parallel). The order-free ladder (xs_lr, xs_lr_cz, ws_rand, ws_block,
direction cosine) does not use the stream order at all; the per-subject AUC of L is reported for stream seed 0 (L is
order-dependent through the hgb prank / ResNet history-delta branches).
Parallel over subjects with joblib (threading backend, 8 jobs, 1 BLAS thread per job; deterministic).
Run: python test/depositioned_transfer_ceiling.py [--smoke]   (smoke: stream0, fold0 only)
"""
import argparse, hashlib, json, sys, time, warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from depositioned_history_delta import arrange, load_db  # noqa: E402
from depositioned_resnet_history_delta import load_raw, stream_order, logit, sig  # noqa: E402
from depositioned_adaptive_threshold import apply as thr_apply  # noqa: E402
from depositioned_symmetric_contrast import impute_scale  # noqa: E402
from causal_kernel_state_mapping import metrics  # noqa: E402
from switch_dual_memory import Tee  # noqa: E402
from depositioned_within_subject_adapt import pseudo_labels, GROUPS, MATURE, STREAM_SEEDS, THR_K, R_STEP, MIN_PAST, \
    MIN_CLS  # noqa: E402

CLIP, EPS, N_JOBS, C_REG, ZMIN_PAST = 3., 1e-6, 8, .01, 16
T0 = time.perf_counter()


def el():
    return f'elapsed={time.perf_counter() - T0:.1f}s'


def zc(sp, sn):
    return np.clip((sn - sp.mean()) / max(sp.std(), EPS), -CLIP, CLIP)


def lr(balanced=False):
    return LogisticRegression(C=C_REG, max_iter=1000, class_weight='balanced' if balanced else None)


def adapt_subject(ZZ, Bl, y):
    """Legal pseudo-label adaptation and leaky true-past-label adaptation for one subject (stream order)."""
    with threadpool_limits(limits=1):
        n = len(Bl); L = Bl.mean(1); z1 = np.full(n, np.nan); zo = np.full(n, np.nan)
        for t0 in range(MIN_PAST, n, R_STEP):
            t1 = min(n, t0 + R_STEP)
            sc = StandardScaler().fit(ZZ[:t0]); xp, xn = sc.transform(ZZ[:t0]), sc.transform(ZZ[t0:t1])
            i, yl = pseudo_labels(L[:t0])
            m = lr().fit(xp[i], yl); z1[t0:t1] = zc(m.decision_function(xp), m.decision_function(xn))
            if min((y[:t0] == 0).sum(), (y[:t0] == 1).sum()) >= MIN_CLS:
                mo = lr().fit(xp, y[:t0]); zo[t0:t1] = zc(mo.decision_function(xp), mo.decision_function(xn))
        return L, L + np.where(np.isfinite(z1), z1, 0.), L + np.where(np.isfinite(zo), zo, 0.)


def blocks_from_cache(y, g, Xraw):
    """State blocks per depositioned row: block id = maximal run of consecutive same-label trials in the subject's
    real time order (highdim order == time order, trial id non-decreasing per subject), mapped onto the depositioned
    rows by the (subject, label, md5(raw float32)) multiset alignment used by depositioned_final_merge.py."""
    hd = np.load(OUT / 'highdim_deep_features.npz')
    raw, yh, gh = hd['raw'], hd['y'].astype(int), hd['groups'].astype(str)
    c = np.load(OUT / 'dense_window_protocol_cache.npz'); ci = np.where(c['canon'])[0]
    assert np.array_equal(c['label'][ci].astype(int), yh) and np.array_equal(c['subject'][ci], gh)
    trial = c['trial'][ci]
    block_hd = np.full(len(yh), -1); nb = 0
    for s in np.unique(gh):
        ii = np.flatnonzero(gh == s)
        assert (np.diff(ii) == 1).all() and (np.diff(trial[ii]) >= 0).all()
        b = np.r_[0, np.cumsum(yh[ii][1:] != yh[ii][:-1])]
        block_hd[ii] = nb + b; nb += b[-1] + 1
    assert nb == 339
    key = defaultdict(list)
    for i in range(len(yh)):
        key[(gh[i], int(yh[i]), hashlib.md5(raw[i].astype(np.float32).tobytes()).hexdigest())].append(i)
    idx = np.empty(len(y), int)
    for i in range(len(y)):
        idx[i] = key[(g[i][4:], int(y[i]), hashlib.md5(Xraw[i].astype(np.float32).tobytes()).hexdigest())].pop(0)
    assert len(np.unique(idx)) == len(y) == len(yh) and sum(len(v) for v in key.values()) == 0
    return block_hd[idx]


def ws_ceilings(Zs, ys, blk):
    """Within-subject ceilings for one subject:
    rand  = stratified 5-fold CV (leaky in time; label information the features carry at all);
    block = leave-one-state-block-out with per-model z-normalisation of the held-out scores (temporal transfer).
    Returns (bacc_rand, auc_rand, bacc_block, auc_block, block_coverage)."""
    with threadpool_limits(limits=1):
        pr = np.full(len(ys), np.nan)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Zs, ys):
            if len(np.unique(ys[tr])) < 2: continue
            pr[te] = lr(True).fit(Zs[tr], ys[tr]).predict_proba(Zs[te])[:, 1]
        okr = np.isfinite(pr)
        br = ar = None
        if okr.sum() >= 20 and len(np.unique(ys[okr])) == 2:
            br = float(balanced_accuracy_score(ys[okr], pr[okr] >= .5))
            ar = float(roc_auc_score(ys[okr], pr[okr]))
        pb = np.full(len(ys), np.nan)
        for b in np.unique(blk):
            te = np.flatnonzero(blk == b); tr = np.flatnonzero(blk != b)
            if len(np.unique(ys[tr])) < 2: continue
            m = lr(True).fit(Zs[tr], ys[tr])
            sp, sn = m.decision_function(Zs[tr]), m.decision_function(Zs[te])
            pb[te] = (sn - sp.mean()) / max(sp.std(), EPS)
    ok = np.isfinite(pb)
    bb = ab = None
    if ok.sum() >= 20 and len(np.unique(ys[ok])) == 2:
        bb = float(balanced_accuracy_score(ys[ok], pb[ok] >= 0))
        ab = float(roc_auc_score(ys[ok], pb[ok]))
    return br, ar, bb, ab, float(ok.mean())


def wz(Zs, ys):
    with threadpool_limits(limits=1):
        return lr().fit(Zs, ys).coef_[0]


def causal_z(Zs):
    """Row t standardised with the expanding mean/std of the strictly earlier rows (>=ZMIN_PAST required)."""
    n = len(Zs); out = Zs.copy()
    c1, c2 = np.cumsum(Zs, 0), np.cumsum(Zs * Zs, 0)
    for t in range(ZMIN_PAST, n):
        mean = c1[t - 1] / t
        sd = np.sqrt(np.maximum(c2[t - 1] / t - mean ** 2, 0.)); sd[sd < 1e-8] = 1.
        out[t] = (Zs[t] - mean) / sd
    return out


def causal_z_all(Zfull, order):
    o = Zfull.copy()
    for loc in order:
        o[loc] = causal_z(Zfull[loc])
    return o


def per_subject_bacc(y, g, p, subs):
    return {s: metrics(y, g, p, g == s)['subject_macro_bACC'] for s in subs}


def per_subject_auc(y, g, p, subs):
    out = {}
    for s in subs:
        q = g == s; yy, pp = y[q], p[q]; ok = np.isfinite(pp)
        out[s] = float(roc_auc_score(yy[ok], pp[ok])) if ok.sum() and np.unique(yy[ok]).size == 2 else None
    return out


def main():
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); args = ap.parse_args()
    stem = 'depositioned_transfer_ceiling' + ('_smoke' if args.smoke else '')
    tee = Tee(sys.stdout, OUT / (stem + '.log')); sys.stdout = tee
    sseeds = STREAM_SEEDS[:1] if args.smoke else STREAM_SEEDS

    man, subs, data = load_db()
    Fc = np.vstack([data[s][0] for s in subs]); y = np.concatenate([data[s][1] for s in subs])
    g = np.concatenate([np.full(len(data[s][1]), s) for s in subs])
    A_ = np.load(OUT / 'depositioned_aperiodic_emg_oof.npz'); B_ = np.load(OUT / 'depositioned_resnet_timefreq_oof.npz')
    assert np.array_equal(A_['y'], y) and np.array_equal(A_['groups'], g)
    for q in ('y', 'groups', 'fold'): assert np.array_equal(A_[q], B_[q]), q
    Fn = A_['new_features']; assert Fc.shape[1] == 858 and Fn.shape == (len(y), 31)
    fold = A_['fold']; n = len(y)
    diag = json.loads((OUT / 'depositioned_subject_diagnosis.json').read_text(encoding='utf-8'))
    grp_of = {s: G for G in GROUPS for s in diag['diagnosis']['groups'][G]['subjects']}
    assert all(s in grp_of for s in subs)
    Xraw, Fc2, y2, g2 = load_raw(subs)
    assert np.array_equal(y2, y) and np.array_equal(g2, g) and np.array_equal(Fc2, Fc)
    block = blocks_from_cache(y, g, Xraw)
    del Xraw
    v2 = json.loads((OUT / 'depositioned_within_subject_adapt_v2.json').read_text(encoding='utf-8'))

    Xa, ya, ga, _ = arrange(subs, data, 0, 'independent'); fold_of = {}
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Xa, ya, ga)):
        for s in np.unique(ga[te]): fold_of[s] = k
    assert np.array_equal(np.array([fold_of[s] for s in g]), fold)
    run_folds = [0] if args.smoke else list(range(5))
    run_subs = [s for s in subs if fold_of[s] in run_folds]

    result = {'meta': {'deployable': False, 'purpose': 'transfer / adaptation ceiling ladder + a/b/c attribution',
                       'leaky_rows': ['oracle_past', 'xs_lr_oriented', 'ws_ceiling'],
                       'legal_rows': ['pipeline', 'xs_lr', 'xs_lr_cz', 'L'],
                       'refit_step': R_STEP, 'min_past': MIN_PAST, 'min_per_class': MIN_CLS, 'clip': CLIP,
                       'C': C_REG, 'thr': {'k': THR_K, 'min_past': 16, 'mode': 'mean'}, 'threshold': .5,
                       'mature_index': MATURE, 'stream_seeds': list(sseeds), 'folds': run_folds,
                       'smoke': args.smoke, 'auc_L_stream': STREAM_SEEDS[0],
                       'attribution_rule': '(c) if median AUC_ws<.70; else (b) if oriented-recovery>=.75 and '
                                           '(median cos<=.30 or frac cos<0>=.25); else (a)'},
              'checks': {}, 'streams': {}, 'order_free': {}, 'groups': {}, 'attribution': {}}
    checks = result['checks']; sidx = {s: j for j, s in enumerate(subs)}

    # ---------------- order-free ladder: xs_lr, xs_lr_cz, ws_ceiling, direction cos, auc_L ----------------
    xs = np.full(n, np.nan); xsc = np.full(n, np.nan)
    wsr, cosv, wnorm = {}, {}, {}
    print(f'[start] rows={n} subjects={len(subs)} folds={run_folds} sseeds={sseeds} {el()}', flush=True)
    order0, _ = stream_order(g, subs, 0)
    for k in run_folds:
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        tfo, tfn = impute_scale(Fc, tr), impute_scale(Fn, tr)
        Zk = np.hstack([tfo(Fc), tfn(Fn)])
        with threadpool_limits(limits=1):
            mxs = LogisticRegression(C=C_REG, max_iter=1000).fit(Zk[tr], y[tr])
        xs[te] = mxs.decision_function(Zk[te])
        Zc = causal_z_all(Zk, order0)
        with threadpool_limits(limits=1):
            mzc = LogisticRegression(C=C_REG, max_iter=1000).fit(Zc[tr], y[tr])
        xsc[te] = mzc.decision_function(Zc[te])
        ps = [s for s in np.unique(g[te])]
        res = Parallel(n_jobs=N_JOBS, backend='threading')(
            delayed(ws_ceilings)(Zk[g == s], y[g == s], block[g == s]) for s in ps)
        for s, (br, ar, bb, ab, cov) in zip(ps, res):
            wsr[s] = {'bacc_rand': br, 'auc_rand': ar, 'bacc_block': bb, 'auc_block': ab, 'block_coverage': cov}
        print(f'[fold {k}] cross-subject LR + causal-z LR + within-subject rand/block ceilings for {len(ps)} '
              f'subjects {el()}', flush=True)
        # direction cosine needs the fold's cross-subject direction (only meaningful for this fold's subjects)
        dirs = Parallel(n_jobs=N_JOBS, backend='threading')(
            delayed(wz)(Zk[g == s], y[g == s]) for s in ps)
        for s, w in zip(ps, dirs):
            cosv[s] = float(np.dot(w, mxs.coef_[0]) / (np.linalg.norm(w) * np.linalg.norm(mxs.coef_[0]) + 1e-12))
            wnorm[s] = float(np.linalg.norm(w))
    xs_or = np.full(n, np.nan)
    for s in subs:
        q = g == s; a = per_subject_auc(y, g, xs, [s])[s]
        if a is None: continue
        xs_or[q] = xs[q] if a >= .5 else -xs[q]

    # ---------------- per-stream legal / oracle adaptation ----------------
    L = None
    for sd in sseeds:
        order, t_c = stream_order(g, subs, sd)
        Bl = np.c_[logit(A_[f'stream{sd}__all_prank']), logit(B_[f'stream{sd}__resnet_whiten_logit_delta']),
                   logit(B_[f'stream{sd}__resnet_fbank_logit_delta'])]
        Ls = Bl.mean(1); assert np.isfinite(Ls).all()
        if L is None: L = Ls   # per-subject AUC of L is reported for the first stream (L is order-dependent)
        ev = np.isin(fold, run_folds)
        ZZt = np.full((n, 889), np.nan)
        for k in run_folds:
            tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
            tfo, tfn = impute_scale(Fc, tr), impute_scale(Fn, tr)
            ZZt[te] = np.hstack([tfo(Fc[te]), tfn(Fn[te])])
        f1 = np.full(n, np.nan); fo = np.full(n, np.nan)
        ts = time.perf_counter()
        gen = Parallel(n_jobs=N_JOBS, backend='threading', return_as='generator')(
            delayed(adapt_subject)(ZZt[order[sidx[s]]], Bl[order[sidx[s]]], y[order[sidx[s]]]) for s in run_subs)
        for i, (s, (_, a1, ao)) in enumerate(zip(run_subs, gen)):
            loc = order[sidx[s]]; f1[loc] = a1; fo[loc] = ao
            if (i + 1) % 20 == 0 or i + 1 == len(run_subs):
                print(f'[s{sd}] adapted {i + 1}/{len(run_subs)} {time.perf_counter() - ts:.1f}s {el()}', flush=True)
        P = {'pipeline': sig(thr_apply(np.where(ev, f1, np.nan), order, THR_K, 'mean')),
             'oracle_past': sig(thr_apply(np.where(ev, fo, np.nan), order, THR_K, 'mean')),
             'fuseH_wf': sig(Ls), 'fuseH_wf_thr': sig(thr_apply(np.where(ev, Ls, np.nan), order, THR_K, 'mean'))}
        for q in P: assert np.isfinite(P[q][ev]).all(), q
        # ---- checks
        for s in run_subs[:3]:
            loc = order[sidx[s]]; Z0, B0, y0 = ZZt[loc], Bl[loc], y[loc]; ns = len(loc)
            l0, f0, o0 = adapt_subject(Z0, B0, y0)
            assert np.array_equal(f0, f1[loc], equal_nan=True), ('isolation', s)
            assert np.array_equal(o0, fo[loc], equal_nan=True), ('isolation oracle', s)
            _, fsh, osh = adapt_subject(Z0, B0, np.random.default_rng(1).permutation(y0))
            assert np.array_equal(f0, fsh, equal_nan=True), ('label_shuffle', s)
            assert not np.array_equal(o0, osh, equal_nan=True), ('oracle uses labels', s)
            cut = MIN_PAST + R_STEP * ((ns - MIN_PAST) // (2 * R_STEP)) + 5
            rng = np.random.default_rng(int(sidx[s])); Z1, B1 = Z0.copy(), B0.copy()
            Z1[cut:] += rng.normal(0, 2, Z1[cut:].shape); B1[cut:] += rng.normal(0, 2, B1[cut:].shape)
            _, f1p, o1p = adapt_subject(Z1, B1, y0)
            for nm, a, b in (('pipeline', f0, f1p), ('oracle_past', o0, o1p)):
                assert np.array_equal(a[:cut], b[:cut], equal_nan=True), ('prefix', s, nm)
                d = np.abs(a[cut:] - b[cut:]); fin = np.isfinite(d)
                assert fin.any() and d[fin].max() > 0, ('prefix-changed', s, nm)
            checks[f's{sd}_{s}'] = {'isolation': 'ok', 'label_shuffle_legal': 'ok', 'label_shuffle_oracle_changed': 'ok',
                                    'prefix_invariance': 'ok', 'cut': int(cut)}
            print(f'[check s{sd}] {s}: isolation / label shuffle / prefix OK (cut={cut})', flush=True)
        # ---- reproduction against v2
        R = result['streams'][f'stream{sd}'] = {'methods': {}, 'subjects': {}}
        for nm, p in P.items():
            R['methods'][nm] = {sg: metrics(y, g, p, ev & (True if sg == 'all' else t_c >= MATURE)) for sg in ('all', 'mature')}
            print(f'[result s{sd}] {nm:14s} all: AUC={R["methods"][nm]["all"]["AUC"]:.4f} '
                  f'mac={R["methods"][nm]["all"]["subject_macro_bACC"]:.4f}', flush=True)
        if not args.smoke:
            for nm, key, js in (('pipeline', 'fuse_pseudo_lr_full_thr', v2), ('oracle_past', 'fuse_oracle_lr_full_c_thr', v2)):
                ref = js[f'stream{sd}']['methods'][key]['all']['subject_macro_bACC']
                got = R['methods'][nm]['all']['subject_macro_bACC']
                assert abs(got - ref) < 1e-9, ('v2 reproduction', nm, got, ref)
                checks[f's{sd}_{nm}_reproduced'] = ref
            print(f'[check s{sd}] pipeline/oracle reproduce v2 fuse_pseudo_lr_full_thr / fuse_oracle_lr_full_c_thr', flush=True)
        # per-subject values
        ev_subs = [s for s in subs if fold_of[s] in run_folds]
        pv = {nm: per_subject_bacc(y, g, p, ev_subs) for nm, p in P.items()}
        R['subjects'] = {nm: {s: v for s, v in q.items()} for nm, q in pv.items()}
        # group means should also reproduce v2's group means (same per-subject bACC, same membership)
        if not args.smoke:
            for nm, key in (('pipeline', 'fuse_pseudo_lr_full_thr'), ('oracle_past', 'fuse_oracle_lr_full_c_thr')):
                for G in GROUPS:
                    ss = [s for s in ev_subs if grp_of[s] == G and pv[nm][s] is not None]
                    got = float(np.mean([pv[nm][s] for s in ss]))
                    ref = v2[f'stream{sd}']['groups'][key][G]['mean']
                    assert abs(got - ref) < 1e-9, ('v2 group reproduction', sd, nm, G, got, ref)
            print(f'[check s{sd}] per-subject group means reproduce v2', flush=True)
        R['groups'] = {}
        for nm in pv:
            R['groups'][nm] = {}
            for G in GROUPS:
                ss = [s for s in ev_subs if grp_of[s] == G and pv[nm][s] is not None]
                if not ss: continue
                a = np.array([pv[nm][s] for s in ss])
                R['groups'][nm][G] = {'n': len(ss), 'mean': float(a.mean()), 'median': float(np.median(a))}
            print(f'[group s{sd}] {nm:14s} ' + ' | '.join(
                f'{G}(n={q["n"]}): mean={q["mean"]:.4f} med={q["median"]:.4f}' for G, q in R['groups'][nm].items()),
                flush=True)
        (OUT / (stem + '.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')

    # ---------------- order-free summary + attribution ----------------
    assert L is not None
    ev_subs = [s for s in subs if fold_of[s] in run_folds]
    auc_L = per_subject_auc(y, g, L, ev_subs)
    xs_auc = per_subject_auc(y, g, xs, ev_subs); xsc_auc = per_subject_auc(y, g, xsc, ev_subs)
    OF = result['order_free'] = {'subjects': {}, 'groups': {}, 'overall': {}}
    for s in ev_subs:
        q = g == s; aa = xs_auc[s]; w = wsr[s]
        OF['subjects'][s] = {'n_windows': int(q.sum()), 'auc_L': auc_L[s], 'auc_xs': aa,
                             'auc_xs_cz': xsc_auc[s], 'auc_xs_oriented': max(aa, 1 - aa) if aa is not None else None,
                             'auc_ws_rand': w['auc_rand'], 'bacc_ws_rand': w['bacc_rand'],
                             'auc_ws_block': w['auc_block'], 'bacc_ws_block': w['bacc_block'],
                             'ws_block_coverage': w['block_coverage'],
                             'cos_ws_xs': cosv[s], 'norm_ws': wnorm[s], 'n_blocks': int(len(np.unique(block[q])))}
    def fm(v):
        return 'nan' if v is None else f'{v:.3f}'
    def med(v): v = [x for x in v if x is not None]; return float(np.median(v)) if v else None
    FIELDS = ('auc_L', 'auc_xs', 'auc_xs_cz', 'auc_xs_oriented', 'auc_ws_rand', 'bacc_ws_rand', 'auc_ws_block',
              'bacc_ws_block', 'cos_ws_xs', 'ws_block_coverage', 'n_windows', 'n_blocks')
    for G in GROUPS:
        ss = [s for s in ev_subs if grp_of[s] == G]
        if not ss: continue
        q = OF['groups'][G] = {'n': len(ss)}
        for f_ in FIELDS:
            q[f'_median_{f_}'] = med([OF['subjects'][s][f_] for s in ss])
            vals = [OF['subjects'][s][f_] for s in ss if OF['subjects'][s][f_] is not None]
            q[f'_mean_{f_}'] = float(np.mean(vals)) if vals else None
        q['frac_cos_neg'] = float(np.mean([OF['subjects'][s]['cos_ws_xs'] < 0 for s in ss]))
        q['frac_xs_below_half'] = float(np.mean([OF['subjects'][s]['auc_xs'] < .5 for s in ss]))
        print(f'[order-free {G}] n={len(ss)} AUC_L={fm(q["_median_auc_L"])} AUC_xs={fm(q["_median_auc_xs"])} '
              f'AUC_xs_or={fm(q["_median_auc_xs_oriented"])} AUC_xs_cz={fm(q["_median_auc_xs_cz"])} '
              f'AUC_ws_rand={fm(q["_median_auc_ws_rand"])} bACC_ws_rand={fm(q["_median_bacc_ws_rand"])} '
              f'AUC_ws_block={fm(q["_median_auc_ws_block"])} bACC_ws_block={fm(q["_median_bacc_ws_block"])} '
              f'cos={fm(q["_median_cos_ws_xs"])} frac_cos<0={q["frac_cos_neg"]:.2f} '
              f'frac_xs<.5={q["frac_xs_below_half"]:.2f}', flush=True)
        # attribution
        a_xs, a_or, a_ws = q['_median_auc_xs'], q['_median_auc_xs_oriented'], q['_median_auc_ws_rand']
        if a_ws is not None and a_ws < .70:
            lab = 'c'
        elif a_ws is not None and a_or is not None and a_xs is not None and a_ws - a_xs > 1e-9 and \
                (a_or - a_xs) / (a_ws - a_xs) >= .75 and (q['_median_cos_ws_xs'] <= .30 or q['frac_cos_neg'] >= .25):
            lab = 'b'
        else:
            lab = 'a'
        result['attribution'][G] = {'label': lab, 'recovery_frac': float((a_or - a_xs) / (a_ws - a_xs))
                                    if a_ws is not None and a_xs is not None and a_ws - a_xs > 1e-9 else None,
                                    'AUC_ws_rand': a_ws, 'AUC_xs': a_xs, 'AUC_xs_oriented': a_or,
                                    'AUC_ws_block': q['_median_auc_ws_block'],
                                    'median_cos': q['_median_cos_ws_xs'], 'frac_cos_neg': q['frac_cos_neg']}
        print(f'[attribution {G}] -> ({lab}) {result["attribution"][G]}', flush=True)
    allv = [OF['subjects'][s] for s in ev_subs]
    OF['overall'] = {f'median_{f_}': med([q[f_] for q in allv]) for f_ in FIELDS}
    OF['overall']['n_subjects'] = len(ev_subs)

    # external reference: per_subject_ceiling.py (16 Welch features + LR/HGB/CNN, leave-one-block-out)
    ref = json.loads((OUT / 'per_subject_ceiling_results.json').read_text(encoding='utf-8'))['per_subject']
    refg = {}
    for G in GROUPS:
        rr = [r for r in ref if grp_of.get('sub-' + str(r['subject'])) == G]
        refg[G] = {k: med([r[f'{k}_bacc'] for r in rr]) for k in ('lr', 'hgb', 'cnn', 'mean3')}
        refg[G]['n'] = len(rr)
    result['per_subject_ceiling_ref'] = {'note': 'per_subject_ceiling.py: 16 Welch features, LR/HGB/CNN, '
                                                 'leave-one-state-block-out (pooled over block-folds), median bACC '
                                                 'per group',
                                         'groups': refg}
    print(f'[ref per_subject_ceiling] {json.dumps(refg, ensure_ascii=False)}', flush=True)

    # ladder gap per group (pipeline actual vs each ceiling) using the last stream's per-subject bACC
    sd_last = sseeds[-1]; pv = result['streams'][f'stream{sd_last}']['subjects']
    result['ladder_gaps'] = {}
    for G in GROUPS:
        ss = [s for s in ev_subs if grp_of[s] == G]
        if not ss: continue
        gq = {'n': len(ss)}
        for nm in ('pipeline', 'oracle_past'):
            gq[nm] = float(np.mean([pv[nm][s] for s in ss if pv[nm][s] is not None]))
        gq['bacc_xs'] = float(np.mean([balanced_accuracy_score(y[g == s], xs[g == s] >= .5) for s in ss]))
        gq['bacc_ws_rand'] = float(np.mean([wsr[s]['bacc_rand'] for s in ss if wsr[s]['bacc_rand'] is not None]))
        gq['bacc_ws_block'] = float(np.mean([wsr[s]['bacc_block'] for s in ss if wsr[s]['bacc_block'] is not None]))
        result['ladder_gaps'][G] = gq
        print(f'[ladder {G}] pipeline={fm(gq["pipeline"])} oracle_past={fm(gq["oracle_past"])} '
              f'xs_lr={fm(gq["bacc_xs"])} ws_rand={fm(gq["bacc_ws_rand"])} ws_block={fm(gq["bacc_ws_block"])}',
              flush=True)

    result['seconds'] = time.perf_counter() - T0
    (OUT / (stem + '.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[done] all checks passed {el()} -> {stem}.json', flush=True)
    sys.stdout = tee.stream; tee.f.close()


if __name__ == '__main__':
    main()
