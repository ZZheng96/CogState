"""Test-time unlabeled polarity (direction) of a within-subject unsupervised axis, de-positioned streams.

Protocol: stream order = concat(relax,task) permuted by default_rng([stream_seed, subject_index]) (stream_order);
outer folds / fusionM_prank / fusion3_resnet_fbank scores reused from depositioned_feature_multiscale_oof.npz and
depositioned_resnet_timefreq_oof.npz.

--phase oracle  (ORACLE DIAGNOSIS, uses test labels for the sign; never a legal method)
  z = global median impute + global StandardScaler of the 858 features (per-subject re-standardization follows, so the
  global scaler only matters for imputation). Per subject: Q = (z - mean_s)/sd_s (sd floor 1e-6 -> 1).
  Families (scores normalised by (s - median)/std of the fitting windows, so threshold .5 == fitting-set median):
    pc1        projection on PC1;
    pcbest{k}  k=3,10: best single PC among first k (oracle choice of index and sign);
    km{k}      k=3,10: KMeans(2, n_init 4, rs0) on first k PC scores, score = |t-c0|^2-|t-c1|^2;
    gmm{k}     k=3,10: GaussianMixture(2, full, reg_covar 1e-3, rs0) on first k PC scores, score = log-posterior ratio.
  full-batch: fit on all windows of the subject. causal: refit every 8 windows on windows [0, start) only, applied to
  rows [start, start+8); first fit at start=64 (rows t<64 have no axis). Refit-to-refit sign continuity: loading
  w (858-dim, standardized space) flipped if w . w_prev < 0.
  Oracle sign per subject = sign(AUC_s - .5) of the whole (causal) score sequence. Also: fusionM_prank flipped by true
  direction (threshold .5, and oracle per-subject median threshold); oracle fusion sig(logit fM + sign*s).
--phase legal   (fixed after reading depositioned_polarity_oracle.log, before any legal run; no tuning afterwards)
  Oracle outcome that fixed the design: no unsupervised axis + oracle sign reaches .85 (best causal pcbest10+fusionM
  t>=64 macro .718; fusionM flipped by true direction .657), so legal variants target the pc1 axis only (pcbest needs an
  oracle index choice), and fuse with fusionM_prank exactly as the oracle fusion: p = sig(logit fM + sgn*conf*v).
  Axis: causal PC1 of the subject's past windows [0,st) (per-subject standardisation, sd floor 1e-6), refit every 8
  windows from st=64; v = (Q_t.w - median_past)/std_past. Rows t<64: p = fM. Features contain no NaN (asserted) and
  axes / Cohen's d are invariant to the per-feature affine fold scaler, so axes are computed once per stream from
  globally scaled z; everything that uses labels (dbar, library, meta-LR) uses training-fold subjects only.
  True sign of a training refit = sign(AUC(y_s, all windows of s projected by that refit) - .5) (training labels).
  dbar = mean over training subjects of per-subject Cohen's d (task-relax, pooled sd) on z; for training snapshots of
  subject s the leave-s-out mean is used.
  V1 pol_align : sgn = sign(w . dbar), conf = 1.
  V2 pol_knn   : library l_s = normalised mean over refits of correctly oriented w (each training subject); for test w,
                 c_s = cos(w, l_s); K = 10 largest |c_s|; vote = sum c_s; sgn = sign(vote), conf = |vote| / sum |c_s|.
  V3 pol_metalr: o = sign(w . dbar); x = [o*w (858), |cos(w, dbar)|]; target = 1[o == true sign]; StandardScaler +
                 LogisticRegression(C=.05, max_iter 2000) on all training-subject refit snapshots (dbar leave-s-out);
                 q = P(correct); sgn = o*sign(q-.5), conf = |2q-1|.
  Reported: fused (fusionX = pol_X) and axis-only (sig(sgn*conf*v), t>=64); refit-level sign accuracy vs true sign
  (evaluation only). Paired subject bootstrap 1000 vs fusionM_prank and fusion3_resnet_fbank (all / mature t>=120).
  Checks: position-only AUC ~.5 (stored); prefix invariance (future rows +100 -> prefix scores identical, future
  changed); subject isolation (axis of subject alone == pooled; test subjects absent from dbar/library/meta training);
  test-label shuffle -> identical fused scores.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from depositioned_history_delta import boot, load_db  # noqa: E402
from depositioned_resnet_history_delta import stream_order, logit, sig  # noqa: E402
from causal_kernel_state_mapping import metrics  # noqa: E402
from switch_dual_memory import Tee  # noqa: E402

REFIT, WARM, KMAX = 8, 64, 10
FAMS = ('pc1', 'pcbest3', 'pcbest10', 'km3', 'km10', 'gmm3', 'gmm10')
STREAM_SEEDS = (0, 1, 2)
T0 = time.perf_counter()


def el():
    return f'elapsed={time.perf_counter() - T0:.1f}s'


def std_fit(P):
    mu, sd = P.mean(0), P.std(0)
    return mu, np.where(sd > 1e-6, sd, 1.)


def fit_axes(P, fams):
    """Fit on past/all windows P. Returns dict fam -> (scorer(X)->score, w) ; pcbest returns list over components."""
    mu, sd = std_fit(P); Q = (P - mu) / sd
    V = np.linalg.svd(Q, full_matrices=False)[2][:KMAX]; T = Q @ V.T
    proj = lambda X: ((X - mu) / sd) @ V.T  # noqa: E731
    out = {}
    for f in fams:
        if f == 'pc1':
            out[f] = (lambda X: proj(X)[:, 0], V[0].copy())
        elif f.startswith('pcbest'):
            k = int(f[6:]); out[f] = [(lambda X, j=j: proj(X)[:, j], V[j].copy()) for j in range(k)]
        elif f.startswith('km'):
            k = int(f[2:]); c = KMeans(2, n_init=4, random_state=0).fit(T[:, :k]).cluster_centers_
            out[f] = (lambda X, c=c, k=k: ((proj(X)[:, :k] - c[0]) ** 2).sum(1) - ((proj(X)[:, :k] - c[1]) ** 2).sum(1),
                      V[:k].T @ (c[1] - c[0]))
        else:
            k = int(f[3:]); gm = GaussianMixture(2, covariance_type='full', reg_covar=1e-3, random_state=0).fit(T[:, :k])
            def sc(X, gm=gm, k=k):
                lw = gm._estimate_weighted_log_prob(proj(X)[:, :k]); return lw[:, 1] - lw[:, 0]
            out[f] = (sc, V[:k].T @ (gm.means_[1] - gm.means_[0]))
    return out


def norm_apply(scorer, P, X):
    sp = scorer(P); s = scorer(X)
    return (s - np.median(sp)) / (sp.std() + 1e-9)


def subject_axes(zs, fams, causal):
    """zs: subject windows (stream order for causal). Returns fam -> score array (pcbest -> (n, k) array)."""
    n = len(zs)
    def alloc(f): return np.full((n, int(f[6:])), np.nan) if f.startswith('pcbest') else np.full(n, np.nan)
    out = {f: alloc(f) for f in fams}
    if not causal:
        A = fit_axes(zs, fams)
        for f in fams:
            if f.startswith('pcbest'): out[f] = np.stack([norm_apply(s, zs, zs) for s, _ in A[f]], 1)
            else: out[f] = norm_apply(A[f][0], zs, zs)
        return out
    prev = {}
    for st in range(WARM, n, REFIT):
        P, X = zs[:st], zs[st:st + REFIT]; A = fit_axes(P, fams)
        for f in fams:
            items = A[f] if f.startswith('pcbest') else [A[f]]
            for j, (s, w) in enumerate(items):
                sgn = 1. if (f, j) not in prev or w @ prev[(f, j)] >= 0 else -1.
                prev[(f, j)] = sgn * w; v = sgn * norm_apply(s, P, X)
                if f.startswith('pcbest'): out[f][st:st + REFIT, j] = v
                else: out[f][st:st + REFIT] = v
    return out


def sauc(y, s):
    m = np.isfinite(s); return float(roc_auc_score(y[m], s[m]))


def oracle_signed(y, S):
    """S: (n,) or (n,k). Oracle choose column + sign maximizing |AUC-.5|. Returns signed (n,) score."""
    S = S if S.ndim == 2 else S[:, None]
    a = np.array([sauc(y, S[:, j]) for j in range(S.shape[1])]); j = int(np.argmax(np.abs(a - .5)))
    return S[:, j] * (1. if a[j] >= .5 else -1.)


def summarize(name, y, g, p, mask, res, tag='oracle'):
    r = metrics(y, g, p, mask)
    ba = np.array([q for s in np.unique(g[mask]) if (q := metrics(y, g, p, mask & (g == s))['subject_macro_bACC']) is not None])
    au = np.array([q for s in np.unique(g[mask]) if (q := metrics(y, g, p, mask & (g == s))['AUC']) is not None])
    r.update(n_bACC_ge_0_9=int((ba >= .9).sum()), n_bACC_ge_0_85=int((ba >= .85).sum()), median_subject_AUC=float(np.median(au)),
             mean_subject_AUC=float(au.mean()))
    res[name] = r
    print(f'[{tag}] {name:34s} AUC={r["AUC"]:.4f} mac={r["subject_macro_bACC"]:.4f} mic={r["micro_bACC"]:.4f} '
          f'subjAUC_mean={r["mean_subject_AUC"]:.4f} n>=.9={r["n_bACC_ge_0_9"]} n>=.85={r["n_bACC_ge_0_85"]}', flush=True)


def run_oracle(args):
    stem = 'depositioned_polarity_oracle' + ('_smoke' if args.smoke else '')
    tee = Tee(sys.stdout, OUT / (stem + '.log')); sys.stdout = tee
    man, subs, data = load_db()
    Fc = np.vstack([data[s][0] for s in subs]); y = np.concatenate([data[s][1] for s in subs])
    g = np.concatenate([np.full(len(data[s][1]), s) for s in subs])
    ms = np.load(OUT / 'depositioned_feature_multiscale_oof.npz'); tf = np.load(OUT / 'depositioned_resnet_timefreq_oof.npz')
    assert np.array_equal(ms['y'], y) and np.array_equal(ms['groups'], g) and np.array_equal(tf['groups'], g)
    use = subs[:8] if args.smoke else subs; sseeds = STREAM_SEEDS[:1] if args.smoke else STREAM_SEEDS
    med = np.nanmedian(np.where(np.isfinite(Fc), Fc, np.nan), 0); med = np.where(np.isfinite(med), med, 0.)
    Z = StandardScaler().fit_transform(np.where(np.isfinite(Fc), Fc, med))
    print(f'[start oracle] rows={len(y)} subjects={len(use)} sseeds={sseeds} fams={FAMS} refit={REFIT} warm={WARM}', flush=True)
    res = {'meta': {'ORACLE': 'test labels used for sign/column choice; diagnosis only', 'refit': REFIT, 'warm': WARM,
                    'fams': FAMS, 'smoke': args.smoke}}
    inset = np.isin(g, use); ev = inset & np.isfinite(ms['stream0__fusionM_prank'])
    # full-batch (order independent)
    FB = {f: np.full(len(y), np.nan) for f in FAMS}; per_sub = {}
    with threadpool_limits(limits=8):
        for j, s in enumerate(use):
            loc = np.flatnonzero(g == s); A = subject_axes(Z[loc], FAMS, False)
            per_sub[s] = {}
            for f in FAMS:
                v = oracle_signed(y[loc], A[f]); FB[f][loc] = v; per_sub[s][f'full_{f}'] = sauc(y[loc], v)
            if j % 10 == 0 or j == len(use) - 1: print(f'[full {j + 1}/{len(use)}] {s} ' + ' '.join(
                f'{f}={per_sub[s][f"full_{f}"]:.3f}' for f in FAMS) + f' {el()}', flush=True)
    R = res['full_batch'] = {}
    for f in FAMS: summarize(f'full_{f}', y, g, sig(FB[f]), ev, R)
    for sd in sseeds:
        order, t = stream_order(g, subs, sd); sidx = {s: j for j, s in enumerate(subs)}
        R = res[f'stream{sd}'] = {}
        fm, f3 = ms[f'stream{sd}__fusionM_prank'], tf[f'stream{sd}__fusion3_resnet_fbank']
        assert np.allclose(tf[f'stream{sd}__fusionM_prank'][ev], fm[ev])
        m64 = ev & (t >= WARM)
        for nm, p in (('fusionM_prank', fm), ('fusion3_resnet_fbank', f3)):
            summarize(f's{sd}_{nm}', y, g, p, ev, R); summarize(f's{sd}_{nm}__t>=64', y, g, p, m64, R)
            fl = p.copy(); flm = np.full(len(y), np.nan)
            for s in use:
                m = g == s
                if sauc(y[m], p[m]) < .5: fl[m] = 1 - p[m]
                flm[m] = sig(logit(fl[m]) - np.median(logit(fl[m])))
            summarize(f's{sd}_{nm}_FLIP', y, g, fl, ev, R); summarize(f's{sd}_{nm}_FLIP_medthr', y, g, flm, ev, R)
        CA = {f: np.full(len(y), np.nan) for f in FAMS}
        with threadpool_limits(limits=8):
            for j, s in enumerate(use):
                loc = order[sidx[s]]; A = subject_axes(Z[loc], FAMS, True)
                for f in FAMS:
                    v = np.full(len(loc), np.nan); ok = np.arange(len(loc)) >= WARM
                    sv = oracle_signed(y[loc][ok], (A[f] if A[f].ndim == 1 else A[f])[ok]); v[ok] = sv; CA[f][loc] = v
                if j % 10 == 0 or j == len(use) - 1:
                    print(f'[causal s{sd} {j + 1}/{len(use)}] {s} ' + ' '.join(
                        f'{f}={sauc(y[loc], CA[f][loc]):.3f}' for f in FAMS) + f' {el()}', flush=True)
        for f in FAMS:
            pa = sig(np.nan_to_num(CA[f]))
            summarize(f's{sd}_causal_{f}__t>=64', y, g, pa, m64, R)
            fu = sig(logit(fm) + np.nan_to_num(CA[f]))
            summarize(f's{sd}_causal_{f}+fusionM__all', y, g, fu, ev, R); summarize(f's{sd}_causal_{f}+fusionM__t>=64', y, g, fu, m64, R)
        (OUT / (stem + '.json')).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding='utf-8')
    res['per_subject_full'] = per_sub
    hits = {k: v for blk in res.values() if isinstance(blk, dict) for k, v in blk.items()
            if isinstance(v, dict) and 'subject_macro_bACC' in v and v['subject_macro_bACC'] is not None and v['subject_macro_bACC'] >= .85}
    res['combos_macro_bACC_ge_0.85'] = {k: v['subject_macro_bACC'] for k, v in hits.items()}
    print(f'[oracle summary] combos with macro bACC >= .85: {res["combos_macro_bACC_ge_0.85"]}', flush=True)
    res['seconds'] = time.perf_counter() - T0
    (OUT / (stem + '.json')).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[done] {el()} -> {stem}.json', flush=True)
    sys.stdout = tee.stream; tee.f.close()


LEG = ('pol_align', 'pol_knn', 'pol_metalr')
KNN, C_META, MATURE = 10, .05, 120


def causal_pc1(zs):
    """Row t (t>=WARM) uses zs[:st] for the fit (st = block start <= t) and zs[t] itself. Returns v, refit id, PJ
    (n x R projections of every row of zs by each refit, used only with training labels), W (R x 858)."""
    n = len(zs); v = np.full(n, np.nan); rid = np.full(n, -1); W, PJ = [], []
    for st in range(WARM, n, REFIT):
        P = zs[:st]; mu, sd = std_fit(P); Q = (P - mu) / sd
        w = np.linalg.svd(Q, full_matrices=False)[2][0]; sp = Q @ w
        v[st:st + REFIT] = (((zs[st:st + REFIT] - mu) / sd) @ w - np.median(sp)) / (sp.std() + 1e-9)
        rid[st:st + REFIT] = len(W); W.append(w); PJ.append(((zs - mu) / sd) @ w)
    return v, rid, np.array(PJ).T, np.array(W)


def cohen_d(zs, ys):
    a, b = zs[ys == 0], zs[ys == 1]
    sd = np.sqrt(((len(a) - 1) * a.var(0, ddof=1) + (len(b) - 1) * b.var(0, ddof=1)) / (len(a) + len(b) - 2))
    return np.where(sd > 1e-9, (b.mean(0) - a.mean(0)) / np.where(sd > 1e-9, sd, 1.), 0.)


def fold_models(train_subs, AX, ylab):
    """All label use is here, on training subjects only. ylab: s -> labels in stream order."""
    D = {s: cohen_d(AX[s]['zs'], ylab[s]) for s in train_subs}; Ds = sum(D.values()); n = len(train_subs)
    dbar = Ds / n; L, Xm, Tm = [], [], []
    for s in train_subs:
        W = AX[s]['W']
        if not len(W): continue
        ts = np.array([1. if roc_auc_score(ylab[s], AX[s]['PJ'][:, r]) >= .5 else -1. for r in range(len(W))])
        l = (ts[:, None] * W).mean(0); L.append(l / (np.linalg.norm(l) + 1e-12))
        dl = (Ds - D[s]) / (n - 1); cs = W @ dl / (np.linalg.norm(dl) + 1e-12); o = np.where(cs >= 0, 1., -1.)
        Xm.append(np.hstack([o[:, None] * W, np.abs(cs)[:, None]])); Tm.append((o == ts).astype(int))
    Xm, Tm = np.vstack(Xm), np.concatenate(Tm)
    sc = StandardScaler().fit(Xm); lr = LogisticRegression(C=C_META, max_iter=2000).fit(sc.transform(Xm), Tm)
    return {'dbar': dbar, 'L': np.array(L), 'sc': sc, 'lr': lr, 'meta_train_pos_rate': float(Tm.mean()), 'n_snap': int(len(Tm))}


def polarity(W, FM):
    """W: R x 858 refit loadings of one subject -> {variant: (sgn, conf)} per refit."""
    out = {}
    if not len(W): return {v: (np.zeros(0), np.zeros(0)) for v in LEG}
    dbar = FM['dbar']; cs = W @ dbar / (np.linalg.norm(dbar) + 1e-12); o = np.where(cs >= 0, 1., -1.)
    out['pol_align'] = (o, np.ones(len(W)))
    C = W @ FM['L'].T; idx = np.argsort(-np.abs(C), 1)[:, :KNN]; Ck = np.take_along_axis(C, idx, 1)
    vote = Ck.sum(1); out['pol_knn'] = (np.where(vote >= 0, 1., -1.), np.abs(vote) / (np.abs(Ck).sum(1) + 1e-12))
    q = FM['lr'].predict_proba(FM['sc'].transform(np.hstack([o[:, None] * W, np.abs(cs)[:, None]])))[:, 1]
    out['pol_metalr'] = (o * np.where(q >= .5, 1., -1.), np.abs(2 * q - 1))
    return out


def fuse(fm_s, v, rid, pol):
    """fm_s, v, rid: one subject stream order. Returns fused and axis-only probs per variant."""
    fu, ax = {}, {}
    for k, (sg, cf) in pol.items():
        a = np.zeros(len(v)); ok = rid >= 0
        a[ok] = sg[rid[ok]] * cf[rid[ok]] * v[ok]
        fu[k] = sig(logit(fm_s) + a); ax[k] = np.where(ok, sig(a), np.nan)
    return fu, ax


def run_legal(args):
    stem = 'depositioned_polarity' + ('_smoke' if args.smoke else '')
    tee = Tee(sys.stdout, OUT / (stem + '.log')); sys.stdout = tee
    man, subs, data = load_db()
    Fc = np.vstack([data[s][0] for s in subs]); y = np.concatenate([data[s][1] for s in subs])
    g = np.concatenate([np.full(len(data[s][1]), s) for s in subs]); assert np.isfinite(Fc).all()
    ms = np.load(OUT / 'depositioned_feature_multiscale_oof.npz'); tf = np.load(OUT / 'depositioned_resnet_timefreq_oof.npz')
    assert np.array_equal(ms['y'], y) and np.array_equal(ms['groups'], g) and np.array_equal(tf['groups'], g)
    fold = ms['fold']; Z = StandardScaler().fit_transform(Fc)
    sseeds = STREAM_SEEDS[:1] if args.smoke else STREAM_SEEDS; nb = 100 if args.smoke else 1000
    run_folds = [0] if args.smoke else list(range(5))
    if args.smoke:
        f0 = list(np.unique(g[fold == 0])); use = f0 + [s for s in subs if s not in f0][:10]
    else: use = list(subs)
    print(f'[start legal] rows={len(y)} subjects={len(use)} sseeds={sseeds} folds={run_folds} variants={LEG} '
          f'refit={REFIT} warm={WARM} knn={KNN} C_meta={C_META}', flush=True)
    res = {'meta': {'variants': LEG, 'refit': REFIT, 'warm': WARM, 'knn': KNN, 'C_meta': C_META, 'mature': MATURE,
                    'smoke': args.smoke, 'n_boot': nb, 'fusion': 'sig(logit fusionM_prank + sgn*conf*v)'}, 'checks': {}}
    checks = res['checks']; oof = {'y': y, 'groups': g, 'fold': fold}; sidx = {s: j for j, s in enumerate(subs)}
    with threadpool_limits(limits=8):
        for sd in sseeds:
            order, t = stream_order(g, subs, sd); fm = ms[f'stream{sd}__fusionM_prank']
            AX = {}
            for j, s in enumerate(use):
                loc = order[sidx[s]]; v, rid, PJ, W = causal_pc1(Z[loc])
                AX[s] = {'loc': loc, 'zs': Z[loc], 'v': v, 'rid': rid, 'PJ': PJ, 'W': W}
                if j % 10 == 0 or j == len(use) - 1: print(f'[s{sd} axes {j + 1}/{len(use)}] {s} refits={len(W)} {el()}', flush=True)
            s1 = use[0]; v1 = causal_pc1(Z[order[sidx[s1]]])[0]
            assert np.array_equal(v1, AX[s1]['v'], equal_nan=True); checks[f'isolation_axis_s{sd}'] = True
            ylab = {s: y[AX[s]['loc']] for s in use}
            S = {f'{p}{k}': np.full(len(y), np.nan) for k in LEG for p in ('fusion_', 'axis_')}
            sacc = {k: [] for k in LEG}
            for k_ in run_folds:
                te_subs = [s for s in use if fold[AX[s]['loc'][0]] == k_]; tr_subs = [s for s in use if fold[AX[s]['loc'][0]] != k_]
                assert not set(te_subs) & set(tr_subs)
                FM = fold_models(tr_subs, AX, ylab)
                for s in te_subs:
                    A = AX[s]; pol = polarity(A['W'], FM); fu, ax = fuse(fm[A['loc']], A['v'], A['rid'], pol)
                    for k in LEG:
                        S[f'fusion_{k}'][A['loc']] = fu[k]; S[f'axis_{k}'][A['loc']] = ax[k]
                        ts = np.array([1. if roc_auc_score(ylab[s], A['PJ'][:, r]) >= .5 else -1. for r in range(len(A['W']))])
                        sacc[k].append((s, float((pol[k][0] == ts).mean()), float(np.mean(pol[k][1]))))
                if k_ == run_folds[0]:  # test-label shuffle -> identical fused scores
                    y2 = dict(ylab)
                    for s in te_subs: y2[s] = np.random.default_rng(1).permutation(ylab[s])
                    FM2 = fold_models(tr_subs, AX, y2)
                    for s in te_subs:
                        A = AX[s]; f2 = fuse(fm[A['loc']], A['v'], A['rid'], polarity(A['W'], FM2))[0]
                        for k in LEG: assert np.array_equal(f2[k], S[f'fusion_{k}'][A['loc']])
                    checks[f'test_label_shuffle_invariant_s{sd}'] = True
                for s in te_subs[:2]:  # prefix invariance
                    A = AX[s]; n = len(A['loc']); cut = n // 2; alt = A['zs'].copy(); alt[cut:] += 100.
                    va, ra, _, Wa = causal_pc1(alt); fa = fuse(fm[A['loc']], va, ra, polarity(Wa, FM))[0]
                    for k in LEG:
                        ref = S[f'fusion_{k}'][A['loc']]
                        d_pre = float(np.abs(fa[k][:cut] - ref[:cut]).max()); d_fut = float(np.abs(fa[k][cut:] - ref[cut:]).max())
                        assert d_pre == 0. and d_fut > 1e-6, (k, d_pre, d_fut)
                        checks[f'prefix_s{sd}_f{k_}_{s}_{k}'] = {'prefix': d_pre, 'future': d_fut}
                print(f'[s{sd} fold {k_ + 1}/5] test={len(te_subs)} train={len(tr_subs)} meta_snap={FM["n_snap"]} '
                      f'pos_rate={FM["meta_train_pos_rate"]:.3f} ' + ' '.join(
                          f'{k}: signacc={np.mean([a for s_, a, _ in sacc[k] if s_ in te_subs]):.3f}' for k in LEG) +
                      f' checks ok {el()}', flush=True)
            ev = np.isfinite(S[f'fusion_{LEG[0]}']); m64 = ev & (t >= WARM)
            M = {'all': ev, 'mature': ev & (t >= MATURE)}
            S['fusionM_prank'] = np.where(ev, fm, np.nan); S['fusion3_resnet_fbank'] = np.where(ev, tf[f'stream{sd}__fusion3_resnet_fbank'], np.nan)
            R = res[f'stream{sd}'] = {'methods': {}, 'axis_only_t>=64': {}, 'bootstrap': {}, 'sign_acc': {}, 'sanity': {}}
            for c in ('lr', 'hgb'):
                a = metrics(y, g, ms[f'stream{sd}__position_only__{c}'], ev)['AUC']; ok = abs(a - .5) < .03
                R['sanity'][f'position_only__{c}'] = {'AUC': a, 'ok': ok}
                print(f'[sanity stream {sd}] position_only__{c} AUC={a:.4f} -> {"OK" if ok else "FLAG"}', flush=True)
                if not args.smoke: assert ok
            for n_ in ['fusionM_prank', 'fusion3_resnet_fbank'] + [f'fusion_{k}' for k in LEG]:
                R['methods'][n_] = {}
                for seg, mk in M.items(): summarize(f's{sd}_{n_}__{seg}', y, g, S[n_], mk, R['methods'][n_], 'legal')
                summarize(f's{sd}_{n_}__t>=64', y, g, S[n_], m64, R['methods'][n_], 'legal')
            for k in LEG:
                summarize(f's{sd}_axis_{k}__t>=64', y, g, S[f'axis_{k}'], m64, R['axis_only_t>=64'], 'legal')
                a = np.array([x for _, x, _ in sacc[k]]); c = np.array([x for _, _, x in sacc[k]])
                R['sign_acc'][k] = {'refit_mean': float(a.mean()), 'n_subj_majority_correct': int((a > .5).sum()),
                                    'n_subj': int(len(a)), 'mean_conf': float(c.mean()), 'per_subject': {s_: x for s_, x, _ in sacc[k]}}
                print(f'[sign stream {sd}] {k:10s} refit sign acc={a.mean():.3f} subjects majority-correct '
                      f'{(a > .5).sum()}/{len(a)} mean conf={c.mean():.3f}', flush=True)
            for k in LEG:
                for b_ in ('fusionM_prank', 'fusion3_resnet_fbank'):
                    for seg in M:
                        bb = R['bootstrap'][f'fusion_{k}__vs__{b_}__{seg}'] = boot(S[f'fusion_{k}'], S[b_], y, g, M[seg], nb)
                        print(f'[boot stream {sd}] fusion_{k:10s} vs {b_:20s} {seg:6s} '
                              f'dAUC={bb["dAUC"]["mean"]:+.4f}[{bb["dAUC"]["p2.5"]:+.4f},{bb["dAUC"]["p97.5"]:+.4f}] '
                              f'dmac={bb["dmacro_bACC"]["mean"]:+.4f}[{bb["dmacro_bACC"]["p2.5"]:+.4f},{bb["dmacro_bACC"]["p97.5"]:+.4f}]', flush=True)
            for n_, p in S.items(): oof[f'stream{sd}__{n_}'] = p
            (OUT / (stem + '.json')).write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding='utf-8')
    res['seconds'] = time.perf_counter() - T0
    (OUT / (stem + '.json')).write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding='utf-8')
    np.savez_compressed(OUT / (stem + '_oof.npz'), **oof)
    print(f'[done] {el()} -> {stem}.json', flush=True)
    sys.stdout = tee.stream; tee.f.close()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--phase', choices=('oracle', 'legal'), required=True); args = ap.parse_args()
    if args.phase == 'oracle': run_oracle(args)
    else: run_legal(args)


if __name__ == '__main__':
    main()
