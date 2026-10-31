"""DIRECTION DIAGNOSIS: can label-free statistics of the past (unlabeled) windows tell that a subject's fuseH_wf
"task" direction is REVERSED, so that flipping the seed pseudo labels repairs pseudo_lr_full?

Protocol identical to depositioned_within_subject_adapt_v2.py / depositioned_within_subject_repr.py: folds
SGKF(5, shuffle, rs0) on the seed-0 'independent' arrangement (asserted == npz fold); stream_order(g, subs, sd);
branch logits a = logit(A['stream{sd}__all_prank']), w = logit(B['stream{sd}__resnet_whiten_logit_delta']),
f = logit(B['stream{sd}__resnet_fbank_logit_delta']), L = mean(a, w, f), fuseH_wf = sig(L). ZZ (889) = fold-train
median imputation + StandardScaler(fold train rows) of the 858 features and the 31 new_features, built for the test
subjects of each fold only. Per subject in stream order, refit points t0 = 48, 64, 80, ... (MIN_PAST=48, R_STEP=16);
every statistic and every model only see the subject's rows [0, t0) (past PCA16 = StandardScaler(past ZZ) -> PCA(16,
svd full); pseudo labels = top / bottom 30% of past L, PSEUDO_Q=.3, stable argsort, top=1); rows t < 48 have no refit.
A subject-level value of every statistic = plain mean over its refit points (NaN refits dropped).

Statistics per refit point (before running, all fixed):
  LEAKY (truth diagnosis only, never fed back):  acc_seed = accuracy of the seed pseudo labels vs TRUE past labels;
    auc_L_true = within-subject AUC of past L vs TRUE past labels (NaN unless both classes present).
  LEGAL (past unlabeled windows + L only):  gmm2 = GaussianMixture(2, covariance_type='full', random_state=0) on the
    past PCA16; dGMM = p(comp1) - p(comp0); sori_abs = |Spearman(dGMM, L)|; kappa_gmm = Cohen kappa between the seed
    pseudo labels and the GMM hard partition ON THE 2*q SEED ROWS, best of the two label matchings (orientation-free by
    construction); ashman = sqrt(2) * |mu1-mu0| / sqrt(s1^2+s0^2) with the means / SDs taken on the projections onto
    (mu1 - mu0); abs_skew = |skew(dGMM)|; dbic = BIC(1 component) - BIC(2 components) on the past PCA16 (positive =
    two components preferred); branch_r = mean of the 3 pairwise within-subject Spearman correlations of a / w / f on
    the past rows; branch_L_pos = mean of sign(Spearman(branch_i, L)) over the 3 branches (in [-1, 1]); ac1 = lag-1
    Pearson autocorrelation of past L; drift = (mean(L second half) - mean(L first half)) / std(L).
  CROSS-SUBJECT REFERENCE (still legal: training labels only, no test label): pop_dir = Spearman(L, past ZZ
    projected on dhat), where dhat = unit mean over the fold's TRAINING subjects of (mean ZZ of that subject's task
    rows - mean ZZ of its relax rows), computed with the same fold imputation / scaler (so it lives in the scaled
    889-d space). The listed within-subject statistics are all invariant to a global sign flip of L (dGMM's cluster
    order is arbitrary, the kappa matching is free, D / |skew| / dBIC / branch correlation are sign-free), so they
    cannot by themselves identify a reversal; pop_dir is the only statistic with an absolute reference, and it is the
    one the legal-flip variant is allowed to use. Negative pop_dir = the subject's L runs against the majority.

Reversal bookkeeping (LEAKY, diagnosis only): subject is 'reversed' iff its subject-mean auc_L_true < .5 OR its
subject-mean acc_seed < .5. Flipping a subject's pseudo labels wholesale (seed labels 1 <-> 0) is equivalent to
training the same LR on swapped labels; pseudo_lr_flip is fitted explicitly (LogisticRegression(C=.01, lbfgs,
max_iter 1000) on the past-scaled ZZ with labels 1-yl), and the script asserts z_flip == -z_full.

Detection analysis (subject level, 57 subjects): target bad = acc_seed < .6 (ACC_BAD); for every legal statistic the
score = mult * statistic (mult in DETECT; only pop_dir and branch_L_pos have a pre-registered direction, the rest are
reported exploratory with mult = +1) -> cross-subject ROC-AUC + subject bootstrap CI (1000; smoke 100) + Spearman rho
vs subject-mean acc_seed; mean of every statistic per group high / mid / low (depositioned_subject_diagnosis.json).

Outputs (threshold .5; fuse_X = sig(L + z_X), fallback L where z is NaN; fuse_X_thr = thr_mean256 of the fused logit,
depositioned_adaptive_threshold.apply):
  fuse_pseudo_lr_full(_thr)   reference, asserted == v2 json
  fuse_oracle_flip(_thr)      LEAKY upper bound: pseudo_lr_flip for the reversed subjects, pseudo_lr_full elsewhere
  fuse_flipL(_thr)            LEAKY pure upper bound: sig(-L) for the reversed subjects, sig(L) elsewhere
  fuse_flipE(_thr)            LEGAL: same flip rule but the reversed set comes from the best pre-registered legal
                              statistic (max detection AUC among neg_pop_dir / branch_L_neg, threshold fixed at 0:
                              flip iff the subject-level value is > 0). Skipped (and recorded) if that statistic's
                              bootstrap CI lower bound is <= .5.
Evaluation as repr: all / mature (t >= 120); AUC, subject-macro bACC, micro bACC, #subjects bACC >= .9, max; groups
high/mid/low (mean and Delta vs fuse_pseudo_lr_full(_thr)); paired subject bootstrap (1000; smoke 100) of every output
vs fuse_pseudo_lr_full(_thr).
Checks (assert): fold == npz; v2 reproduction; subject isolation (alone == pooled, z and statistics); prefix invariance
(future ZZ / branch logits perturbed -> earlier refit points' z and statistics identical, later ones change); test-label
shuffle -> all legal statistics and both z variants identical; z_flip == -z_full. All hyper-parameters were fixed
before the first run and are not tuned afterwards.
Parallel over subjects: joblib loky, 8 jobs, 1 BLAS thread each (deterministic).
Run: python test/depositioned_direction_diagnosis.py [--smoke]   (smoke: stream0, fold0, bootstrap 100)
"""
import argparse, json, sys, time, warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import skew, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
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
from depositioned_within_subject_adapt import pseudo_labels, GROUPS, MATURE, STREAM_SEEDS, THR_K, R_STEP, \
    MIN_PAST, NPC, PSEUDO_Q  # noqa: E402

CLIP, EPS, N_JOBS, ACC_BAD = 3., 1e-6, 8, .6
VARIANTS = ('pseudo_lr_full', 'pseudo_lr_flip')
LEAKY_STATS = ('acc_seed', 'auc_L_true')
STATS = ('acc_seed', 'auc_L_true', 'sori_abs', 'kappa_gmm', 'ashman', 'abs_skew', 'dbic', 'branch_r',
         'branch_L_pos', 'ac1', 'drift', 'pop_dir')
# detection score = mult * statistic, higher = more likely REVERSED. mult = 0 -> exploratory, no pre-registered sign.
DETECT = (('neg_pop_dir', 'pop_dir', -1.), ('branch_L_neg', 'branch_L_pos', -1.),
          ('kappa_gmm', 'kappa_gmm', 0.), ('neg_sori_abs', 'sori_abs', 0.), ('ashman', 'ashman', 0.),
          ('abs_skew', 'abs_skew', 0.), ('dbic', 'dbic', 0.), ('branch_r', 'branch_r', 0.),
          ('ac1', 'ac1', 0.), ('drift', 'drift', 0.))
CANDIDATES = ('neg_pop_dir', 'branch_L_neg')
T0 = time.perf_counter()


def el():
    return f'elapsed={time.perf_counter() - T0:.1f}s'


def zc(sp, sn):
    return np.clip((sn - sp.mean()) / max(sp.std(), EPS), -CLIP, CLIP)


def lr():
    return LogisticRegression(C=.01, max_iter=1000)


def kappa_best(a, b):
    """Cohen kappa between binary a, b under the better of the two label matchings (orientation-free)."""
    if len(a) == 0 or np.unique(a).size < 2 or np.unique(b).size < 2: return np.nan
    p1a, p1b = float(a.mean()), float(b.mean())
    po = float(np.mean(a == b))
    pe = p1a * p1b + (1 - p1a) * (1 - p1b)
    k1 = (po - pe) / max(1 - pe, EPS)
    po2 = 1 - po
    pe2 = p1a * (1 - p1b) + (1 - p1a) * p1b
    k2 = (po2 - pe2) / max(1 - pe2, EPS)
    return float(max(k1, k2))


def refit(zp, Bp, yp, zn, dhat):
    """One refit point: past ZZ / branch logits / true past labels / new ZZ / population direction -> (z, statistics)."""
    out, dg = {}, {}
    lp = Bp.mean(1)
    sc = StandardScaler().fit(zp); xp, xn = sc.transform(zp), sc.transform(zn)
    idx, yl = pseudo_labels(lp)
    dg['acc_seed'] = float((yp[idx] == yl).mean())
    ok = np.unique(yp).size == 2 and np.std(lp) > 0
    dg['auc_L_true'] = float(roc_auc_score(yp, lp)) if ok else np.nan
    m = lr().fit(xp[idx], yl); out['pseudo_lr_full'] = zc(m.decision_function(xp), m.decision_function(xn))
    m = lr().fit(xp[idx], 1 - yl); out['pseudo_lr_flip'] = zc(m.decision_function(xp), m.decision_function(xn))
    pca = PCA(NPC, svd_solver='full').fit(xp); pp = pca.transform(xp)
    try:
        gm = GaussianMixture(2, covariance_type='full', random_state=0).fit(pp)
        post = gm.predict_proba(pp); dgm = post[:, 1] - post[:, 0]; lab = gm.predict(pp)
        r = spearmanr(dgm, lp)[0]
        dg['sori_abs'] = abs(float(r)) if np.isfinite(r) else np.nan
        dg['kappa_gmm'] = kappa_best(yl, lab[idx])
        w = gm.means_[1] - gm.means_[0]
        p1, p0 = pp[lab == 1] @ w, pp[lab == 0] @ w
        dg['ashman'] = float(np.sqrt(2.) * abs(w @ w) / max(np.sqrt(p1.var(ddof=1) + p0.var(ddof=1)), EPS)) \
            if len(p1) > 1 and len(p0) > 1 else np.nan
        dg['abs_skew'] = float(abs(skew(dgm))) if np.std(dgm) > 0 else np.nan
        gm1 = GaussianMixture(1, covariance_type='full', random_state=0).fit(pp)
        dg['dbic'] = float(gm1.bic(pp) - gm.bic(pp))
    except Exception:
        for k in ('sori_abs', 'kappa_gmm', 'ashman', 'abs_skew', 'dbic'): dg[k] = np.nan
    rr = [float(spearmanr(Bp[:, i], Bp[:, j])[0]) for i, j in ((0, 1), (0, 2), (1, 2))]
    rr = [v for v in rr if np.isfinite(v)]
    dg['branch_r'] = float(np.mean(rr)) if rr else np.nan
    sl = [float(np.sign(spearmanr(Bp[:, i], lp)[0])) for i in range(3) if np.std(Bp[:, i]) > 0 and np.std(lp) > 0]
    dg['branch_L_pos'] = float(np.mean(sl)) if sl else np.nan
    if np.std(lp) > 0 and len(lp) > 2:
        dg['ac1'] = float(np.corrcoef(lp[:-1], lp[1:])[0, 1])
        h = len(lp) // 2
        dg['drift'] = float((lp[h:].mean() - lp[:h].mean()) / max(lp.std(), EPS))
    else:
        dg['ac1'] = np.nan; dg['drift'] = np.nan
    r = spearmanr(lp, xp @ dhat)[0]
    dg['pop_dir'] = float(r) if np.isfinite(r) else np.nan
    return out, dg


def adapt_subject(Z, Bl, y, dhat):
    """One subject in stream order: -> ({variant: z (n,)}, [statistics per refit point])."""
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        n = len(y); res = {v: np.full(n, np.nan) for v in VARIANTS}; st = []
        for t0 in range(MIN_PAST, n, R_STEP):
            t1 = min(n, t0 + R_STEP)
            o, dg = refit(Z[:t0], Bl[:t0], y[:t0], Z[t0:t1], dhat); st.append(dg)
            for v in VARIANTS: res[v][t0:t1] = o[v]
    return res, st


def mean_stats(st):
    return {k: (float(np.mean([d[k] for d in st if np.isfinite(d[k])]))
                if any(np.isfinite(d[k]) for d in st) else np.nan) for k in STATS}


def det_auc(score, bad):
    return float(roc_auc_score(bad, score)) if np.unique(bad).size == 2 else None


def boot_auc(score, bad, nb):
    rng = np.random.default_rng(0); n = len(bad); out = []
    for _ in range(nb):
        i = rng.integers(0, n, n)
        if np.unique(bad[i]).size == 2: out.append(roc_auc_score(bad[i], score[i]))
    o = np.array(out) if out else np.array([np.nan])
    return {'mean': float(o.mean()), 'p2.5': float(np.percentile(o, 2.5)), 'p97.5': float(np.percentile(o, 97.5)),
            'n': int(len(out))}


def dir_fold(Fc, Fn, tr, y, g):
    """Illegal-free population direction: unit mean over TRAINING subjects of (task mean - relax mean) in ZZ space."""
    tfo, tfn = impute_scale(Fc, tr), impute_scale(Fn, tr)
    X = np.hstack([tfo(Fc[tr]), tfn(Fn[tr])]); yt, gt = y[tr], g[tr]
    ds = []
    for s in np.unique(gt):
        q = gt == s; a, b = X[q & (yt == 1)], X[q & (yt == 0)]
        if len(a) and len(b): ds.append(a.mean(0) - b.mean(0))
    d = np.mean(ds, 0); nrm = np.linalg.norm(d)
    return d / nrm if nrm > 0 else np.full(len(d), np.nan)


def main():
    warnings.filterwarnings('ignore', category=FutureWarning)
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); args = ap.parse_args()
    sfx = '_smoke' if args.smoke else ''
    stem = 'depositioned_direction_diagnosis' + sfx
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
    diag = json.loads((OUT / 'depositioned_subject_diagnosis.json').read_text(encoding='utf-8'))
    grp_of = {s: G for G in GROUPS for s in diag['diagnosis']['groups'][G]['subjects']}
    assert all(s in grp_of for s in subs)
    print(f'[start] rows={len(y)} subjects={len(subs)} smoke={args.smoke} sseeds={sseeds} R={R_STEP} MIN={MIN_PAST} '
          f'PCA={NPC} pseudo_q={PSEUDO_Q} clip=+-{CLIP} acc_bad={ACC_BAD} jobs={N_JOBS} '
          f'(acc_seed / auc_L_true / oracle flip / flipL are LEAKY)', flush=True)

    Xa, ya, ga, _ = arrange(subs, data, 0, 'independent'); fold_of = {}
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Xa, ya, ga)):
        for s in np.unique(ga[te]): fold_of[s] = k
    fold = np.array([fold_of[s] for s in g]); assert np.array_equal(fold, A_['fold'])
    print('[check] fold == depositioned_aperiodic_emg_oof.npz fold', flush=True)
    run_folds = [0] if args.smoke else list(range(5))
    ZZt = np.full((len(y), 889), np.nan); dhat_of = {}
    for k in run_folds:
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        tfo, tfn = impute_scale(Fc, tr), impute_scale(Fn, tr)
        ZZt[te] = np.hstack([tfo(Fc[te]), tfn(Fn[te])])
        dhat_of[k] = dir_fold(Fc, Fn, tr, y, g)
        print(f'[zz] fold {k} ZZ built ({len(te)} test rows); |d|={np.linalg.norm(dhat_of[k]):.3f} {el()}', flush=True)

    result = {'meta': {'refit_step': R_STEP, 'min_past': MIN_PAST, 'pca': NPC, 'pseudo_q': PSEUDO_Q, 'clip': CLIP,
                       'eps': EPS, 'acc_bad': ACC_BAD, 'mature_index': MATURE, 'threshold': .5,
                       'thr': {'k': THR_K, 'min_past': 16, 'mode': 'mean'}, 'bootstrap_n': nb,
                       'stream_seeds': list(sseeds), 'smoke': args.smoke, 'leaky_stats': list(LEAKY_STATS),
                       'detect_scores': [{'name': n_, 'stat': k, 'mult': m_} for n_, k, m_ in DETECT],
                       'legal_flip_candidates': list(CANDIDATES), 'legal_flip_threshold': 0.,
                       'split': 'pop_dir dhat = unit mean over training subjects of (task mean - relax mean) in ZZ'},
              'checks': {}}
    checks = result['checks']; sidx = {s: j for j, s in enumerate(subs)}
    run_subs = [s for s in subs if fold_of[s] in run_folds]
    for sd in sseeds:
        order, t = stream_order(g, subs, sd)
        Bl = np.c_[logit(A_[f'stream{sd}__all_prank']), logit(B_[f'stream{sd}__resnet_whiten_logit_delta']),
                   logit(B_[f'stream{sd}__resnet_fbank_logit_delta'])]
        L = Bl.mean(1); assert np.isfinite(L).all()
        ev = np.isin(fold, run_folds)
        print(f'[s{sd}] std(L)={L[ev].std():.3f} mean(L)={L[ev].mean():+.3f}', flush=True)
        Zn = {v: np.full(len(y), np.nan) for v in VARIANTS}; ST = {}
        ts = time.perf_counter()
        gen = Parallel(n_jobs=N_JOBS, backend='loky', return_as='generator')(
            delayed(adapt_subject)(ZZt[order[sidx[s]]], Bl[order[sidx[s]]], y[order[sidx[s]]], dhat_of[fold_of[s]])
            for s in run_subs)
        for i, (s, (r, st)) in enumerate(zip(run_subs, gen)):
            loc = order[sidx[s]]; ST[s] = mean_stats(st)
            for v in VARIANTS: Zn[v][loc] = r[v]
            if (i + 1) % 8 == 0 or i + 1 == len(run_subs):
                print(f'[s{sd}] subjects {i + 1}/{len(run_subs)} stream_s={time.perf_counter() - ts:.1f} {el()}', flush=True)
        # ---- checks
        for s in run_subs[:2]:
            loc = order[sidx[s]]; Z0, B0, y0 = ZZt[loc], Bl[loc], y[loc]; n = len(loc); dh = dhat_of[fold_of[s]]
            r0, st0 = adapt_subject(Z0, B0, y0, dh)
            for v in VARIANTS: assert np.array_equal(r0[v], Zn[v][loc], equal_nan=True), ('isolation', s, v)
            cut = MIN_PAST + R_STEP * ((n - MIN_PAST) // (2 * R_STEP)) + 5
            rng = np.random.default_rng(sidx[s]); Z1, B1 = Z0.copy(), B0.copy()
            Z1[cut:] += rng.normal(0, 2, Z1[cut:].shape); B1[cut:] += rng.normal(0, 2, B1[cut:].shape)
            r1, st1 = adapt_subject(Z1, B1, y0, dh)
            for v in VARIANTS:
                assert np.array_equal(r0[v][:cut], r1[v][:cut], equal_nan=True), ('prefix', s, v)
                a, b = r0[v][cut:], r1[v][cut:]; f = np.isfinite(a) & np.isfinite(b)
                assert f.any() and np.abs(a[f] - b[f]).max() > 0, ('prefix-changed', s, v)
            nst = 0
            for j, t0 in enumerate(range(MIN_PAST, n, R_STEP)):
                if t0 > cut: break
                for k in STATS:
                    assert np.nan_to_num(st0[j][k], nan=1e9) == np.nan_to_num(st1[j][k], nan=1e9), ('stat_prefix', s, k)
                nst += 1
            r2, st2 = adapt_subject(Z0, B0, np.random.default_rng(1).permutation(y0), dh)
            for v in VARIANTS: assert np.array_equal(r0[v], r2[v], equal_nan=True), ('label_shuffle', s, v)
            for j in range(len(st0)):
                for k in STATS:
                    if k in LEAKY_STATS: continue
                    assert np.nan_to_num(st0[j][k], nan=1e9) == np.nan_to_num(st2[j][k], nan=1e9), ('stat_shuffle', s, k)
            assert np.allclose(r0['pseudo_lr_flip'], -r0['pseudo_lr_full'], atol=1e-4, equal_nan=True), ('flip_sym', s)
            checks[f's{sd}_{s}'] = {'isolation': 'ok', 'prefix_invariance': 'ok', 'label_shuffle_identical': 'ok',
                                    'flip_symmetry': 'ok', 'cut': int(cut), 'stat_prefix_refits': nst}
            print(f'[check s{sd}] {s}: isolation / prefix (z + {nst} refits of statistics) / label shuffle / flip '
                  f'symmetry OK', flush=True)
        # ---- subject table
        ev_subs = list(np.unique(g[ev]))
        sub = {}
        for s in ev_subs:
            q = ST[s]; acc = q['acc_seed']; auc = q['auc_L_true']
            sub[s] = {'group': grp_of[s], **{k: q[k] for k in STATS},
                      'reversed_oracle': bool((np.isfinite(auc) and auc < .5) or acc < .5),
                      'bad': bool(acc < ACC_BAD)}
        nrev = sum(v['reversed_oracle'] for v in sub.values()); nbad = sum(v['bad'] for v in sub.values())
        print(f'[subjects s{sd}] n={len(sub)} reversed_oracle={nrev} acc_seed<{ACC_BAD}: {nbad} '
              f'(low reversed={sum(v["reversed_oracle"] for v in sub.values() if v["group"] == "low")}/'
              f'{sum(1 for v in sub.values() if v["group"] == "low")})', flush=True)
        for s in ev_subs:
            q = sub[s]
            print(f'[subject s{sd}] {s} {q["group"]:4s} acc_seed={q["acc_seed"]:.4f} auc_L_true={q["auc_L_true"]:.4f} '
                  f'pop_dir={q["pop_dir"]:+.3f} branch_L_pos={q["branch_L_pos"]:+.2f} kappa={q["kappa_gmm"]:+.3f} '
                  f'sori={q["sori_abs"]:.3f} ashman={q["ashman"]:.2f} dbic={q["dbic"]:+.1f} ac1={q["ac1"]:+.2f} '
                  f'drift={q["drift"]:+.2f}{" REV" if q["reversed_oracle"] else ""}', flush=True)
        # ---- detection
        bad = np.array([sub[s]['bad'] for s in ev_subs])
        det = {}
        for name, k, mult in DETECT:
            sc = np.array([sub[s][k] for s in ev_subs], float) * (mult if mult != 0 else 1.)
            a = det_auc(sc, bad); ci = boot_auc(sc, bad, nb)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                rho = float(spearmanr(sc, [sub[s]['acc_seed'] for s in ev_subs])[0])
            aa = None if a is None else max(a, 1. - a)
            det[name] = {'stat': k, 'mult': mult, 'AUC': a, 'AUC_abs_posthoc': aa, 'AUC_ci': ci,
                         'rho_acc_seed': rho if np.isfinite(rho) else None, 'exploratory': mult == 0,
                         'n_bad': int(bad.sum()), 'n': int(len(bad))}
            print(f'[detect s{sd}] {name:14s} AUC={a if a is None else round(a, 4)}'
                  f'[{ci["p2.5"]:.3f},{ci["p97.5"]:.3f}] rho(acc_seed)={rho:+.3f}'
                  f'{" (exploratory, direction post hoc: |AUC-.5| -> " + str(round(aa, 4)) + ")" if mult == 0 else ""}',
                  flush=True)
        gstat = {}
        for k in STATS:
            gstat[k] = {G: (float(np.mean([sub[s][k] for s in ev_subs if sub[s]['group'] == G]))
                            if any(sub[s]['group'] == G for s in ev_subs) else None) for G in GROUPS}
            print(f'[stat-group s{sd}] {k:12s} ' + ' '.join(f'{G}={gstat[k][G]:+.3f}' if gstat[k][G] is not None
                                                            else f'{G}=NA' for G in GROUPS), flush=True)
        # ---- scores
        def flip_rows(flag):
            m = np.zeros(len(y), bool)
            for s in ev_subs:
                if flag[s]: m |= (g == s)
            return m
        rv = flip_rows({s: sub[s]['reversed_oracle'] for s in ev_subs})
        z = np.where(rv, Zn['pseudo_lr_flip'], Zn['pseudo_lr_full'])
        S = {'fuseH_wf': sig(L), 'fuseH_wf_thr': sig(thr_apply(np.where(ev, L, np.nan), order, THR_K, 'mean')),
             'fuse_pseudo_lr_full': sig(np.where(ev, np.where(np.isfinite(Zn['pseudo_lr_full']),
                                                              L + Zn['pseudo_lr_full'], L), np.nan)),
             'fuse_oracle_flip': sig(np.where(ev, np.where(np.isfinite(z), L + z, L), np.nan)),
             'fuse_flipL': sig(np.where(ev, np.where(rv, -L, L), np.nan))}
        lf = np.where(ev, np.where(np.isfinite(Zn['pseudo_lr_full']), L + Zn['pseudo_lr_full'], L), np.nan)
        S['fuse_pseudo_lr_full_thr'] = sig(thr_apply(lf, order, THR_K, 'mean'))
        S['fuse_oracle_flip_thr'] = sig(thr_apply(np.where(ev, np.where(np.isfinite(z), L + z, L), np.nan),
                                                  order, THR_K, 'mean'))
        S['fuse_flipL_thr'] = sig(thr_apply(np.where(ev, np.where(rv, -L, L), np.nan), order, THR_K, 'mean'))
        # ---- legal flip (best pre-registered legal detector, threshold 0)
        best = max(CANDIDATES, key=lambda n_: (det[n_]['AUC'] if det[n_]['AUC'] is not None else -.1))
        picked = None; ci_lo = det[best]['AUC_ci']['p2.5']
        if det[best]['AUC'] is not None and ci_lo > .5:
            k = det[best]['stat']; mult = det[best]['mult']
            flag = {s: bool(mult * sub[s][k] > 0) for s in ev_subs}
            re = flip_rows(flag)
            ze = np.where(re, Zn['pseudo_lr_flip'], Zn['pseudo_lr_full'])
            S['fuse_flipE'] = sig(np.where(ev, np.where(np.isfinite(ze), L + ze, L), np.nan))
            lfe = np.where(ev, np.where(np.isfinite(ze), L + ze, L), np.nan)
            S['fuse_flipE_thr'] = sig(thr_apply(lfe, order, THR_K, 'mean'))
            picked = {'stat': k, 'mult': mult, 'threshold': 0., 'n_flipped': int(sum(flag.values())),
                      'overlap_with_oracle': int(sum(flag[s] and sub[s]['reversed_oracle'] for s in ev_subs)),
                      'legal_AUC': det[best]['AUC'], 'legal_AUC_ci': det[best]['AUC_ci'],
                      'note': 'flipped iff mult*subject-mean statistic > 0'}
            print(f'[legal-flip s{sd}] {best}: flips {picked["n_flipped"]} subjects, overlap with oracle reversed = '
                  f'{picked["overlap_with_oracle"]}', flush=True)
        else:
            print(f'[legal-flip s{sd}] skipped: best legal detector {best} AUC={det[best]["AUC"]} '
                  f'CI_lo={ci_lo:.3f} <= .5', flush=True)
        for n_ in S: S[n_] = np.where(ev, S[n_], np.nan)
        M = {'all': ev, 'mature': ev & (t >= MATURE)}
        R = result[f'stream{sd}'] = {'detect': det, 'stat_groups': gstat, 'subjects': sub, 'legal_flip': picked,
                                     'n_reversed_oracle': nrev, 'n_bad': nbad,
                                     'methods': {}, 'bootstrap': {}, 'groups': {}, 'subject_scores': {}}
        PS = {}
        for n_, p in S.items():
            r = R['methods'][n_] = {s: metrics(y, g, p, M[s]) for s in M}
            PS[n_] = {s: metrics(y, g, p, ev & (g == s))['subject_macro_bACC'] for s in ev_subs}
            ba = np.array([q for q in PS[n_].values() if q is not None])
            R['subject_scores'][n_] = {'n_ge_0.9': int((ba >= .9).sum()), 'max': float(ba.max()), 'n': int(len(ba))}
            tag = ' [LEAKY]' if any(x in n_ for x in ('oracle', 'flipL')) else ''
            print(f'[result stream {sd}] {n_:22s} ' + ' | '.join(
                f'{s}: AUC={r[s]["AUC"]:.4f} mac={r[s]["subject_macro_bACC"]:.4f} mic={r[s]["micro_bACC"]:.4f}'
                for s in M) + f' | subj>=.9={R["subject_scores"][n_]["n_ge_0.9"]} '
                f'max={R["subject_scores"][n_]["max"]:.3f}{tag}', flush=True)
        for suf in ('', '_thr'):
            ref = v2[f'stream{sd}']['methods'][f'fuse_pseudo_lr_full{suf}']['all']['subject_macro_bACC']
            got = R['methods'][f'fuse_pseudo_lr_full{suf}']['all']['subject_macro_bACC']
            assert abs(got - ref) < 1e-9, ('v2 reproduction', suf, got, ref)
            checks[f's{sd}_v2_reproduced{suf}'] = ref
        print(f'[check s{sd}] fuse_pseudo_lr_full(_thr) reproduces v2 mac='
              f'{checks[f"s{sd}_v2_reproduced"]:.4f}/{checks[f"s{sd}_v2_reproduced_thr"]:.4f}', flush=True)
        for n_ in S:
            R['groups'][n_] = {}
            for G in GROUPS:
                ss = [s for s in ev_subs if grp_of.get(s) == G and PS[n_][s] is not None]
                if not ss: continue
                a = np.array([PS[n_][s] for s in ss]); b = np.array([PS['fuse_pseudo_lr_full_thr'][s] for s in ss])
                R['groups'][n_][G] = {'n': len(ss), 'mean': float(a.mean()), 'median': float(np.median(a)),
                                      'd_mean_vs_ref_thr': float(a.mean() - b.mean())}
            print(f'[group stream {sd}] {n_:22s} ' + ' | '.join(
                f'{G}(n={q["n"]}): mean={q["mean"]:.4f} dmean={q["d_mean_vs_ref_thr"]:+.4f}'
                for G, q in R['groups'][n_].items()), flush=True)
        pairs = [(n_, 'fuse_pseudo_lr_full') for n_ in ('fuse_oracle_flip', 'fuse_flipL', 'fuse_flipE') if n_ in S]
        pairs += [(n_, 'fuse_pseudo_lr_full_thr') for n_ in ('fuse_oracle_flip_thr', 'fuse_flipL_thr', 'fuse_flipE_thr')
                  if n_ in S]
        for a_, b_ in pairs:
            for seg in M:
                bb = R['bootstrap'][f'{a_}__vs__{b_}__{seg}'] = boot(S[a_], S[b_], y, g, M[seg], nb)
                print(f'[boot stream {sd}] {a_:22s} vs {b_:22s} {seg:6s} '
                      f'dAUC={bb["dAUC"]["mean"]:+.4f}[{bb["dAUC"]["p2.5"]:+.4f},{bb["dAUC"]["p97.5"]:+.4f}] '
                      f'dmac={bb["dmacro_bACC"]["mean"]:+.4f}[{bb["dmacro_bACC"]["p2.5"]:+.4f},{bb["dmacro_bACC"]["p97.5"]:+.4f}] '
                      f'dmic={bb["dmicro_bACC"]["mean"]:+.4f}[{bb["dmicro_bACC"]["p2.5"]:+.4f},{bb["dmicro_bACC"]["p97.5"]:+.4f}]',
                      flush=True)
        print(f'[s{sd}] done {el()}', flush=True)
        (OUT / (stem + '.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    # ---- summary over streams
    st = [result[f'stream{sd}'] for sd in sseeds]; summ = {}
    for n_ in st[0]['methods']:
        summ[n_] = {'mac': float(np.mean([r['methods'][n_]['all']['subject_macro_bACC'] for r in st])),
                    'AUC': float(np.mean([r['methods'][n_]['all']['AUC'] for r in st])),
                    'mature_mac': float(np.mean([r['methods'][n_]['mature']['subject_macro_bACC'] for r in st])),
                    'n_ge_0.9': int(np.mean([r['subject_scores'][n_]['n_ge_0.9'] for r in st]))}
        if n_ in ('fuse_oracle_flip', 'fuse_flipL', 'fuse_flipE'):
            bk = f'{n_}__vs__fuse_pseudo_lr_full__all'
            for m_ in ('dmacro_bACC', 'dAUC'):
                summ[n_][m_] = [float(np.mean([r['bootstrap'][bk][m_][p] for r in st])) for p in ('mean', 'p2.5', 'p97.5')]
            summ[n_]['low_dmac'] = float(np.mean([r['groups'][n_].get('low', {}).get('d_mean_vs_ref_thr', np.nan) for r in st]))
        elif n_ in ('fuse_oracle_flip_thr', 'fuse_flipL_thr', 'fuse_flipE_thr'):
            bk = f'{n_}__vs__fuse_pseudo_lr_full_thr__all'
            for m_ in ('dmacro_bACC', 'dAUC'):
                summ[n_][m_] = [float(np.mean([r['bootstrap'][bk][m_][p] for r in st])) for p in ('mean', 'p2.5', 'p97.5')]
            summ[n_]['low_dmac'] = float(np.mean([r['groups'][n_].get('low', {}).get('d_mean_vs_ref_thr', np.nan) for r in st]))
        print(f'[summary] {n_:22s} mac={summ[n_]["mac"]:.4f} AUC={summ[n_]["AUC"]:.4f} '
              f'mature_mac={summ[n_]["mature_mac"]:.4f}'
              + (f' dmac={summ[n_]["dmacro_bACC"]} dAUC={summ[n_]["dAUC"]} low_dmac={summ[n_]["low_dmac"]:+.4f}'
                 if 'dmacro_bACC' in summ[n_] else ''), flush=True)
    det_s = {}
    for name, k, mult in DETECT:
        det_s[name] = {'AUC': float(np.mean([r['detect'][name]['AUC'] for r in st])),
                       'AUC_abs_posthoc': float(np.mean([r['detect'][name]['AUC_abs_posthoc'] for r in st])),
                       'AUC_ci_lo': float(np.mean([r['detect'][name]['AUC_ci']['p2.5'] for r in st])),
                       'AUC_ci_hi': float(np.mean([r['detect'][name]['AUC_ci']['p97.5'] for r in st])),
                       'rho_acc_seed': float(np.mean([r['detect'][name]['rho_acc_seed'] for r in st]))
                       if all(r['detect'][name]['rho_acc_seed'] is not None for r in st) else None}
        print(f'[summary-detect] {name:14s} AUC={det_s[name]["AUC"]:.4f} '
              f'abs={det_s[name]["AUC_abs_posthoc"]:.4f} '
              f'[{det_s[name]["AUC_ci_lo"]:.3f},{det_s[name]["AUC_ci_hi"]:.3f}] '
              f'rho={det_s[name]["rho_acc_seed"]}', flush=True)
    for k in STATS:
        det_s[f'group_mean_{k}'] = {G: float(np.mean([r['stat_groups'][k][G] for r in st
                                                      if r['stat_groups'][k][G] is not None]))
                                    for G in GROUPS if any(r['stat_groups'][k][G] is not None for r in st)}
    summ['detect'] = det_s
    summ['legal_flip'] = [r['legal_flip'] for r in st]
    summ['n_reversed_oracle'] = [r['n_reversed_oracle'] for r in st]
    result['summary'] = summ; result['seconds'] = time.perf_counter() - T0
    (OUT / (stem + '.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[done] all checks passed {el()} -> {stem}.json', flush=True)
    sys.stdout = tee.stream; tee.f.close()


if __name__ == '__main__':
    main()
