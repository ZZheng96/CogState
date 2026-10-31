"""Direction H: 1/f aperiodic + periodic-residual + EMG/high-frequency features per 6 s window (de-positioned streams).

Protocol identical to depositioned_feature_multiscale.py: stream order = concat(relax,task) permuted by
default_rng([stream_seed, subject_index]); outer folds = SGKF(5, rs0) on seed-0 arrangement (re-derived, asserted == fold in
depositioned_feature_multiscale_oof.npz). Per fold: train-median imputation + StandardScaler(train). No feature selection.
New features use only the current window's raw samples (load_raw X, 1500 pts @ 250 Hz) -> causal by construction; history
features (prank form) use previous windows of the same subject only; no labels of test subjects are used.

Data pre-check (done before fixing hyper-parameters, repeated + logged by the script): X float32 (N,1500) = 6 s @ 250 Hz;
Welch spectra decay smoothly to 125 Hz (no low-pass); 50 Hz (and 100 Hz) line peaks present in some subjects (no notch).
-> high band fixed to HF = 55-95 Hz (avoids 50 Hz and 100 Hz harmonic, < fs/2). If the logged global check finds the
data low-passed (log10 P(55-95)/P(30-40) < -3) the HF/EMG features are dropped (1/f part only). The npz key 'e'
(manifest emg_feature_names) is NOT used.

Fixed feature hyper-parameters:
  Welch fs=250, nperseg=500 (0.5 Hz bins), noverlap=250, hann; P floored at 1e-30 before log10.
  Aperiodic fit range 2-40 Hz (inclusive), x=log10 f, y=log10 P.
   (a) fixed (no knee) robust: OLS line; drop points with residual > 80th percentile of residuals; refit OLS.
       -> ap_offset, ap_exponent (=-slope), ap_r2, ap_rmse (r2/rmse on all fit-range points).
   (b) knee: log10 P = b - log10(k + f^chi), scipy curve_fit, p0=(offset_a, 1, clip(exp_a,.05,5.9)),
       bounds b[-40,20] k[0,1e4] chi[.01,6], maxfev=2000, all fit-range points; on failure / non-finite -> (a) values
       with k=0 and kn_fail=1.  -> kn_offset, kn_logknee=log10(k+1), kn_exponent, kn_r2, kn_rmse, kn_fail.
  Periodic residual r = y - fit(a) on theta 4-8, alpha 8-13, beta1 13-20, beta2 20-30, gamlow 30-40 (lo<=f<hi):
       max(r), peak freq argmax, mean(r)  -> 15 features.
  EMG/HF: hf_logpow = log10 mean P(55-95); hf_rel = log10(sum P(55-95) / sum P(2-40)); hf_slope = OLS slope of
       log10 P vs log10 f on 40-95 Hz excluding 45-55 Hz; hf_lf = log10(sum P(55-95)/sum P(2-12));
       time domain: Butterworth order-4 band-pass 55-95 Hz, sosfiltfilt inside the window -> hf_logrms, hf_kurt (Fisher).
  Non-finite values -> nan -> fold train-median imputation.
Models (HGB = make_clf('hgb') = HGB(200, depth3, lr .06, l2 1, rs0); LR(C=.05)):
  new_lr [zn] LR | new_hgb [zn] HGB | new_prank feat_subject(zn,'hgb_prank') HGB |
  all_prank feat_subject([z858, zn],'hgb_prank') HGB | hgb_prank recomputed (assert == stored, max|d|<1e-9).
  fusions sigmoid(mean logit): fuse_newprank=(fusionM_prank, new_prank); fuse_allprank_resnet=(all_prank,
  resnet_whiten_logit_delta from depositioned_resnet_histcontext_oof.npz); fuse3=(fusionM_prank, new_hgb).
  fusionM_prank taken from depositioned_feature_multiscale_oof.npz (asserted == sig(mean logit(hgb_prank, whiten))).
Evaluation: all / mature (t>=120); AUC, subject-macro bACC, micro bACC at .5; paired subject bootstrap (1000; smoke 100)
vs fusionM_prank; per group (depositioned_subject_diagnosis.json high/mid/low) subject macro bACC median/mean and change
vs fusionM_prank; #subjects with bACC>=.9.
Diagnostics (label-using, analysis only, not a method): within-subject SKF(5, shuffle, rs0) LR(C=.05) on new features
-> per-subject AUC, group medians; Cohen's d per subject x feature; direction consistency = corr(d_s, mean d of other
subjects); group mean d per feature.
Checks (assert): position-only AUC ~.5 (|d|<.03, full run); subject isolation (alone == pooled, raw->features->scores);
prefix invariance (future raw windows changed -> prefix features + scores identical, future changed); test-label shuffle
-> identical scores; fold assert; hgb_prank reproduction.
"""
import argparse, json, sys, time, warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import curve_fit
from scipy.signal import welch, butter, sosfiltfilt
from scipy.stats import kurtosis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from depositioned_history_delta import arrange, make_clf, boot, load_db  # noqa: E402
from depositioned_resnet_history_delta import load_raw, stream_order, logit, sig  # noqa: E402
from depositioned_feature_multiscale import feat_subject  # noqa: E402
from causal_kernel_state_mapping import metrics  # noqa: E402
from switch_dual_memory import Tee  # noqa: E402

FS, NPERSEG, NOVER = 250., 500, 250
FIT_LO, FIT_HI, PEAK_Q = 2., 40., .8
KNEE_P0K, KNEE_LB, KNEE_UB, MAXFEV = 1., (-40., 0., .01), (20., 1e4, 6.), 2000
BANDS = (('theta', 4, 8), ('alpha', 8, 13), ('beta1', 13, 20), ('beta2', 20, 30), ('gamlow', 30, 40))
HF, LF, HF_SLOPE, LINE_EXCL, BP_ORDER = (55., 95.), (2., 12.), (40., 95.), (45., 55.), 4
LOWPASS_THR = -3.
MATURE, STREAM_SEEDS, N_JOBS, CHUNK = 120, (0, 1, 2), 12, 256
SOS = butter(BP_ORDER, HF, btype='bandpass', fs=FS, output='sos')
AP_NAMES = (['ap_offset', 'ap_exponent', 'ap_r2', 'ap_rmse', 'kn_offset', 'kn_logknee', 'kn_exponent', 'kn_r2', 'kn_rmse',
             'kn_fail'] + [f'res_{b}_{q}' for b, _, _ in BANDS for q in ('max', 'pkf', 'mean')])
HF_NAMES = ['hf_logpow', 'hf_rel', 'hf_slope', 'hf_lf', 'hf_logrms', 'hf_kurt']
T0 = time.perf_counter()


def el():
    return f'elapsed={time.perf_counter() - T0:.1f}s'


def _knee(f, b, k, chi):
    return b - np.log10(k + f ** chi)


def _r2_rmse(y, yh):
    r = y - yh; ss = ((y - y.mean()) ** 2).sum()
    return (1 - (r * r).sum() / ss if ss > 0 else np.nan), float(np.sqrt((r * r).mean()))


def feats_chunk(x):
    """x (n, 1500) raw windows -> (n, 31) features; each row uses only its own window."""
    x = np.asarray(x, np.float64)
    f, P = welch(x, fs=FS, window='hann', nperseg=NPERSEG, noverlap=NOVER, axis=-1)
    P = np.maximum(P, 1e-30); lp = np.log10(P)
    m = (f >= FIT_LO) & (f <= FIT_HI); fx = f[m]; X1 = np.log10(fx); A = np.c_[np.ones_like(X1), X1]
    hf = (f >= HF[0]) & (f < HF[1]); lfm = (f >= LF[0]) & (f < LF[1]); bb = (f >= FIT_LO) & (f < FIT_HI)
    sm = (f >= HF_SLOPE[0]) & (f < HF_SLOPE[1]) & ~((f >= LINE_EXCL[0]) & (f < LINE_EXCL[1]))
    As = np.c_[np.ones(sm.sum()), np.log10(f[sm])]
    xb = sosfiltfilt(SOS, x, axis=-1)
    out = np.full((len(x), len(AP_NAMES) + len(HF_NAMES)), np.nan)
    for i in range(len(x)):
        Y = lp[i, m]
        c = np.linalg.lstsq(A, Y, rcond=None)[0]; r = Y - A @ c
        keep = r <= np.quantile(r, PEAK_Q)
        c = np.linalg.lstsq(A[keep], Y[keep], rcond=None)[0]; yh = A @ c
        r2a, rma = _r2_rmse(Y, yh); off, ex = c[0], -c[1]
        kn = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                p, _ = curve_fit(_knee, fx, Y, p0=(off, KNEE_P0K, float(np.clip(ex, .05, 5.9))),
                                 bounds=(KNEE_LB, KNEE_UB), maxfev=MAXFEV)
            yk = _knee(fx, *p)
            if np.all(np.isfinite(p)) and np.all(np.isfinite(yk)):
                kn = [p[0], np.log10(p[1] + 1), p[2], *_r2_rmse(Y, yk), 0.]
        except (RuntimeError, ValueError):
            pass
        if kn is None: kn = [off, 0., ex, r2a, rma, 1.]
        res = lp[i] - (c[0] + c[1] * np.log10(np.maximum(f, 1e-12)))
        per = []
        for _, lo, hi in BANDS:
            bm = (f >= lo) & (f < hi); rb = res[bm]; j = int(np.argmax(rb))
            per += [rb[j], f[bm][j], rb.mean()]
        hs = np.linalg.lstsq(As, lp[i, sm], rcond=None)[0][1]
        ph = P[i, hf].sum()
        hfv = [np.log10(P[i, hf].mean()), np.log10(ph / P[i, bb].sum()), hs, np.log10(ph / P[i, lfm].sum()),
               np.log10(np.sqrt((xb[i] ** 2).mean()) + 1e-30), kurtosis(xb[i])]
        out[i] = [off, ex, r2a, rma] + kn + per + hfv
    out[~np.isfinite(out)] = np.nan
    return out


def extract(X, tag):
    idx = list(range(0, len(X), CHUNK)); res = [None] * len(idx); t1 = time.perf_counter()
    gen = Parallel(n_jobs=N_JOBS, return_as='generator')(delayed(feats_chunk)(X[a:a + CHUNK]) for a in idx)
    for i, r in enumerate(gen):
        res[i] = r
        if i % 8 == 0 or i == len(idx) - 1:
            print(f'[feat {tag} {i + 1}/{len(idx)}] chunk_s={time.perf_counter() - t1:.1f} {el()}', flush=True)
    return np.vstack(res)


def data_check(X, g, subs):
    """Welch (same params) median spectrum of selected subjects + global sample; filter / line / HF diagnostics."""
    def spec(x):
        f, P = welch(np.asarray(x, np.float64), fs=FS, window='hann', nperseg=NPERSEG, noverlap=NOVER, axis=-1)
        return f, np.median(P, 0)
    def lb(f, P, lo, hi): return float(np.log10(P[(f >= lo) & (f < hi)].mean()))
    out = {'shape': list(X.shape), 'dtype': str(X.dtype), 'n_samples_per_window': int(X.shape[1]),
           'expected_samples': int(FS * 6), 'subjects': {}}
    rng = np.random.default_rng(0); samp = rng.choice(len(X), 2000, replace=False)
    groups = [(s, np.flatnonzero(g == s)) for s in subs[::8]] + [('global2000', samp)]
    for s, loc in groups:
        f, P = spec(X[loc]); bands = [(2, 12), (12, 30), (30, 40), (40, 45), (45, 48), (49.5, 50.5), (52, 55), (55, 65),
                                       (65, 80), (80, 95), (99.5, 100.5), (105, 115), (115, 125.1)]
        lv = {f'{a}-{b}': lb(f, P, a, b) for a, b in bands}
        nb50 = np.log10(np.median(P[((f >= 46) & (f < 49)) | ((f > 51) & (f <= 54))]))
        nb100 = np.log10(np.median(P[((f >= 96) & (f < 99)) | ((f > 101) & (f <= 104))]))
        d = {'log10P': lv, 'line50_dB': 10 * (lv['49.5-50.5'] - nb50), 'line100_dB': 10 * (lv['99.5-100.5'] - nb100),
             'hf55_95_vs_30_40_log10': lb(f, P, 55, 95) - lv['30-40'], 'p115_125_vs_30_40_log10': lv['115-125.1'] - lv['30-40']}
        out['subjects'][str(s)] = d
        print(f'[data-check] {s:10s} ' + ' '.join(f'{k}:{v:.2f}' for k, v in lv.items()) +
              f' | line50={d["line50_dB"]:+.1f}dB line100={d["line100_dB"]:+.1f}dB '
              f'HF55-95/30-40={d["hf55_95_vs_30_40_log10"]:+.2f} 115-125/30-40={d["p115_125_vs_30_40_log10"]:+.2f}', flush=True)
    gl = out['subjects']['global2000']
    out['lowpassed'] = bool(gl['hf55_95_vs_30_40_log10'] < LOWPASS_THR)
    out['hf_band'] = list(HF) if not out['lowpassed'] else None
    print(f'[data-check] shape={X.shape} dtype={X.dtype} (expect 1500=6s@250Hz) lowpassed={out["lowpassed"]} '
          f'line50_global={gl["line50_dB"]:+.1f}dB -> HF band {out["hf_band"]} (avoid 50/100 Hz, < fs/2=125)', flush=True)
    return out


def impute_scale(F, tr):
    med = np.nanmedian(np.where(np.isfinite(F[tr]), F[tr], np.nan), 0); med = np.where(np.isfinite(med), med, 0.)
    sc = StandardScaler().fit(np.where(np.isfinite(F), F, med)[tr])
    return lambda A: sc.transform(np.where(np.isfinite(A), A, med))


def feat_all_prank(Z, order):
    out = np.empty((len(Z), 3 * Z.shape[1]))
    for loc in order: out[loc] = feat_subject(Z[loc], 'hgb_prank')
    return out


def diagnostics(Fn, y, g, subs, names, grp_of):
    print(f'[diag] within-subject LR + Cohen d on {len(names)} new features {el()}', flush=True)
    D = {}; auc = {}
    med = np.nanmedian(Fn, 0); Fi = np.where(np.isfinite(Fn), Fn, med)
    for s in subs:
        loc = np.flatnonzero(g == s); yy = y[loc]; p = np.zeros(len(loc))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(loc, yy):
            tf = impute_scale(Fn[loc], tr); A = tf(Fn[loc])
            p[te] = make_clf('lr').fit(A[tr], yy[tr]).predict_proba(A[te])[:, 1]
        auc[s] = float(roc_auc_score(yy, p))
        a, b = Fi[loc][yy == 0], Fi[loc][yy == 1]
        with np.errstate(invalid='ignore', divide='ignore'):
            sp = np.sqrt(((len(a) - 1) * a.var(0, ddof=1) + (len(b) - 1) * b.var(0, ddof=1)) / (len(a) + len(b) - 2))
            D[s] = np.where(sp > 0, (b.mean(0) - a.mean(0)) / sp, np.nan)
    Dm = np.array([D[s] for s in subs]); cons = {}
    for j, s in enumerate(subs):
        other = np.nanmean(np.delete(Dm, j, 0), 0); ok = np.isfinite(Dm[j]) & np.isfinite(other)
        cons[s] = float(np.corrcoef(Dm[j][ok], other[ok])[0, 1])
    res = {'within_auc': auc, 'consistency': cons, 'groups': {}}
    for G in ('high', 'mid', 'low', 'all'):
        ss = [s for s in subs if G == 'all' or grp_of.get(s) == G]
        if not ss: continue
        gd = np.nanmean(np.array([D[s] for s in ss]), 0)
        res['groups'][G] = {'n': len(ss), 'within_auc_median': float(np.median([auc[s] for s in ss])),
                            'consistency_median': float(np.nanmedian([cons[s] for s in ss])),
                            'consistency_frac_pos': float(np.mean([cons[s] > 0 for s in ss])),
                            'mean_d': {n: float(v) for n, v in zip(names, gd)}}
        r = res['groups'][G]
        print(f'[diag] {G:4s} n={len(ss):2d} within-subj AUC median={r["within_auc_median"]:.4f} '
              f'd-consistency median={r["consistency_median"]:+.3f} frac>0={r["consistency_frac_pos"]:.2f}', flush=True)
    for j, n in enumerate(names):
        print(f'[diag d] {n:16s} ' + ' '.join(f'{G}={res["groups"][G]["mean_d"][n]:+.3f}' for G in res['groups']), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); args = ap.parse_args()
    stem = 'depositioned_aperiodic_emg' + ('_smoke' if args.smoke else '')
    tee = Tee(sys.stdout, OUT / (stem + '.log')); sys.stdout = tee
    sseeds = STREAM_SEEDS[:1] if args.smoke else STREAM_SEEDS
    nb = 100 if args.smoke else 1000
    man, subs, data = load_db()
    Xraw, Fc, y, g = load_raw(subs)
    assert np.array_equal(Fc, np.vstack([data[s][0] for s in subs])) and np.array_equal(y, np.concatenate([data[s][1] for s in subs]))
    assert man['fs'] == FS and Xraw.shape[1] == int(FS * man['window_seconds']) == 1500
    prior = np.load(OUT / 'depositioned_feature_multiscale_oof.npz')
    prior_rn = np.load(OUT / 'depositioned_resnet_histcontext_oof.npz')
    assert np.array_equal(prior['y'], y) and np.array_equal(prior['groups'], g)
    diag = json.loads((OUT / 'depositioned_subject_diagnosis.json').read_text(encoding='utf-8'))
    grp_of = {s: G for G in ('high', 'mid', 'low') for s in diag['diagnosis']['groups'][G]['subjects']}
    print(f'[start] rows={len(y)} subjects={len(subs)} smoke={args.smoke} sseeds={sseeds} '
          f'groups={ {G: sum(v == G for v in grp_of.values()) for G in ("high", "mid", "low")} }', flush=True)
    dc = data_check(Xraw, g, subs)
    names = AP_NAMES + ([] if dc['lowpassed'] else HF_NAMES)

    Fn = extract(Xraw, 'all')[:, :len(names)]
    nanfrac = np.isnan(Fn).mean(0)
    print(f'[feat] dim={Fn.shape[1]} nan_frac max={nanfrac.max():.4f} ({names[int(nanfrac.argmax())]}) '
          f'knee_fail_frac={np.nanmean(Fn[:, names.index("kn_fail")]):.4f} '
          f'ap_exponent median={np.nanmedian(Fn[:, 1]):.3f} ap_r2 median={np.nanmedian(Fn[:, 2]):.3f} {el()}', flush=True)

    Xa, ya, ga, _ = arrange(subs, data, 0, 'independent'); fold_of = {}
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Xa, ya, ga)):
        for s in np.unique(ga[te]): fold_of[s] = k
    fold = np.array([fold_of[s] for s in g]); assert np.array_equal(fold, prior['fold'])
    print('[check] fold == depositioned_feature_multiscale_oof.npz fold', flush=True)
    run_folds = [0] if args.smoke else list(range(5))

    result = {'meta': {'mature_index': MATURE, 'threshold': .5, 'rows': int(len(y)), 'subjects': len(subs),
                       'stream_seeds': list(sseeds), 'smoke': args.smoke, 'new_feature_names': names,
                       'welch': {'fs': FS, 'nperseg': NPERSEG, 'noverlap': NOVER, 'window': 'hann'},
                       'fit_range': [FIT_LO, FIT_HI], 'peak_quantile': PEAK_Q, 'knee_bounds': [KNEE_LB, KNEE_UB],
                       'maxfev': MAXFEV, 'bands': BANDS, 'hf_band': HF, 'lf_band': LF, 'hf_slope_band': HF_SLOPE,
                       'line_excl': LINE_EXCL, 'bp_order': BP_ORDER, 'bootstrap_n': nb},
              'data_check': dc, 'feature_stats': {'nan_frac': dict(zip(names, map(float, nanfrac)))}, 'checks': {}}
    checks = result['checks']
    result['diagnostics'] = diagnostics(Fn, y, g, subs, names, grp_of)
    oof = {'y': y, 'groups': g, 'fold': fold, 'new_features': Fn}
    sidx = {s: j for j, s in enumerate(subs)}; RAWC = {}
    for sd in sseeds:
        order, t = stream_order(g, subs, sd)
        base = ['position_only__lr', 'position_only__hgb', 'hgb_prank', 'new_lr', 'new_hgb', 'new_prank', 'all_prank']
        S = {n: np.full(len(y), np.nan) for n in base}
        with threadpool_limits(limits=8):
            for k in run_folds:
                tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k); te_subs = list(np.unique(g[te]))
                T = t[:, None].astype(float)
                for c in ('lr', 'hgb'):
                    S[f'position_only__{c}'][te] = make_clf(c).fit(T[tr], y[tr]).predict_proba(T[te])[:, 1]
                tfo, tfn = impute_scale(Fc, tr), impute_scale(Fn, tr)
                Z, Zn = tfo(Fc), tfn(Fn)
                t1 = time.perf_counter(); Fp = feat_all_prank(Z, order)
                S['hgb_prank'][te] = make_clf('hgb').fit(Fp[tr], y[tr]).predict_proba(Fp[te])[:, 1]; del Fp
                dmax = float(np.abs(S['hgb_prank'][te] - prior[f'stream{sd}__hgb_prank'][te]).max())
                assert dmax < 1e-9, dmax; checks[f'hgb_prank_reproduced_s{sd}_f{k}'] = dmax
                print(f'[s{sd} fold {k + 1}/5] hgb_prank reproduced (maxabs {dmax:.1e}) fit_s={time.perf_counter() - t1:.1f} {el()}', flush=True)
                feats = {'new_lr': (lambda: Zn, 'lr'), 'new_hgb': (lambda: Zn, 'hgb'),
                         'new_prank': (lambda: feat_all_prank(Zn, order), 'hgb'),
                         'all_prank': (lambda: feat_all_prank(np.hstack([Z, Zn]), order), 'hgb')}
                for v, (mk, c) in feats.items():
                    t1 = time.perf_counter(); F = mk(); clf = make_clf(c).fit(F[tr], y[tr])
                    S[v][te] = clf.predict_proba(F[te])[:, 1]; tf = time.perf_counter() - t1
                    if v != 'all_prank' or k == run_folds[0]:  # test-label shuffle -> identical scores
                        ys = y.copy(); ys[te] = np.random.default_rng(k).permutation(y[te])
                        d_sh = float(np.abs(make_clf(c).fit(F[tr], ys[tr]).predict_proba(F[te])[:, 1] - S[v][te]).max())
                        assert d_sh == 0., d_sh; checks[f'label_shuffle_s{sd}_f{k}_{v}'] = d_sh
                    # raw -> features -> scores: subject isolation + prefix invariance (2 test subjects)
                    for s0 in te_subs[:2]:
                        loc = order[sidx[s0]]; cut = len(loc) // 2
                        if (sd, s0) not in RAWC:  # raw->feature recompute is fold/method independent: cache per stream/subject
                            Xalt = Xraw[loc].astype(np.float64); Xalt[cut:] = Xalt[cut:] * -3. + 1e-3 * np.sin(np.arange(1500) / 3.)
                            RAWC[(sd, s0)] = (feats_chunk(Xraw[loc])[:, :len(names)], feats_chunk(Xalt)[:, :len(names)])
                        fn0, fna = RAWC[(sd, s0)]
                        assert np.array_equal(np.isnan(fn0), np.isnan(Fn[loc])) and np.array_equal(np.nan_to_num(fn0), np.nan_to_num(Fn[loc]))
                        assert np.array_equal(np.nan_to_num(fna[:cut]), np.nan_to_num(fn0[:cut]))
                        za = Z[loc].copy(); za[cut:] += 100.
                        def build(zo, fnn, full=True):
                            zn = tfn(fnn)
                            if v in ('new_lr', 'new_hgb'): return zn
                            return feat_subject(zn if v == 'new_prank' else np.hstack([zo, zn]), 'hgb_prank')
                        alone = build(Z[loc], fn0); assert np.array_equal(alone, F[loc])
                        Fa = build(za, fna); assert np.array_equal(Fa[:cut], F[loc][:cut])
                        assert np.array_equal(build(Z[loc][:cut], fn0[:cut]), F[loc][:cut])
                        pa = clf.predict_proba(Fa)[:, 1]
                        d_pre = float(np.abs(pa[:cut] - S[v][loc[:cut]]).max()); d_fut = float(np.abs(pa[cut:] - S[v][loc[cut:]]).max())
                        assert d_pre == 0. and d_fut > 1e-6, (d_pre, d_fut)
                        checks[f'causal_s{sd}_f{k}_{v}_{s0}'] = {'score_prefix_after_future_change': d_pre, 'score_future_changed': d_fut}
                    s1 = np.unique(g[tr])[0]; loc = order[sidx[s1]]
                    if ('tr', sd, s1) not in RAWC: RAWC[('tr', sd, s1)] = feats_chunk(Xraw[loc])[:, :len(names)]
                    zn1 = tfn(RAWC[('tr', sd, s1)])
                    ref = zn1 if v in ('new_lr', 'new_hgb') else feat_subject(zn1 if v == 'new_prank' else np.hstack([Z[loc], zn1]), 'hgb_prank')
                    assert np.array_equal(ref, F[loc])
                    print(f'[s{sd} fold {k + 1}/5] {v:10s} dim={F.shape[1]} fit_s={tf:.1f} checks ok '
                          f'(shuffle/prefix/isolation) {el()}', flush=True)
                    del F, clf
        ev = np.isfinite(S['hgb_prank'])
        for c in ('lr', 'hgb'):
            n = f'position_only__{c}'; assert np.abs(S[n][ev] - prior[f'stream{sd}__{n}'][ev]).max() < 1e-9
        W = prior_rn[f'stream{sd}__resnet_whiten_logit_delta']
        assert np.allclose(W, prior[f'stream{sd}__resnet_whiten_logit_delta'], equal_nan=True)
        FM = prior[f'stream{sd}__fusionM_prank']
        assert np.abs(sig((logit(S['hgb_prank']) + logit(W)) / 2)[ev] - FM[ev]).max() < 1e-9
        S['fusionM_prank'] = np.where(ev, FM, np.nan)
        S['fuse_newprank'] = sig((logit(FM) + logit(S['new_prank'])) / 2)
        S['fuse_allprank_resnet'] = sig((logit(S['all_prank']) + logit(W)) / 2)
        S['fuse3'] = sig((logit(FM) + logit(S['new_hgb'])) / 2)
        M = {'all': ev, 'mature': ev & (t >= MATURE)}
        R = result[f'stream{sd}'] = {'methods': {}, 'bootstrap': {}, 'subjects': {}, 'groups': {}}
        ev_subs = list(np.unique(g[ev])); PS = {}
        for n, p in S.items():
            oof[f'stream{sd}__{n}'] = p
            r = R['methods'][n] = {s: metrics(y, g, p, M[s]) for s in M}
            PS[n] = {s: metrics(y, g, p, ev & (g == s))['subject_macro_bACC'] for s in ev_subs}
            ba = np.array([q for q in PS[n].values() if q is not None])
            R['subjects'][n] = {'n_ge_0.9': int((ba >= .9).sum()), 'max': float(ba.max()), 'n': int(len(ba))}
            print(f'[result stream {sd}] {n:22s} ' + ' | '.join(
                f'{s}: AUC={r[s]["AUC"]:.4f} mac={r[s]["subject_macro_bACC"]:.4f} mic={r[s]["micro_bACC"]:.4f}' for s in M) +
                f' | subj>=.9={R["subjects"][n]["n_ge_0.9"]} max={R["subjects"][n]["max"]:.3f}', flush=True)
        for n in S:
            R['groups'][n] = {}
            for G in ('high', 'mid', 'low'):
                ss = [s for s in ev_subs if grp_of.get(s) == G and PS[n][s] is not None]
                if not ss: continue
                a = np.array([PS[n][s] for s in ss]); b = np.array([PS['fusionM_prank'][s] for s in ss])
                R['groups'][n][G] = {'n': len(ss), 'median': float(np.median(a)), 'mean': float(a.mean()),
                                     'd_median_vs_fusionM_prank': float(np.median(a) - np.median(b)),
                                     'd_mean_vs_fusionM_prank': float(a.mean() - b.mean())}
            print(f'[group stream {sd}] {n:22s} ' + ' | '.join(
                f'{G}(n={q["n"]}): med={q["median"]:.4f} mean={q["mean"]:.4f} dmed={q["d_median_vs_fusionM_prank"]:+.4f} '
                f'dmean={q["d_mean_vs_fusionM_prank"]:+.4f}' for G, q in R['groups'][n].items()), flush=True)
        for c in ('lr', 'hgb'):
            a = R['methods'][f'position_only__{c}']['all']['AUC']
            R.setdefault('sanity', {})[f'position_only__{c}'] = {'AUC': a, 'ok': abs(a - .5) < .03}
            print(f'[sanity stream {sd}] position_only__{c} AUC={a:.4f} -> {"OK" if abs(a - .5) < .03 else "FLAG"}', flush=True)
            if not args.smoke: assert abs(a - .5) < .03
        pairs = [(a_, 'fusionM_prank') for a_ in ('new_lr', 'new_hgb', 'new_prank', 'all_prank', 'fuse_newprank',
                                                   'fuse_allprank_resnet', 'fuse3')] + [('all_prank', 'hgb_prank')]
        for a_, b_ in pairs:
            for seg in M:
                bb = R['bootstrap'][f'{a_}__vs__{b_}__{seg}'] = boot(S[a_], S[b_], y, g, M[seg], nb)
                print(f'[boot stream {sd}] {a_:20s} vs {b_:13s} {seg:6s} '
                      f'dAUC={bb["dAUC"]["mean"]:+.4f}[{bb["dAUC"]["p2.5"]:+.4f},{bb["dAUC"]["p97.5"]:+.4f}] '
                      f'dmac={bb["dmacro_bACC"]["mean"]:+.4f}[{bb["dmacro_bACC"]["p2.5"]:+.4f},{bb["dmacro_bACC"]["p97.5"]:+.4f}] '
                      f'dmic={bb["dmicro_bACC"]["mean"]:+.4f}[{bb["dmicro_bACC"]["p2.5"]:+.4f},{bb["dmicro_bACC"]["p97.5"]:+.4f}]',
                      flush=True)
        (OUT / (stem + '.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    result['seconds'] = time.perf_counter() - T0
    (OUT / (stem + '.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    np.savez_compressed(OUT / (stem + '_oof.npz'), **oof)
    print(f'[done] all checks passed {el()} -> {stem}.json', flush=True)
    sys.stdout = tee.stream; tee.f.close()


if __name__ == '__main__':
    main()
