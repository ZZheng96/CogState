"""Per-window preprocessing study for the 858-d highdim feature block (DC offset / zero-state filter transient).

Rules: every feature uses only the current 6 s window (1500 samples @ 250 Hz); zero-phase filtering is applied INSIDE
the window only; nothing crosses windows. History (prank) features use previous windows of the same subject only, exactly
as in depositioned_aperiodic_emg.py. Hyper-parameters below were fixed before running.

Preprocessing variants (applied to the raw window x before the unchanged 858 feature definitions):
  P0  current: preprocessing.causal_filter (Butterworth order-4 1-45 Hz SOS + iirnotch 50 Hz Q30, sosfilt zero state).
  P1  x - mean(x), then the same causal_filter (zero state).
  P2  scipy.signal.detrend(x, type='linear'), then sosfiltfilt(same SOS) (default padtype='odd', padlen=3*(2*len(sos)+1)
      -- reflection of the window's own samples, no outside data).
  P3  P2, then clip to median +- 6 * 1.4826 * MAD of the filtered window.
Injection (monkeypatch, per worker call): highdim_deep_ablation.causal_filter and nonstationary_ablation.causal_filter
  -> variant function. Secondary filters: highdim_deep_ablation.sosfilt (Hilbert band filters, butter(3) per band) and
  cogstate.sosfilt (inside causal_features) -> sosfiltfilt for P2/P3 (zero-phase, no zero-state transient), unchanged
  (sosfilt) for P0/P1. causal_features outputs (raw44/quality53) are computed but not part of the 858 vector. Not changed
  by any variant (definition uses raw x): 9 quality_metrics features (cols 79-87) and 11 'raw_' nonlinear features.
  P0r = P0 recomputed from stored float32 x (control for float32 rounding); P0 = stored database f.
  P2+P0 = concatenation [P2 858 | stored P0 858].
Extraction: joblib Parallel(n_jobs=8), chunks of 128 windows, features cast to float32 (as build_database.py);
  cache test/preproc_study_feat_{P}.npz (smoke: preproc_study_smoke_feat_{P}.npz).
Diagnosis (label use = analysis only): per subject median over windows of |mean(x)|/std(x); transient ratio
  mean(y[:250]^2)/mean(y[250:]^2) of the filtered window for P0/P1/P2/P3; 50 Hz line dB = 10 log10(mean P(49.5-50.5) /
  median P(46-49 U 51-54)) on the per-subject median Welch spectrum (nperseg 500, noverlap 250, hann) of demeaned x and
  of P2 output. Label info: per-subject AUC of window mean(x) and of log mean(y_P0[:250]^2) -> mean AUC, mean |AUC-.5|;
  pooled cross-subject AUC. Reproduction: allclose(P0r, f, rtol=1e-4, atol=1e-6); float32 sensitivity of mismatching
  columns = rel. change after adding +-0.5 ulp(float32) uniform noise to x on 64 windows.
Evaluation (identical to depositioned_aperiodic_emg.py all_prank): folds = stored 'fold' of
  depositioned_aperiodic_emg_oof.npz; per fold train-median imputation + StandardScaler(train) for the 858 block and the
  stored 31 aperiodic/EMG features ('new_features'); feat_subject([z858, z31], 'hgb_prank') in stream order
  stream_order(g, subs, seed), seeds (0, 1, 2); make_clf('hgb'); threadpool_limits(8). P0 asserted == stored
  stream{sd}__all_prank (max|d| < 1e-9). fuseH_wf_P = sig(mean logit(all_prank_P, resnet_whiten_logit_delta,
  resnet_fbank_logit_delta)) from depositioned_resnet_timefreq_oof.npz. Metrics on rows with finite scores (all): AUC,
  subject-macro bACC, micro bACC at .5; paired subject bootstrap (boot from depositioned_history_delta, 1000; smoke 100)
  vs P0 of the same model; groups high/mid/low (depositioned_subject_diagnosis.json) macro bACC mean (reporting only);
  #subjects bACC>=.9 and max. Summary = mean over seeds (bootstrap: mean over seeds of mean / p2.5 / p97.5).
Smoke: diagnosis + extraction on the first 6 subjects, variant pipeline on those 6 subjects (stored folds, first fold
  present, seed 0), plus full-data P0 all_prank reproduction for seed 0 fold 0.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.signal import sosfilt, sosfiltfilt, detrend, welch
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT)); sys.path.insert(0, str(OUT.parent / 'src'))
from depositioned_history_delta import make_clf, boot, load_db  # noqa: E402
from depositioned_resnet_history_delta import load_raw, stream_order, logit, sig  # noqa: E402
from depositioned_aperiodic_emg import impute_scale, feat_all_prank  # noqa: E402
from causal_kernel_state_mapping import metrics  # noqa: E402
from switch_dual_memory import Tee  # noqa: E402
from preprocessing import sos_filters, PreprocessingConfig, causal_filter as CF0, quality_metrics  # noqa: E402

FS, N_JOBS, CHUNK, CLIP_K, MAD_C = 250., 8, 128, 6., 1.4826
STREAM_SEEDS, N_SMOKE_SUBS, TOL = (0, 1, 2), 6, dict(rtol=1e-4, atol=1e-6)
EXTRACT = ('P0r', 'P1', 'P2', 'P3')
EVAL = ('P0', 'P0r', 'P1', 'P2', 'P3', 'P2+P0')
SOS = sos_filters(PreprocessingConfig(sfreq=FS))
T0 = time.perf_counter()


def el():
    return f'elapsed={time.perf_counter() - T0:.1f}s'


def pre(var, x, axis=-1):
    """Vectorised preprocessing (diagnosis); x (..., 1500)."""
    if var == 'P0': return sosfilt(SOS, x, axis=axis)
    if var == 'P1': return sosfilt(SOS, x - x.mean(axis, keepdims=True), axis=axis)
    y = sosfiltfilt(SOS, detrend(x, axis=axis, type='linear'), axis=axis)
    if var == 'P3':
        m = np.median(y, axis, keepdims=True); s = MAD_C * np.median(np.abs(y - m), axis, keepdims=True)
        y = np.clip(y, m - CLIP_K * s, m + CLIP_K * s)
    return y


def extract_chunk(X, var):
    sys.path.insert(0, str(OUT)); sys.path.insert(0, str(OUT.parent / 'src'))
    import highdim_deep_ablation as H, nonstationary_ablation as N, cogstate as C, preprocessing as PP
    import scipy.signal as ss
    if var == 'P0r':
        cf = PP.causal_filter
    elif var == 'P1':
        def cf(x, config=PP.PreprocessingConfig(), zi=None):
            x = np.asarray(x, float); return PP.causal_filter(x - x.mean(), config)
    else:
        def cf(x, config=PP.PreprocessingConfig(), zi=None):
            y = ss.sosfiltfilt(PP.sos_filters(config), ss.detrend(np.asarray(x, float), type='linear'))
            if var == 'P3':
                m = np.median(y); s = MAD_C * np.median(np.abs(y - m)); y = np.clip(y, m - CLIP_K * s, m + CLIP_K * s)
            return y
    H.causal_filter = N.causal_filter = cf
    H.sosfilt = C.sosfilt = ss.sosfiltfilt if var in ('P2', 'P3') else ss.sosfilt
    return np.array([H.highdim_features(x.astype(float), FS) for x in X], dtype=np.float32)


def extract(X, var, path):
    if path.exists():
        F = np.load(path)['F']
        if F.shape == (len(X), 858):
            print(f'[feat {var}] cache hit {path.name} {el()}', flush=True); return F
    idx = list(range(0, len(X), CHUNK)); res = [None] * len(idx); t1 = time.perf_counter()
    gen = Parallel(n_jobs=N_JOBS, return_as='generator')(delayed(extract_chunk)(X[a:a + CHUNK], var) for a in idx)
    for i, r in enumerate(gen):
        res[i] = r
        if i % 10 == 0 or i == len(idx) - 1:
            print(f'[feat {var} {i + 1}/{len(idx)}] var_s={time.perf_counter() - t1:.1f} {el()}', flush=True)
    F = np.vstack(res); np.savez_compressed(path, F=F)
    return F


def auc_stats(v, y, g, subs):
    a = np.array([roc_auc_score(y[g == s], v[g == s]) for s in subs])
    return {'within_mean_AUC': float(a.mean()), 'within_mean_absdev': float(np.abs(a - .5).mean()),
            'within_median_absdev': float(np.median(np.abs(a - .5))), 'cross_subject_AUC': float(roc_auc_score(y, v))}


def line_db(x):
    f, P = welch(x - x.mean(-1, keepdims=True), fs=FS, window='hann', nperseg=500, noverlap=250, axis=-1)
    P = np.median(P, 0)
    nb = np.median(P[((f >= 46) & (f < 49)) | ((f > 51) & (f <= 54))])
    return float(10 * np.log10(P[(f >= 49.5) & (f <= 50.5)].mean() / nb))


def diagnosis(X, Fc, F0r, y, g, subs):
    d = {'subjects': {}}; X = X.astype(np.float64)
    mu, sd = X.mean(1), X.std(1)
    Y = {v: pre(v, X) for v in ('P0', 'P1', 'P2', 'P3')}
    TR = {v: (Y[v][:, :250] ** 2).mean(1) / (Y[v][:, 250:] ** 2).mean(1) for v in Y}
    for s in subs:
        m = g == s
        d['subjects'][s] = {'dc_over_std': float(np.median(np.abs(mu[m]) / sd[m])),
                            **{f'transient_{v}': float(np.median(TR[v][m])) for v in TR},
                            'line50_dB_raw': line_db(X[m]), 'line50_dB_P2': line_db(Y['P2'][m])}
    for k in d['subjects'][subs[0]]:
        a = np.array([d['subjects'][s][k] for s in subs])
        d[f'across_{k}'] = {'median': float(np.median(a)), 'min': float(a.min()), 'max': float(a.max())}
        print(f'[diag] {k:16s} per-subject median: median={np.median(a):.3g} min={a.min():.3g} max={a.max():.3g}', flush=True)
    for s in subs:
        q = d['subjects'][s]
        print(f'[diag {s}] |DC|/std={q["dc_over_std"]:.1f} transient P0={q["transient_P0"]:.0f} P1={q["transient_P1"]:.1f} '
              f'P2={q["transient_P2"]:.2f} P3={q["transient_P3"]:.2f} line50 raw={q["line50_dB_raw"]:+.1f}dB '
              f'P2={q["line50_dB_P2"]:+.1f}dB', flush=True)
    e1 = np.log((Y['P0'][:, :250] ** 2).mean(1))
    d['label_info'] = {'window_mean': auc_stats(mu, y, g, subs), 'first1s_logenergy_P0': auc_stats(e1, y, g, subs)}
    for k, v in d['label_info'].items():
        print(f'[diag label] {k:22s} ' + ' '.join(f'{a}={b:.4f}' for a, b in v.items()), flush=True)
    # reproduction of stored f
    close = np.isclose(F0r.astype(float), Fc, **TOL); bad_cols = np.flatnonzero(~close.all(0))
    rel = np.abs(F0r - Fc) / (np.abs(Fc) + 1e-6)
    rep = {'n_windows': int(len(Fc)), 'allclose_all': bool(close.all()), 'frac_elements_close': float(close.mean()),
           'mismatch_cols': bad_cols.tolist(), 'frac_windows_mismatch_per_col': {int(c): float((~close[:, c]).mean()) for c in bad_cols},
           'max_rel_err_per_col': {int(c): float(rel[:, c].max()) for c in bad_cols},
           'allclose_excluding_mismatch_cols': bool(np.delete(close, bad_cols, 1).all())}
    rng = np.random.default_rng(0); pick = rng.choice(len(X), 64, replace=False); sens = []
    QN = ('low_freq_drift', 'high_freq_ratio')
    for i in pick:
        x32 = X[i].astype(np.float32); ulp = np.spacing(np.abs(x32)).astype(float)
        q0 = quality_metrics(x32.astype(float), FS); q1 = quality_metrics(x32 + rng.uniform(-.5, .5, 1500) * ulp, FS)
        sens.append([abs(q1[n] - q0[n]) / (abs(q0[n]) + 1e-12) for n in QN])
    rep['float32_halfulp_sensitivity_max_rel'] = dict(zip(QN, map(float, np.max(sens, 0))))
    rep['float32_halfulp_sensitivity_median_rel'] = dict(zip(QN, map(float, np.median(sens, 0))))
    d['reproduction'] = rep
    print(f'[verify] P0r vs stored f on {len(Fc)} windows ({len(subs)} subjects): allclose={rep["allclose_all"]} '
          f'frac_close={rep["frac_elements_close"]:.6f} mismatch_cols={rep["mismatch_cols"]} '
          f'max_rel={rep["max_rel_err_per_col"]} win_frac={rep["frac_windows_mismatch_per_col"]} '
          f'allclose_excl={rep["allclose_excluding_mismatch_cols"]} | float32 +-0.5ulp sensitivity max_rel='
          f'{rep["float32_halfulp_sensitivity_max_rel"]} median={rep["float32_halfulp_sensitivity_median_rel"]}', flush=True)
    return d


def run_variant(F858, Fae, fold, y, g, subs, sd, folds, tag):
    order, _ = stream_order(g, subs, sd); p = np.full(len(y), np.nan)
    with threadpool_limits(limits=8):
        for k in folds:
            tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k); t1 = time.perf_counter()
            Z = np.hstack([impute_scale(F858, tr)(F858), impute_scale(Fae, tr)(Fae)])
            Fp = feat_all_prank(Z, order)
            p[te] = make_clf('hgb').fit(Fp[tr], y[tr]).predict_proba(Fp[te])[:, 1]; del Fp
            print(f'[eval {tag} s{sd} fold {k + 1}/5] dim={3 * Z.shape[1]} fit_s={time.perf_counter() - t1:.1f} {el()}', flush=True)
    return p


def evaluate(S, y, g, ev, nb, grp_of):
    out = {}; subs = list(np.unique(g[ev]))
    for n, p in S.items():
        r = metrics(y, g, p, ev); ps = {s: metrics(y, g, p, ev & (g == s))['subject_macro_bACC'] for s in subs}
        ba = np.array([v for v in ps.values() if v is not None])
        r.update({'n_ge_0.9': int((ba >= .9).sum()), 'max_subject_bACC': float(ba.max()),
                  'groups': {G: float(np.mean([ps[s] for s in subs if grp_of.get(s) == G and ps[s] is not None]))
                             for G in ('high', 'mid', 'low') if any(grp_of.get(s) == G for s in subs)}})
        model, var = n.split('|'); ref = S[f'{model}|P0']
        if var != 'P0': r['boot_vs_P0'] = boot(p, ref, y, g, ev, nb)
        out[n] = r
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); args = ap.parse_args()
    stem = 'preproc_study' + ('_smoke' if args.smoke else '')
    tee = Tee(sys.stdout, OUT / (stem + '.log')); sys.stdout = tee
    man, subs_all, data = load_db()
    Xall, Fcall, yall, gall = load_raw(subs_all)
    A = np.load(OUT / 'depositioned_aperiodic_emg_oof.npz'); B = np.load(OUT / 'depositioned_resnet_timefreq_oof.npz')
    for k in ('y', 'groups', 'fold'): assert np.array_equal(A[k], B[k]), k
    assert np.array_equal(A['y'], yall) and np.array_equal(A['groups'], gall)
    fold_all, Fae_all = A['fold'], A['new_features']
    diag = json.loads((OUT / 'depositioned_subject_diagnosis.json').read_text(encoding='utf-8'))['diagnosis']['groups']
    grp_of = {s: G for G in ('high', 'mid', 'low') for s in diag[G]['subjects']}
    subs = subs_all[:N_SMOKE_SUBS] if args.smoke else subs_all
    m = np.isin(gall, subs); X, Fc, y, g, fold, Fae = Xall[m], Fcall[m], yall[m], gall[m], fold_all[m], Fae_all[m]
    sseeds = STREAM_SEEDS[:1] if args.smoke else STREAM_SEEDS; nb = 100 if args.smoke else 1000
    run_folds = [int(np.unique(fold)[0])] if args.smoke else list(range(5))
    print(f'[start] smoke={args.smoke} rows={len(y)} subjects={len(subs)} seeds={sseeds} folds={run_folds} {el()}', flush=True)
    result = {'meta': {'variants_extract': EXTRACT, 'variants_eval': EVAL, 'seeds': list(sseeds), 'folds': run_folds,
                       'bootstrap_n': nb, 'clip_k': CLIP_K, 'tol': TOL, 'rows': int(len(y)), 'subjects': len(subs)}}

    if args.smoke:  # full-data reproduction of stored all_prank, seed 0 fold 0
        p = run_variant(Fcall, Fae_all, fold_all, yall, gall, subs_all, 0, [0], 'P0-full')
        te = fold_all == 0; dmax = float(np.abs(p[te] - A['stream0__all_prank'][te]).max())
        assert dmax < 1e-9, dmax; result['meta']['repro_full_s0_f0'] = dmax
        print(f'[check] full-data P0 all_prank s0 fold1 reproduced max|d|={dmax:.1e} {el()}', flush=True)

    F = {'P0': Fc}
    for v in EXTRACT:
        F[v] = extract(X, v, OUT / f'{stem}_feat_{v}.npz').astype(float)
    result['diagnosis'] = diagnosis(X, Fc, F['P0r'], y, g, subs)
    F['P2+P0'] = np.hstack([F['P2'], Fc])
    (OUT / (stem + '_results.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')

    oof = {'y': y, 'groups': g, 'fold': fold}
    for sd in sseeds:
        S = {}
        W = B[f'stream{sd}__resnet_whiten_logit_delta'][m]; Fb = B[f'stream{sd}__resnet_fbank_logit_delta'][m]
        for v in EVAL:
            p = run_variant(F[v], Fae, fold, y, g, subs, sd, run_folds, v)
            if v == 'P0' and not args.smoke:
                ev0 = np.isfinite(A[f'stream{sd}__all_prank'])
                dmax = float(np.abs(p[ev0] - A[f'stream{sd}__all_prank'][ev0]).max()); assert dmax < 1e-9, dmax
                result['meta'][f'repro_all_prank_s{sd}'] = dmax
                print(f'[check] P0 all_prank == stored stream{sd}__all_prank max|d|={dmax:.1e}', flush=True)
            S[f'all_prank|{v}'] = p
            S[f'fuseH_wf|{v}'] = sig((logit(p) + logit(W) + logit(Fb)) / 3)
        ev = np.isfinite(S['all_prank|P0']) & np.isfinite(Fb)
        S = {k: np.where(ev, v, np.nan) for k, v in S.items()}
        R = result[f'stream{sd}'] = evaluate(S, y, g, ev, nb, grp_of)
        for n, r in R.items():
            oof[f'stream{sd}__{n}'] = S[n]
            b = r.get('boot_vs_P0')
            print(f'[result s{sd}] {n:18s} AUC={r["AUC"]:.4f} mac={r["subject_macro_bACC"]:.4f} mic={r["micro_bACC"]:.4f} '
                  f'n>=.9={r["n_ge_0.9"]} max={r["max_subject_bACC"]:.3f} groups=' +
                  ' '.join(f'{G}:{q:.4f}' for G, q in r['groups'].items()) +
                  (f' | vs P0 dAUC={b["dAUC"]["mean"]:+.4f}[{b["dAUC"]["p2.5"]:+.4f},{b["dAUC"]["p97.5"]:+.4f}] '
                   f'dmac={b["dmacro_bACC"]["mean"]:+.4f}[{b["dmacro_bACC"]["p2.5"]:+.4f},{b["dmacro_bACC"]["p97.5"]:+.4f}] '
                   f'dmic={b["dmicro_bACC"]["mean"]:+.4f}[{b["dmicro_bACC"]["p2.5"]:+.4f},{b["dmicro_bACC"]["p97.5"]:+.4f}]' if b else ''),
                  flush=True)
        (OUT / (stem + '_results.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')

    summ = result['summary'] = {}
    print('[summary] mean over seeds', flush=True)
    for n in result[f'stream{sseeds[0]}']:
        rs = [result[f'stream{sd}'][n] for sd in sseeds]
        q = {k: float(np.mean([r[k] for r in rs])) for k in ('AUC', 'subject_macro_bACC', 'micro_bACC', 'n_ge_0.9', 'max_subject_bACC')}
        q['groups'] = {G: float(np.mean([r['groups'][G] for r in rs])) for G in rs[0]['groups']}
        if 'boot_vs_P0' in rs[0]:
            q['boot_vs_P0'] = {k: {s: float(np.mean([r['boot_vs_P0'][k][s] for r in rs])) for s in ('mean', 'p2.5', 'p97.5')}
                               for k in ('dAUC', 'dmacro_bACC', 'dmicro_bACC')}
        summ[n] = q; b = q.get('boot_vs_P0')
        print(f'[summary] {n:18s} AUC={q["AUC"]:.4f} mac={q["subject_macro_bACC"]:.4f} mic={q["micro_bACC"]:.4f} '
              f'n>=.9={q["n_ge_0.9"]:.2f} max={q["max_subject_bACC"]:.3f} ' + ' '.join(f'{G}:{v:.4f}' for G, v in q['groups'].items()) +
              (f' | dAUC={b["dAUC"]["mean"]:+.4f}[{b["dAUC"]["p2.5"]:+.4f},{b["dAUC"]["p97.5"]:+.4f}] '
               f'dmac={b["dmacro_bACC"]["mean"]:+.4f}[{b["dmacro_bACC"]["p2.5"]:+.4f},{b["dmacro_bACC"]["p97.5"]:+.4f}] '
               f'dmic={b["dmicro_bACC"]["mean"]:+.4f}[{b["dmicro_bACC"]["p2.5"]:+.4f},{b["dmicro_bACC"]["p97.5"]:+.4f}]' if b else ''),
              flush=True)
    result['seconds'] = time.perf_counter() - T0
    (OUT / (stem + '_results.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    np.savez_compressed(OUT / (stem + '_oof.npz'), **oof)
    print(f'[done] {el()} -> {stem}_results.json', flush=True)
    sys.stdout = tee.stream; tee.f.close()


if __name__ == '__main__':
    main()
