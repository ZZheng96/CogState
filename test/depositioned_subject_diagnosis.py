"""Direction A: per-subject diagnosis of de-positioned OOF scores + a fixed subject-reweighted transfer variant.

Part 1 (diagnosis, read-only on stored OOF depositioned_feature_multiscale_oof.npz; methods fusionM_prank, hgb_prank):
  per subject x stream (0,1,2): AUC, bACC@.5; subject score = mean over 3 streams.
  Groups by mean bACC(fusionM_prank): high >= .8, mid [.6, .8), low < .6.
  Unlabeled attributes: n_windows; raw RMS (mean over windows of window std of x); relative band power (Welch fs 250,
    nperseg 500, bands delta 1-4, theta 4-8, alpha 8-13, beta 13-30, gamma 30-45, relative to 1-45 Hz, mean over
    windows); feat_var = mean over 858 features of within-subject variance of globally z-scored features (global median
    impute + global StandardScaler, diagnosis only); score stats of fusionM_prank logit (stream0): std, bimodality
    coefficient (skew^2+1)/(kurt_excess+3(n-1)^2/((n-2)(n-3))), |mean prob - .5|.
  Labeled attributes (relax vs task inside one subject): Cohen's d per feature (pooled sd); d_max, d_top10 (mean of 10
    largest |d|), n_d_ge_0.5; within-subject AUC = LogisticRegression(C=.05) on per-subject StandardScaler'd z,
    StratifiedKFold(5, shuffle, rs0) (optimistic, adjacent windows share drift; used as upper bound of separability);
    direction consistency = corr(d_s, mean d of other subjects); bACC at oracle threshold = per-subject median score
    (ranking vs calibration split); sign_flip = AUC < .5.
  Spearman(attribute, subject mean AUC / mean bACC of fusionM_prank), n=57.
Part 2 (pre-fixed variant hgb_prank_clean): same outer folds / features (hgb_prank [z, z-h32, prank128]) / HGB(200,
  depth3, lr .06, l2 1, rs0). Inside each outer fold, GroupKFold(4) over training subjects (no shuffle) -> inner OOF
  score per training row -> inner AUC_s per training subject; sample weight w_s = clip((AUC_s - .5)/.2, .1, 1.).
  Outer model = HGB fit on all training rows with sample_weight. fusionM_clean = sigmoid(mean(logit hgb_prank_clean,
  logit resnet_whiten_logit_delta)) (ResNet scores from depositioned_resnet_histcontext_oof.npz). Threshold .5.
  Paired subject bootstrap 1000 vs fusionM_prank and hgb_prank (stored). Checks: hgb_prank reproduced from features
  (stored OOF); position-only AUC ~.5 (recomputed == stored); prefix invariance + subject isolation of features/scores;
  weights invariant to test labels (test labels shuffled -> identical weights).
No hyperparameter is tuned on results.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
from scipy.signal import welch
from scipy.stats import spearmanr, skew, kurtosis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from depositioned_history_delta import make_clf, boot, load_db  # noqa: E402
from depositioned_resnet_history_delta import stream_order, logit, sig, load_raw  # noqa: E402
from depositioned_feature_multiscale import feat_all, feat_subject, MATURE  # noqa: E402
from causal_kernel_state_mapping import metrics  # noqa: E402
from switch_dual_memory import Tee  # noqa: E402

STREAM_SEEDS = (0, 1, 2)
BANDS = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}
HI, LO = .8, .6
W_LO, W_SCALE, W_MIN = .5, .2, .1
T0 = time.perf_counter()


def el():
    return f'elapsed={time.perf_counter() - T0:.1f}s'


def subj_auc_bacc(y, g, p, s):
    m = (g == s) & np.isfinite(p)
    return float(roc_auc_score(y[m], p[m])), float(balanced_accuracy_score(y[m], p[m] >= .5))


def diagnosis(subs, Fc, y, g, ms, res):
    print('[diag] loading raw windows for RMS / band power', flush=True)
    X, F2, y2, g2 = load_raw(subs); assert np.array_equal(y2, y) and np.array_equal(g2, g)
    fr, P = welch(X.astype(np.float64), fs=250., nperseg=500, axis=-1)
    tot = P[:, (fr >= 1) & (fr < 45)].sum(1)
    rel = {b: P[:, (fr >= lo) & (fr < hi)].sum(1) / tot for b, (lo, hi) in BANDS.items()}
    rms = X.std(1); del X, P
    med = np.nanmedian(np.where(np.isfinite(Fc), Fc, np.nan), 0); med = np.where(np.isfinite(med), med, 0.)
    Z = StandardScaler().fit_transform(np.where(np.isfinite(Fc), Fc, med))
    D = {}
    for s in subs:
        m = g == s; a, b = Z[m & (y == 0)], Z[m & (y == 1)]
        sd = np.sqrt(((len(a) - 1) * a.var(0, ddof=1) + (len(b) - 1) * b.var(0, ddof=1)) / (len(a) + len(b) - 2))
        D[s] = np.where(sd > 1e-9, (b.mean(0) - a.mean(0)) / np.where(sd > 1e-9, sd, 1.), 0.)
    Dm = np.array([D[s] for s in subs]); rows = []
    for j, s in enumerate(subs):
        m = g == s; r = {'subject': s}
        for meth in ('fusionM_prank', 'hgb_prank'):
            v = np.array([subj_auc_bacc(y, g, ms[f'stream{sd}__{meth}'], s) for sd in STREAM_SEEDS])
            for sd in STREAM_SEEDS: r[f'{meth}_s{sd}_AUC'], r[f'{meth}_s{sd}_bACC'] = v[sd]
            r[f'{meth}_AUC'], r[f'{meth}_bACC'] = float(v[:, 0].mean()), float(v[:, 1].mean())
        p = ms['stream0__fusionM_prank'][m]; lg = logit(p); n = len(lg)
        r['n_windows'] = int(m.sum()); r['rms'] = float(rms[m].mean())
        for b in BANDS: r[f'rel_{b}'] = float(np.nanmean(rel[b][m]))  # nan for flat windows (zero 1-45 Hz power)
        r['feat_var'] = float(Z[m].var(0).mean())
        r['score_std'] = float(lg.std()); r['score_offset'] = float(abs(p.mean() - .5))
        r['score_bimodality'] = float((skew(lg) ** 2 + 1) / (kurtosis(lg) + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))
        ad = np.sort(np.abs(D[s]))[::-1]
        r['d_max'], r['d_top10'], r['n_d_ge_0.5'] = float(ad[0]), float(ad[:10].mean()), int((ad >= .5).sum())
        oth = np.delete(Dm, j, 0).mean(0); r['dir_consistency'] = float(np.corrcoef(D[s], oth)[0, 1])
        zs = StandardScaler().fit_transform(Z[m]); ys = y[m]; q = np.zeros(len(ys))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(zs, ys):
            q[te] = LogisticRegression(C=.05, max_iter=2000).fit(zs[tr], ys[tr]).decision_function(zs[te])
        r['within_AUC'] = float(roc_auc_score(ys, q))
        r['bACC_median_thr'] = float(balanced_accuracy_score(ys, p >= np.median(p)))
        r['group'] = 'high' if r['fusionM_prank_bACC'] >= HI else ('mid' if r['fusionM_prank_bACC'] >= LO else 'low')
        rows.append(r)
        print(f'[diag {j + 1:2d}/57] {s} AUC={r["fusionM_prank_AUC"]:.3f} bACC={r["fusionM_prank_bACC"]:.3f} '
              f'within={r["within_AUC"]:.3f} dtop10={r["d_top10"]:.2f} dir={r["dir_consistency"]:+.2f} {el()}', flush=True)
    attrs = ['n_windows', 'rms'] + [f'rel_{b}' for b in BANDS] + ['feat_var', 'score_std', 'score_bimodality',
             'score_offset', 'd_max', 'd_top10', 'n_d_ge_0.5', 'within_AUC', 'dir_consistency', 'bACC_median_thr']
    sp = {}
    for a in attrs:
        x = [r[a] for r in rows]
        sp[a] = {}
        for t in ('fusionM_prank_AUC', 'fusionM_prank_bACC'):
            c = spearmanr(x, [r[t] for r in rows], nan_policy='omit'); sp[a][t] = {'rho': float(c.correlation), 'p': float(c.pvalue)}
        print(f'[spearman] {a:18s} vs AUC rho={sp[a]["fusionM_prank_AUC"]["rho"]:+.3f} (p={sp[a]["fusionM_prank_AUC"]["p"]:.3g})'
              f'  vs bACC rho={sp[a]["fusionM_prank_bACC"]["rho"]:+.3f} (p={sp[a]["fusionM_prank_bACC"]["p"]:.3g})', flush=True)
    grp = {}
    for G in ('high', 'mid', 'low'):
        R = [r for r in rows if r['group'] == G]
        grp[G] = {'n': len(R), 'subjects': [r['subject'] for r in R]}
        for a in attrs + ['fusionM_prank_AUC', 'fusionM_prank_bACC', 'hgb_prank_AUC']:
            grp[G][a] = float(np.nanmedian([r[a] for r in R])) if R else None
    print('[groups] median per group: ' + ' '.join(f'{G}(n={grp[G]["n"]})' for G in grp), flush=True)
    for a in attrs + ['fusionM_prank_AUC', 'fusionM_prank_bACC', 'hgb_prank_AUC']:
        print(f'  {a:20s} ' + ' '.join(f'{G}={grp[G][a]:.4g}' if grp[G][a] is not None else f'{G}=NA' for G in grp), flush=True)
    wa = np.array([r['within_AUC'] for r in rows]); xa = np.array([r['fusionM_prank_AUC'] for r in rows])
    dc = np.array([r['dir_consistency'] for r in rows])
    summ = {'n_within_AUC_lt_0.65': int((wa < .65).sum()), 'n_within_ge_0.85_but_cross_lt_0.65': int(((wa >= .85) & (xa < .65)).sum()),
            'n_cross_AUC_lt_0.5': int((xa < .5).sum()), 'median_within_AUC': float(np.median(wa)),
            'median_cross_AUC': float(np.median(xa)), 'n_dir_consistency_lt_0': int((dc < 0).sum()),
            'n_bACC_ge_0.9': int((np.array([r['fusionM_prank_bACC'] for r in rows]) >= .9).sum())}
    low = [r for r in rows if r['group'] == 'low']
    summ['low_group'] = {'n': len(low), 'n_within_lt_0.65': sum(r['within_AUC'] < .65 for r in low),
                         'n_within_ge_0.85': sum(r['within_AUC'] >= .85 for r in low),
                         'n_dir_consistency_lt_0.2': sum(r['dir_consistency'] < .2 for r in low)}
    print(f'[diag summary] {json.dumps(summ)}', flush=True)
    res['diagnosis'] = {'subjects': rows, 'spearman': sp, 'groups': grp, 'summary': summ}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); args = ap.parse_args()
    stem = 'depositioned_subject_diagnosis' + ('_smoke' if args.smoke else '')
    tee = Tee(sys.stdout, OUT / (stem + '.log')); sys.stdout = tee
    sseeds = STREAM_SEEDS[:1] if args.smoke else STREAM_SEEDS
    nb = 100 if args.smoke else 1000
    run_folds = [0] if args.smoke else list(range(5))
    man, subs, data = load_db()
    Fc = np.vstack([data[s][0] for s in subs]); y = np.concatenate([data[s][1] for s in subs])
    g = np.concatenate([np.full(len(data[s][1]), s) for s in subs])
    ms = np.load(OUT / 'depositioned_feature_multiscale_oof.npz'); prior = np.load(OUT / 'depositioned_resnet_histcontext_oof.npz')
    assert np.array_equal(ms['y'], y) and np.array_equal(ms['groups'], g) and np.array_equal(prior['fold'], ms['fold'])
    fold = ms['fold']
    print(f'[start] rows={len(y)} subjects={len(subs)} smoke={args.smoke} sseeds={sseeds} folds={run_folds}', flush=True)
    res = {'meta': {'groups_thr': [LO, HI], 'weight': f'clip((AUC_s-{W_LO})/{W_SCALE},{W_MIN},1)', 'inner_cv': 'GroupKFold(4)',
                    'threshold': .5, 'stream_seeds': list(sseeds), 'smoke': args.smoke, 'n_boot': nb}, 'checks': {}}
    with threadpool_limits(limits=6):
        diagnosis(subs, Fc, y, g, ms, res)
        oof = {'y': y, 'groups': g, 'fold': fold}; checks = res['checks']; sidx = {s: j for j, s in enumerate(subs)}
        for sd in sseeds:
            order, t = stream_order(g, subs, sd)
            S = {n: np.full(len(y), np.nan) for n in ('position_only__lr', 'position_only__hgb', 'hgb_prank_rep', 'hgb_prank_clean')}
            inner_auc = {}
            for k in run_folds:
                tf0 = time.perf_counter()
                tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k); T = t[:, None].astype(float)
                for c in ('lr', 'hgb'):
                    S[f'position_only__{c}'][te] = make_clf(c).fit(T[tr], y[tr]).predict_proba(T[te])[:, 1]
                med = np.nanmedian(np.where(np.isfinite(Fc[tr]), Fc[tr], np.nan), 0); med = np.where(np.isfinite(med), med, 0.)
                Z = StandardScaler().fit(np.where(np.isfinite(Fc), Fc, med)[tr]).transform(np.where(np.isfinite(Fc), Fc, med))
                F = feat_all(Z, order, 'hgb_prank')
                S['hgb_prank_rep'][te] = make_clf('hgb').fit(F[tr], y[tr]).predict_proba(F[te])[:, 1]
                dmax = float(np.abs(S['hgb_prank_rep'][te] - ms[f'stream{sd}__hgb_prank'][te]).max())
                assert dmax < 1e-9, dmax; checks[f'hgb_prank_reproduced_s{sd}_f{k}'] = dmax
                print(f'[s{sd} fold {k + 1}/5] hgb_prank reproduced maxabs={dmax:.1e} {el()}', flush=True)

                def weights(yv):
                    q = np.full(len(tr), np.nan)
                    for ii, (a, b) in enumerate(GroupKFold(4).split(tr, yv[tr], g[tr])):
                        q[b] = make_clf('hgb').fit(F[tr[a]], yv[tr[a]]).predict_proba(F[tr[b]])[:, 1]
                        print(f'[s{sd} fold {k + 1}/5] inner {ii + 1}/4 done {el()}', flush=True)
                    au = {s: float(roc_auc_score(yv[tr][g[tr] == s], q[g[tr] == s])) for s in np.unique(g[tr])}
                    w = np.array([np.clip((au[s] - W_LO) / W_SCALE, W_MIN, 1.) for s in g[tr]])
                    return au, w
                au, w = weights(y)
                if k == run_folds[0]:  # test labels must not affect weights: shuffle test labels, recompute
                    yt = y.copy(); yt[te] = np.random.default_rng(1).permutation(y[te])
                    au2, w2 = weights(yt); assert np.array_equal(w, w2); checks[f'weights_test_label_invariant_s{sd}'] = True
                for s_, v_ in au.items(): inner_auc.setdefault(s_, []).append(v_)
                clf = make_clf('hgb').fit(F[tr], y[tr], sample_weight=w)
                S['hgb_prank_clean'][te] = clf.predict_proba(F[te])[:, 1]
                te_subs = list(np.unique(g[te]))
                for s0 in te_subs[:2]:
                    loc = order[sidx[s0]]; zs = Z[loc]; cut = len(loc) // 2
                    assert np.array_equal(feat_subject(zs, 'hgb_prank'), F[loc])
                    alt = zs.copy(); alt[cut:] += 100.; Fa = feat_subject(alt, 'hgb_prank')
                    assert np.array_equal(Fa[:cut], F[loc][:cut])
                    pa = clf.predict_proba(Fa)[:, 1]
                    d_pre = float(np.abs(pa[:cut] - S['hgb_prank_clean'][loc[:cut]]).max())
                    d_fut = float(np.abs(pa[cut:] - S['hgb_prank_clean'][loc[cut:]]).max())
                    assert d_pre == 0. and d_fut > 1e-6, (d_pre, d_fut)
                    checks[f'causal_s{sd}_f{k}_{s0}'] = {'score_prefix_after_future_change': d_pre, 'score_future_changed': d_fut}
                s1 = np.unique(g[tr])[0]; assert np.array_equal(feat_subject(Z[order[sidx[s1]]], 'hgb_prank'), F[order[sidx[s1]]])
                ws = np.array([np.clip((au[s] - W_LO) / W_SCALE, W_MIN, 1.) for s in au])
                print(f'[s{sd} fold {k + 1}/5] clean fit done; weights mean={ws.mean():.3f} n_min={(ws <= W_MIN).sum()} '
                      f'n_full={(ws >= 1).sum()} fold_s={time.perf_counter() - tf0:.1f} checks ok {el()}', flush=True)
                del F, Z, clf
            ev = np.isfinite(S['hgb_prank_clean'])
            for c in ('lr', 'hgb'):
                n = f'position_only__{c}'; assert np.abs(S[n][ev] - ms[f'stream{sd}__{n}'][ev]).max() < 1e-9
            W = prior[f'stream{sd}__resnet_whiten_logit_delta']
            S['fusionM_clean'] = sig((logit(S['hgb_prank_clean']) + logit(W)) / 2)
            S['hgb_prank'] = np.where(ev, ms[f'stream{sd}__hgb_prank'], np.nan)
            S['fusionM_prank'] = np.where(ev, ms[f'stream{sd}__fusionM_prank'], np.nan)
            assert np.abs(S['fusionM_prank'][ev] - sig((logit(S['hgb_prank_rep']) + logit(W)) / 2)[ev]).max() < 1e-9
            del S['hgb_prank_rep']
            M = {'all': ev, 'mature': ev & (t >= MATURE)}
            R = res[f'stream{sd}'] = {'methods': {}, 'bootstrap': {}, 'subjects': {}, 'sanity': {},
                                      'inner_auc': inner_auc}
            for n, p in S.items():
                oof[f'stream{sd}__{n}'] = p
                r = R['methods'][n] = {s: metrics(y, g, p, M[s]) for s in M}
                ba = np.array([metrics(y, g, p, ev & (g == s))['subject_macro_bACC'] for s in np.unique(g[ev])])
                R['subjects'][n] = {'n_ge_0.9': int((ba >= .9).sum()), 'n_ge_0.8': int((ba >= .8).sum()), 'max': float(ba.max())}
                print(f'[result stream {sd}] {n:20s} ' + ' | '.join(
                    f'{s}: AUC={r[s]["AUC"]:.4f} mac={r[s]["subject_macro_bACC"]:.4f} mic={r[s]["micro_bACC"]:.4f}' for s in M) +
                    f' | subj>=.9={R["subjects"][n]["n_ge_0.9"]} >=.8={R["subjects"][n]["n_ge_0.8"]} max={R["subjects"][n]["max"]:.3f}', flush=True)
            for c in ('lr', 'hgb'):
                a = R['methods'][f'position_only__{c}']['all']['AUC']; ok = abs(a - .5) < .03
                R['sanity'][f'position_only__{c}'] = {'AUC': a, 'ok': ok}
                print(f'[sanity stream {sd}] position_only__{c} AUC={a:.4f} -> {"OK" if ok else "FLAG"}', flush=True)
                if not args.smoke: assert ok
            if not args.smoke:  # inner (train-label) AUC vs outer test AUC of the same subject
                ss = sorted(inner_auc); oa = [subj_auc_bacc(y, g, ms[f'stream{sd}__hgb_prank'], s)[0] for s in ss]
                c = spearmanr([np.mean(inner_auc[s]) for s in ss], oa)
                R['spearman_innerAUC_vs_outer_hgb_prank_AUC'] = float(c.correlation)
                print(f'[inner] stream {sd} spearman(mean inner AUC over 4 outer folds, outer hgb_prank AUC) '
                      f'rho={c.correlation:+.3f} p={c.pvalue:.3g}', flush=True)
            for a_ in ('hgb_prank_clean', 'fusionM_clean'):
                for b_ in ('hgb_prank', 'fusionM_prank'):
                    for seg in M:
                        bb = R['bootstrap'][f'{a_}__vs__{b_}__{seg}'] = boot(S[a_], S[b_], y, g, M[seg], nb)
                        print(f'[boot stream {sd}] {a_:16s} vs {b_:14s} {seg:6s} '
                              f'dAUC={bb["dAUC"]["mean"]:+.4f}[{bb["dAUC"]["p2.5"]:+.4f},{bb["dAUC"]["p97.5"]:+.4f}] '
                              f'dmac={bb["dmacro_bACC"]["mean"]:+.4f}[{bb["dmacro_bACC"]["p2.5"]:+.4f},{bb["dmacro_bACC"]["p97.5"]:+.4f}]', flush=True)
            (OUT / (stem + '.json')).write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding='utf-8')
    res['seconds'] = time.perf_counter() - T0
    (OUT / (stem + '.json')).write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding='utf-8')
    np.savez_compressed(OUT / (stem + '_oof.npz'), **oof)
    print(f'[done] {el()} -> {stem}.json', flush=True)
    sys.stdout = tee.stream; tee.f.close()


if __name__ == '__main__':
    main()
