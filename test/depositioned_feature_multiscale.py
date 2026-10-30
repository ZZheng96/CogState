"""Direction 3: deeper causal per-subject feature-side differencing on the 858 features (de-positioned streams).

Protocol identical to depositioned_resnet_histcontext.py: stream order = concat(relax,task) permuted by
default_rng([stream_seed, subject_index]); outer folds = subject->fold map stored in depositioned_resnet_histcontext_oof.npz
(SGKF(5, rs0) on seed-0 arrangement, re-derived and asserted). z = StandardScaler(train fold) of median-imputed 858 f.
History = previous stream windows of the same subject only (train subjects too); no labels used.
Variants (fixed before running; HGB(200, depth3, lr .06, l2 1, rs0), all 858 features, no feature selection):
  hgb_ms     [z, z-h8, z-h32, z-h128, z-hall]   h_k = mean of previous <=k windows (0 if no past);
                                                hall = sum(previous)/(t+32) (hist_subject shrinkage)   -> 4290 dims
  hgb_prank  [z, z-h32, prank128]               prank128 = (#past<z + .5 #past==z)/n_past over previous <=128 windows,
                                                per feature (causal within-subject percentile; 0.5 if no past) -> 2574
  hgb_zs     [z, (z-h128)/sqrt(var_past128+1e-3)] var = population variance of previous <=128 windows; 0 when fewer
                                                than 8 past windows                                      -> 1716
Fusion with the whiten ResNet (scores reused from depositioned_resnet_histcontext_oof.npz, no retraining; stream0 =
3 ResNet seeds, streams 1/2 = 1 seed): fusionM_X = sigmoid(mean(logit hgb_X, logit resnet_whiten_logit_delta)).
References: hgb_delta32 (recomputed, asserted == stored OOF), fusionH_whiten (recomputed from stored parts, asserted),
hgb_current (from depositioned_resnet_history_delta_oof.npz). Threshold .5. Paired subject bootstrap 1000.
Checks (assert): position-only AUC ~.5; vectorized vs naive stepwise reference (sampled rows); prefix invariance of
features and scores under future change; subject isolation (alone == pooled), test and train side.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from depositioned_history_delta import arrange, make_clf, hist_subject, boot, load_db, P_ALL  # noqa: E402
from depositioned_resnet_history_delta import stream_order, hist_delta, logit, sig  # noqa: E402
from causal_kernel_state_mapping import metrics  # noqa: E402
from switch_dual_memory import Tee  # noqa: E402

MATURE, KR, MIN_ZS, EPS = 120, 128, 8, 1e-3
VARIANTS = ('hgb_ms', 'hgb_prank', 'hgb_zs')
STREAM_SEEDS = (0, 1, 2)
T0 = time.perf_counter()


def el():
    return f'elapsed={time.perf_counter() - T0:.1f}s'


def win_stats(zs, k):
    """Causal mean / population var of previous <=k rows (shifted by row 0 for numerical stability). cnt per row."""
    n = len(zs); c = zs[:1]; d = zs - c
    cs = np.concatenate([np.zeros((1, zs.shape[1])), np.cumsum(d, 0)])
    cq = np.concatenate([np.zeros((1, zs.shape[1])), np.cumsum(d * d, 0)])
    t = np.arange(n); lo = np.maximum(0, t - k); cnt = (t - lo).astype(float)[:, None]
    with np.errstate(invalid='ignore', divide='ignore'):
        m1 = (cs[t] - cs[lo]) / cnt; m2 = (cq[t] - cq[lo]) / cnt
    mu = np.where(cnt > 0, m1 + c, 0.); var = np.where(cnt > 0, np.maximum(m2 - m1 * m1, 0.), 0.)
    return mu, var, cnt[:, 0]


def prank(zs, k=KR):
    n = len(zs); acc = np.zeros(zs.shape, np.int32)
    for j in range(1, min(k, n - 1) + 1):
        cur, prev = zs[j:], zs[:-j]
        acc[j:] += 2 * (prev < cur) + (prev == cur)
    npast = np.minimum(np.arange(n), k)[:, None]
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(npast > 0, acc / (2. * npast), .5)


def feat_subject(zs, v):
    """zs: one subject's z in stream order -> variant features (row t uses zs[:t+1] only)."""
    if v == 'hgb_ms':
        hall = np.concatenate([np.zeros((1, zs.shape[1])), np.cumsum(zs, 0)])[:-1] / (np.arange(len(zs)) + P_ALL)[:, None]
        return np.hstack([zs] + [zs - win_stats(zs, k)[0] for k in (8, 32, KR)] + [zs - hall])
    if v == 'hgb_prank':
        return np.hstack([zs, zs - win_stats(zs, 32)[0], prank(zs)])
    mu, var, cnt = win_stats(zs, KR)
    return np.hstack([zs, np.where(cnt[:, None] >= MIN_ZS, (zs - mu) / np.sqrt(var + EPS), 0.)])


def naive_row(zs, t, v):
    """Plain stepwise reference for row t."""
    def past(k): return zs[max(0, t - k):t] if k else zs[:t]
    def hm(k): P = past(k); return P.mean(0) if len(P) else np.zeros(zs.shape[1])
    z = zs[t]
    if v == 'hgb_ms':
        run = np.zeros(zs.shape[1])
        for j in range(t): run = run + zs[j]
        return np.concatenate([z, z - hm(8), z - hm(32), z - hm(KR), z - run / (t + P_ALL)])
    if v == 'hgb_prank':
        P = past(KR)
        pr = ((P < z).sum(0) + .5 * (P == z).sum(0)) / len(P) if len(P) else np.full(zs.shape[1], .5)
        return np.concatenate([z, z - hm(32), pr])
    P = past(KR)
    zz = (z - P.mean(0)) / np.sqrt(P.var(0) + EPS) if len(P) >= MIN_ZS else np.zeros(zs.shape[1])
    return np.concatenate([z, zz])


def feat_all(Z, order, v):
    out = None
    for loc in order:
        F = feat_subject(Z[loc], v)
        if out is None: out = np.empty((len(Z), F.shape[1]))
        out[loc] = F
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); args = ap.parse_args()
    stem = 'depositioned_feature_multiscale' + ('_smoke' if args.smoke else '')
    tee = Tee(sys.stdout, OUT / (stem + '.log')); sys.stdout = tee
    sseeds = STREAM_SEEDS[:1] if args.smoke else STREAM_SEEDS
    nb = 100 if args.smoke else 1000
    man, subs, data = load_db()
    Fc = np.vstack([data[s][0] for s in subs]); y = np.concatenate([data[s][1] for s in subs])
    g = np.concatenate([np.full(len(data[s][1]), s) for s in subs])
    prior = np.load(OUT / 'depositioned_resnet_histcontext_oof.npz')
    prior_hd = np.load(OUT / 'depositioned_resnet_history_delta_oof.npz')
    assert np.array_equal(prior['y'], y) and np.array_equal(prior['groups'], g)
    Xa, ya, ga, _ = arrange(subs, data, 0, 'independent'); fold_of = {}
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Xa, ya, ga)):
        for s in np.unique(ga[te]): fold_of[s] = k
    fold = np.array([fold_of[s] for s in g]); assert np.array_equal(fold, prior['fold'])
    run_folds = [0] if args.smoke else list(range(5))
    print(f'[start] rows={len(y)} subjects={len(subs)} smoke={args.smoke} sseeds={sseeds} variants={VARIANTS} '
          f'whiten scores reused from depositioned_resnet_histcontext_oof.npz', flush=True)
    result = {'meta': {'mature_index': MATURE, 'K_rank_zs': KR, 'min_past_zs': MIN_ZS, 'eps_zs': EPS, 'P_all': P_ALL,
                       'threshold': .5, 'rows': int(len(y)), 'subjects': len(subs), 'variants': VARIANTS,
                       'stream_seeds': list(sseeds), 'smoke': args.smoke, 'feature_selection': None,
                       'whiten_source': 'depositioned_resnet_histcontext_oof.npz (stream0 3 ResNet seeds, streams1/2 1 seed)'},
              'checks': {}}
    checks = result['checks']; oof = {'y': y, 'groups': g, 'fold': fold}
    for sd in sseeds:
        order, t = stream_order(g, subs, sd); sidx = {s: j for j, s in enumerate(subs)}
        names = ['position_only__lr', 'position_only__hgb', 'hgb_delta32'] + list(VARIANTS)
        S = {n: np.full(len(y), np.nan) for n in names}
        with threadpool_limits(limits=8):
            for k in run_folds:
                tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
                T = t[:, None].astype(float)
                for c in ('lr', 'hgb'):
                    S[f'position_only__{c}'][te] = make_clf(c).fit(T[tr], y[tr]).predict_proba(T[te])[:, 1]
                med = np.nanmedian(np.where(np.isfinite(Fc[tr]), Fc[tr], np.nan), 0); med = np.where(np.isfinite(med), med, 0.)
                Z = StandardScaler().fit(np.where(np.isfinite(Fc), Fc, med)[tr]).transform(np.where(np.isfinite(Fc), Fc, med))
                t1 = time.perf_counter(); ZD = np.hstack([Z, hist_delta(Z, order)])
                S['hgb_delta32'][te] = make_clf('hgb').fit(ZD[tr], y[tr]).predict_proba(ZD[te])[:, 1]
                dmax = float(np.abs(S['hgb_delta32'][te] - prior[f'stream{sd}__hgb_delta32'][te]).max())
                assert dmax < 1e-9, dmax; checks[f'hgb_delta32_reproduced_s{sd}_f{k}'] = dmax
                print(f'[s{sd} fold {k + 1}/5] hgb_delta32 reproduced (maxabs {dmax:.1e}) fit_s={time.perf_counter() - t1:.1f} {el()}', flush=True)
                del ZD
                te_subs = list(np.unique(g[te]))
                for v in VARIANTS:
                    t1 = time.perf_counter(); F = feat_all(Z, order, v); tb = time.perf_counter() - t1
                    # vectorized vs naive (3 test subjects + 1 train subject, sampled rows)
                    for s0 in te_subs[:3] + [np.unique(g[tr])[0]]:
                        loc = order[sidx[s0]]; zs = Z[loc]; n = len(loc)
                        for r in sorted({0, 1, 7, 8, 9, 31, 33, 100, 127, 128, 129, 200, n - 1} & set(range(n))):
                            ref = naive_row(zs, r, v)
                            assert np.allclose(F[loc[r]], ref, rtol=1e-6, atol=1e-6), (v, s0, r, np.abs(F[loc[r]] - ref).max())
                    t2 = time.perf_counter(); clf = make_clf('hgb').fit(F[tr], y[tr])
                    S[v][te] = clf.predict_proba(F[te])[:, 1]; tf = time.perf_counter() - t2
                    # prefix invariance + subject isolation (2 test subjects, 1 train subject for isolation)
                    for s0 in te_subs[:2]:
                        loc = order[sidx[s0]]; zs = Z[loc]; cut = len(loc) // 2
                        alone = feat_subject(zs, v); assert np.array_equal(alone, F[loc])
                        alt = zs.copy(); alt[cut:] += 100.; Fa = feat_subject(alt, v)
                        assert np.array_equal(Fa[:cut], F[loc][:cut]) and np.array_equal(feat_subject(zs[:cut], v), F[loc][:cut])
                        pa = clf.predict_proba(Fa)[:, 1]
                        d_pre = float(np.abs(pa[:cut] - S[v][loc[:cut]]).max()); d_fut = float(np.abs(pa[cut:] - S[v][loc[cut:]]).max())
                        assert d_pre == 0. and d_fut > 1e-6, (d_pre, d_fut)
                        checks[f'causal_s{sd}_f{k}_{v}_{s0}'] = {'score_prefix_after_future_change': d_pre, 'score_future_changed': d_fut}
                    s1 = np.unique(g[tr])[0]; loc = order[sidx[s1]]
                    assert np.array_equal(feat_subject(Z[loc], v), F[loc])
                    print(f'[s{sd} fold {k + 1}/5] {v:10s} dim={F.shape[1]} build_s={tb:.1f} fit_s={tf:.1f} '
                          f'checks ok (naive/prefix/isolation) {el()}', flush=True)
                    del F, clf
        ev = np.isfinite(S['hgb_delta32'])
        for c in ('lr', 'hgb'):
            n = f'position_only__{c}'
            assert np.abs(S[n][ev] - prior[f'stream{sd}__{n}'][ev]).max() < 1e-9
        W = prior[f'stream{sd}__resnet_whiten_logit_delta']
        fh = sig((logit(S['hgb_delta32']) + logit(W)) / 2)
        assert np.abs(fh[ev] - prior[f'stream{sd}__fusionH_whiten'][ev]).max() < 1e-9
        S['resnet_whiten_logit_delta'] = np.where(ev, W, np.nan); S['fusionH_whiten'] = fh
        S['hgb_current'] = np.where(ev, prior_hd[f'stream{sd}__hgb_current'], np.nan)
        for v in VARIANTS:
            S[v.replace('hgb_', 'fusionM_')] = sig((logit(S[v]) + logit(W)) / 2)
        M = {'all': ev, 'mature': ev & (t >= MATURE)}
        R = result[f'stream{sd}'] = {'methods': {}, 'bootstrap': {}, 'subjects': {}}
        for n, p in S.items():
            oof[f'stream{sd}__{n}'] = p
            r = R['methods'][n] = {s: metrics(y, g, p, M[s]) for s in M}
            ba = np.array([q for s in np.unique(g[ev]) if (q := metrics(y, g, p, ev & (g == s))['subject_macro_bACC']) is not None])
            R['subjects'][n] = {'n_ge_0.9': int((ba >= .9).sum()), 'max': float(ba.max()), 'n': int(len(ba))}
            print(f'[result stream {sd}] {n:26s} ' + ' | '.join(
                f'{s}: AUC={r[s]["AUC"]:.4f} mac={r[s]["subject_macro_bACC"]:.4f} mic={r[s]["micro_bACC"]:.4f}' for s in M) +
                f' | subj>=.9={R["subjects"][n]["n_ge_0.9"]} max={R["subjects"][n]["max"]:.3f}', flush=True)
        for c in ('lr', 'hgb'):
            a = R['methods'][f'position_only__{c}']['all']['AUC']
            R.setdefault('sanity', {})[f'position_only__{c}'] = {'AUC': a, 'ok': abs(a - .5) < .03}
            print(f'[sanity stream {sd}] position_only__{c} AUC={a:.4f} -> {"OK" if abs(a - .5) < .03 else "FLAG"}', flush=True)
            if not args.smoke: assert abs(a - .5) < .03
        new = list(VARIANTS) + [v.replace('hgb_', 'fusionM_') for v in VARIANTS]
        for a_ in new:
            for b_ in ('hgb_delta32', 'fusionH_whiten'):
                for seg in M:
                    bb = R['bootstrap'][f'{a_}__vs__{b_}__{seg}'] = boot(S[a_], S[b_], y, g, M[seg], nb)
                    print(f'[boot stream {sd}] {a_:16s} vs {b_:15s} {seg:6s} '
                          f'dAUC={bb["dAUC"]["mean"]:+.4f}[{bb["dAUC"]["p2.5"]:+.4f},{bb["dAUC"]["p97.5"]:+.4f}] '
                          f'dmac={bb["dmacro_bACC"]["mean"]:+.4f}[{bb["dmacro_bACC"]["p2.5"]:+.4f},{bb["dmacro_bACC"]["p97.5"]:+.4f}] '
                          f'dmic={bb["dmicro_bACC"]["mean"]:+.4f}[{bb["dmicro_bACC"]["p2.5"]:+.4f},{bb["dmicro_bACC"]["p97.5"]:+.4f}]',
                          flush=True)
        (OUT / (stem + '.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    result['seconds'] = time.perf_counter() - T0
    (OUT / (stem + '.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    np.savez_compressed(OUT / (stem + '_oof.npz'), **oof)
    print(f'[done] {el()} -> {stem}.json', flush=True)
    sys.stdout = tee.stream; tee.f.close()


if __name__ == '__main__':
    main()
