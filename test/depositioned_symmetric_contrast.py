"""Symmetric (direction-agnostic) within-subject contrast features (de-positioned streams).

Motivation: low-group subjects show feature directions that flip across subjects; test whether the MAGNITUDE of the
deviation from recent own history (sign removed) carries a subject-invariant relax/task signal.

Protocol identical to depositioned_aperiodic_emg.py: stream order = stream_order(g, subs, seed) (concat(relax,task)
permuted by default_rng([stream_seed, subject_index])); outer folds = SGKF(5, shuffle, rs0) on seed-0 'independent'
arrangement, re-derived and asserted == fold in depositioned_aperiodic_emg_oof.npz. Per fold: train-median imputation +
StandardScaler(train) for the 858 features (Z) and for the 31 aperiodic/EMG features (Zn, raw values read from
depositioned_aperiodic_emg_oof.npz['new_features'], not re-extracted). History = previous windows of the same subject in
stream order (row t uses rows <= t only; train subjects too); no labels of test subjects are used.

Fixed hyper-parameters (set before running):
  zz = hstack([Z, Zn]) (889 dims), per subject in stream order:
  mu_k = win_stats(zz_s, k)[0] (mean of previous <=k rows, 0 if none), k in {32, 128}; A_k = |zz - mu_k|.
  P = prank(zz_s, 128) (causal within-subject percentile, .5 if no past); Q = |P - .5|.
  sym_all  = [feat_subject(zz_s,'hgb_prank') = (zz, zz-mu_32, P), A_32, A_128]   -> 4445 dims
  sym_only = [zz, A_32, A_128, Q]                                                 -> 3556 dims
  all_prank = feat_subject(zz_s,'hgb_prank') (recomputed on stream0, asserted == npz stream0__all_prank max|d|<1e-9;
              streams 1/2 read from npz).
  Classifier: make_clf('hgb') = HGB(200, depth3, lr .06, l2 1, rs0), no feature selection, threadpool_limits(8).
  Fusion = sigmoid(mean logit), threshold .5. whiten / fbank = stream{s}__resnet_whiten_logit_delta /
  stream{s}__resnet_fbank_logit_delta from depositioned_resnet_timefreq_oof.npz.
    fuseS_all  = (sym_all, whiten, fbank); fuseS_only = (sym_only, whiten, fbank); fuseH_wf = (all_prank, whiten, fbank)
    *_thr (fuseS_all_thr, fuseH_wf_thr): fused logit minus mean of the same subject's previous <=256 fused logits; no
    shift if fewer than 16 previous rows (== thr_mean256 of depositioned_adaptive_threshold.py).
Evaluation: all / mature (t>=120); AUC, subject-macro bACC, micro bACC; #subjects bACC>=.9 and max; paired subject
bootstrap (1000; smoke 100): sym_all, sym_only, fuseS_all, fuseS_only, fuseS_all_thr, fuseH_wf_thr vs fuseH_wf;
sym_all, sym_only vs all_prank; group (depositioned_subject_diagnosis.json high/mid/low) macro bACC mean/median change
vs fuseH_wf.
Checks (assert): fold == npz; position-only AUC ~.5 (|d|<.03, full run) and == npz; test-label shuffle -> identical
scores; feature prefix invariance (future rows changed -> prefix features + scores identical, changed rows differ);
subject isolation (alone == pooled; test and train side); vectorized vs naive stepwise reference on sampled rows;
hgb_prank part == feat_subject; all_prank reproduction; thr shift vs naive.
Run: python test/depositioned_symmetric_contrast.py [--smoke]   (smoke: stream0, fold0, bootstrap 100)
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from depositioned_history_delta import arrange, make_clf, boot, load_db  # noqa: E402
from depositioned_resnet_history_delta import stream_order, logit, sig  # noqa: E402
from depositioned_feature_multiscale import feat_subject, win_stats, prank, naive_row  # noqa: E402
from depositioned_adaptive_threshold import apply as thr_apply, shift_subject  # noqa: E402
from causal_kernel_state_mapping import metrics  # noqa: E402
from switch_dual_memory import Tee  # noqa: E402

MATURE, STREAM_SEEDS, KS, KP = 120, (0, 1, 2), (32, 128), 128
THR_K, THR_MIN = 256, 16
VARIANTS = ('sym_all', 'sym_only')
T0 = time.perf_counter()


def el():
    return f'elapsed={time.perf_counter() - T0:.1f}s'


def impute_scale(F, tr):
    med = np.nanmedian(np.where(np.isfinite(F[tr]), F[tr], np.nan), 0); med = np.where(np.isfinite(med), med, 0.)
    sc = StandardScaler().fit(np.where(np.isfinite(F), F, med)[tr])
    return lambda A: sc.transform(np.where(np.isfinite(A), A, med))


def sym_subject(zs):
    """One subject's zz in stream order -> {'sym_all', 'sym_only', 'hp'}; row t uses zs[:t+1] only."""
    mu = {k: win_stats(zs, k)[0] for k in KS}
    A = {k: np.abs(zs - mu[k]) for k in KS}
    P = prank(zs, KP); Q = np.abs(P - .5)
    hp = np.hstack([zs, zs - mu[32], P])
    return {'hp': hp, 'sym_all': np.hstack([hp, A[32], A[128]]), 'sym_only': np.hstack([zs, A[32], A[128], Q])}


def naive_sym(zs, t):
    z = zs[t]
    def hm(k): P_ = zs[max(0, t - k):t]; return P_.mean(0) if len(P_) else np.zeros(zs.shape[1])
    P_ = zs[max(0, t - KP):t]
    pr = ((P_ < z).sum(0) + .5 * (P_ == z).sum(0)) / len(P_) if len(P_) else np.full(zs.shape[1], .5)
    a32, a128 = np.abs(z - hm(32)), np.abs(z - hm(128))
    return {'sym_all': np.concatenate([z, z - hm(32), pr, a32, a128]),
            'sym_only': np.concatenate([z, a32, a128, np.abs(pr - .5)])}


def build_all(ZZ, order, v):
    out = np.empty((len(ZZ), {'sym_all': 5, 'sym_only': 4, 'hp': 3}[v] * ZZ.shape[1]))
    for loc in order: out[loc] = sym_subject(ZZ[loc])[v]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); args = ap.parse_args()
    stem = 'depositioned_symmetric_contrast' + ('_smoke' if args.smoke else '')
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
    grp_of = {s: G for G in ('high', 'mid', 'low') for s in diag['diagnosis']['groups'][G]['subjects']}
    print(f'[start] rows={len(y)} subjects={len(subs)} dims={Fc.shape[1]}+{Fn.shape[1]} smoke={args.smoke} '
          f'sseeds={sseeds} groups={ {G: sum(v == G for v in grp_of.values()) for G in ("high", "mid", "low")} }', flush=True)

    Xa, ya, ga, _ = arrange(subs, data, 0, 'independent'); fold_of = {}
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Xa, ya, ga)):
        for s in np.unique(ga[te]): fold_of[s] = k
    fold = np.array([fold_of[s] for s in g]); assert np.array_equal(fold, A_['fold'])
    print('[check] fold == depositioned_aperiodic_emg_oof.npz fold', flush=True)
    run_folds = [0] if args.smoke else list(range(5))

    result = {'meta': {'mature_index': MATURE, 'threshold': .5, 'rows': int(len(y)), 'subjects': len(subs),
                       'stream_seeds': list(sseeds), 'smoke': args.smoke, 'win_ks': list(KS), 'prank_k': KP,
                       'thr': {'k': THR_K, 'min_past': THR_MIN, 'mode': 'mean'}, 'bootstrap_n': nb,
                       'dims': {'zz': 889, 'sym_all': 5 * 889, 'sym_only': 4 * 889}}, 'checks': {}}
    checks = result['checks']
    oof = {'y': y, 'groups': g, 'fold': fold}
    sidx = {s: j for j, s in enumerate(subs)}
    for sd in sseeds:
        order, t = stream_order(g, subs, sd)
        names = ['position_only__lr', 'position_only__hgb', 'all_prank'] + list(VARIANTS)
        S = {n: np.full(len(y), np.nan) for n in names}
        with threadpool_limits(limits=8):
            for k in run_folds:
                tk = time.perf_counter()
                tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k); te_subs = list(np.unique(g[te]))
                T = t[:, None].astype(float)
                for c in ('lr', 'hgb'):
                    S[f'position_only__{c}'][te] = make_clf(c).fit(T[tr], y[tr]).predict_proba(T[te])[:, 1]
                tfo, tfn = impute_scale(Fc, tr), impute_scale(Fn, tr)
                ZZ = np.hstack([tfo(Fc), tfn(Fn)])
                if sd == 0:
                    t1 = time.perf_counter(); Fp = build_all(ZZ, order, 'hp')
                    S['all_prank'][te] = make_clf('hgb').fit(Fp[tr], y[tr]).predict_proba(Fp[te])[:, 1]; del Fp
                    dmax = float(np.abs(S['all_prank'][te] - A_[f'stream{sd}__all_prank'][te]).max())
                    assert dmax < 1e-9, dmax; checks[f'all_prank_reproduced_s{sd}_f{k}'] = dmax
                    print(f'[s{sd} fold {k + 1}/5] all_prank reproduced (maxabs {dmax:.1e}) '
                          f'fit_s={time.perf_counter() - t1:.1f} {el()}', flush=True)
                else:
                    S['all_prank'][te] = A_[f'stream{sd}__all_prank'][te]
                # feature-level checks (vectorized vs naive, hp == feat_subject, prefix invariance, isolation)
                t1 = time.perf_counter(); FEATC = {}
                for s0 in te_subs[:2]:
                    loc = order[sidx[s0]]; zs = ZZ[loc]; cut = len(loc) // 2
                    F0 = sym_subject(zs); FEATC[s0] = F0
                    assert np.array_equal(F0['hp'], feat_subject(zs, 'hgb_prank'))
                    rows = sorted(set([0, 1, 31, 32, 33, 127, 128, 129, cut, len(loc) - 1] +
                                      list(np.random.default_rng([sd, k]).integers(0, len(loc), 6))))
                    for r in rows:
                        nv = naive_sym(zs, r); assert np.allclose(nv['sym_all'][:3 * 889], naive_row(zs, r, 'hgb_prank'))
                        for v in VARIANTS:
                            d = float(np.abs(F0[v][r] - nv[v]).max()); assert d < 1e-9, (v, r, d)
                    za = zs.copy(); za[cut:] += 100.
                    Fa = sym_subject(za)
                    for v in VARIANTS:
                        assert np.array_equal(Fa[v][:cut], F0[v][:cut])
                        assert np.all(np.abs(Fa[v][cut:] - F0[v][cut:]).max(1) > 0)
                        assert np.array_equal(sym_subject(zs[:cut])[v], F0[v][:cut])
                    FEATC[('alt', s0)] = (Fa, cut)
                checks[f'feature_naive_prefix_s{sd}_f{k}'] = 'ok'
                print(f'[s{sd} fold {k + 1}/5] feature checks ok (naive/prefix/hp) s={time.perf_counter() - t1:.1f} {el()}',
                      flush=True)
                s1 = np.unique(g[tr])[0]; loc1 = order[sidx[s1]]; F1 = sym_subject(ZZ[loc1])
                for v in VARIANTS:
                    t1 = time.perf_counter(); F = build_all(ZZ, order, v); tb = time.perf_counter() - t1
                    for s0 in te_subs[:2]: assert np.array_equal(FEATC[s0][v], F[order[sidx[s0]]])  # isolation (test)
                    assert np.array_equal(F1[v], F[loc1])  # isolation (train)
                    clf = make_clf('hgb').fit(F[tr], y[tr]); S[v][te] = clf.predict_proba(F[te])[:, 1]
                    tf = time.perf_counter() - t1 - tb
                    if k == run_folds[0]:  # test-label shuffle -> identical scores
                        ys = y.copy(); ys[te] = np.random.default_rng(k).permutation(y[te])
                        d_sh = float(np.abs(make_clf('hgb').fit(F[tr], ys[tr]).predict_proba(F[te])[:, 1] - S[v][te]).max())
                        assert d_sh == 0., d_sh; checks[f'label_shuffle_s{sd}_f{k}_{v}'] = d_sh
                    for s0 in te_subs[:2]:
                        loc = order[sidx[s0]]; Fa, cut = FEATC[('alt', s0)]
                        pa = clf.predict_proba(Fa[v])[:, 1]
                        d_pre = float(np.abs(pa[:cut] - S[v][loc[:cut]]).max()); d_fut = float(np.abs(pa[cut:] - S[v][loc[cut:]]).max())
                        assert d_pre == 0. and d_fut > 1e-6, (d_pre, d_fut)
                        checks[f'causal_s{sd}_f{k}_{v}_{s0}'] = {'score_prefix_after_future_change': d_pre,
                                                                 'score_future_changed': d_fut}
                    print(f'[s{sd} fold {k + 1}/5] {v:9s} dim={F.shape[1]} build_s={tb:.1f} fit_s={tf:.1f} checks ok '
                          f'(shuffle/prefix/isolation) {el()}', flush=True)
                    del F, clf
                del ZZ, FEATC
                print(f'[s{sd} fold {k + 1}/5] done fold_s={time.perf_counter() - tk:.1f} {el()}', flush=True)
        ev = np.isfinite(S['sym_all'])
        for c in ('lr', 'hgb'):
            n = f'position_only__{c}'; assert np.abs(S[n][ev] - A_[f'stream{sd}__{n}'][ev]).max() < 1e-9
        W, FB = B_[f'stream{sd}__resnet_whiten_logit_delta'], B_[f'stream{sd}__resnet_fbank_logit_delta']
        LH = np.mean([logit(S['all_prank']), logit(W), logit(FB)], 0)
        LA = np.mean([logit(S['sym_all']), logit(W), logit(FB)], 0)
        LO = np.mean([logit(S['sym_only']), logit(W), logit(FB)], 0)
        LH[~ev] = np.nan; LA[~ev] = np.nan; LO[~ev] = np.nan
        S['fuseH_wf'] = sig(LH); S['fuseS_all'] = sig(LA); S['fuseS_only'] = sig(LO)
        S['fuseS_all_thr'] = sig(thr_apply(LA, order, THR_K, 'mean'))
        S['fuseH_wf_thr'] = sig(thr_apply(LH, order, THR_K, 'mean'))
        for loc in order[:2]:  # thr: naive + prefix check
            loc = np.asarray(loc); loc = loc[np.isfinite(LA[loc])]
            if len(loc) == 0: continue
            l0 = LA[loc]; o = shift_subject(l0, THR_K, 'mean')
            for tt in (THR_MIN - 1, THR_MIN, len(l0) // 2, len(l0) - 1):
                ref = l0[tt] if tt < THR_MIN else l0[tt] - l0[max(0, tt - THR_K):tt].mean()
                assert abs(o[tt] - ref) < 1e-12
            assert np.allclose(sig(o), S['fuseS_all_thr'][loc])
        checks[f'thr_naive_s{sd}'] = 'ok'
        M = {'all': ev, 'mature': ev & (t >= MATURE)}
        R = result[f'stream{sd}'] = {'methods': {}, 'bootstrap': {}, 'subjects': {}, 'groups': {}}
        ev_subs = list(np.unique(g[ev])); PS = {}
        for n, p in S.items():
            oof[f'stream{sd}__{n}'] = p
            r = R['methods'][n] = {s: metrics(y, g, p, M[s]) for s in M}
            PS[n] = {s: metrics(y, g, p, ev & (g == s))['subject_macro_bACC'] for s in ev_subs}
            ba = np.array([q for q in PS[n].values() if q is not None])
            R['subjects'][n] = {'n_ge_0.9': int((ba >= .9).sum()), 'max': float(ba.max()), 'n': int(len(ba))}
            print(f'[result stream {sd}] {n:18s} ' + ' | '.join(
                f'{s}: AUC={r[s]["AUC"]:.4f} mac={r[s]["subject_macro_bACC"]:.4f} mic={r[s]["micro_bACC"]:.4f}' for s in M) +
                f' | subj>=.9={R["subjects"][n]["n_ge_0.9"]} max={R["subjects"][n]["max"]:.3f}', flush=True)
        for n in S:
            R['groups'][n] = {}
            for G in ('high', 'mid', 'low'):
                ss = [s for s in ev_subs if grp_of.get(s) == G and PS[n][s] is not None]
                if not ss: continue
                a = np.array([PS[n][s] for s in ss]); b = np.array([PS['fuseH_wf'][s] for s in ss])
                R['groups'][n][G] = {'n': len(ss), 'median': float(np.median(a)), 'mean': float(a.mean()),
                                     'd_median_vs_fuseH_wf': float(np.median(a) - np.median(b)),
                                     'd_mean_vs_fuseH_wf': float(a.mean() - b.mean())}
            print(f'[group stream {sd}] {n:18s} ' + ' | '.join(
                f'{G}(n={q["n"]}): med={q["median"]:.4f} mean={q["mean"]:.4f} dmed={q["d_median_vs_fuseH_wf"]:+.4f} '
                f'dmean={q["d_mean_vs_fuseH_wf"]:+.4f}' for G, q in R['groups'][n].items()), flush=True)
        for c in ('lr', 'hgb'):
            a = R['methods'][f'position_only__{c}']['all']['AUC']
            R.setdefault('sanity', {})[f'position_only__{c}'] = {'AUC': a, 'ok': abs(a - .5) < .03}
            print(f'[sanity stream {sd}] position_only__{c} AUC={a:.4f} -> {"OK" if abs(a - .5) < .03 else "FLAG"}', flush=True)
            if not args.smoke: assert abs(a - .5) < .03
        pairs = [(a_, 'fuseH_wf') for a_ in ('sym_all', 'sym_only', 'fuseS_all', 'fuseS_only', 'fuseS_all_thr',
                                              'fuseH_wf_thr')] + [(a_, 'all_prank') for a_ in VARIANTS]
        for a_, b_ in pairs:
            for seg in M:
                bb = R['bootstrap'][f'{a_}__vs__{b_}__{seg}'] = boot(S[a_], S[b_], y, g, M[seg], nb)
                print(f'[boot stream {sd}] {a_:14s} vs {b_:9s} {seg:6s} '
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
