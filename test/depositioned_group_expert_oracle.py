"""Idea 1 ORACLE upper bound: group routing + group experts (de-positioned streams).

!! LEAKAGE WARNING !! The subject groups (depositioned_subject_diagnosis.json ['diagnosis']['groups'][G]['subjects'],
G in high/mid/low) were derived from the OOF labels/scores of ALL subjects, including each fold's test subjects. Routing
test subjects by their true group therefore uses test-label information. This script is ONLY an upper-bound diagnosis:
if the oracle gain in subject-macro bACC over fuseH_wf is < +.02, the legal (predicted-routing) version is not pursued.

Protocol identical to depositioned_symmetric_contrast.py: stream order = stream_order(g, subs, seed); outer folds =
SGKF(5, shuffle, rs0) on seed-0 'independent' arrangement, re-derived and asserted == depositioned_aperiodic_emg_oof.npz
fold. Per fold: train-median imputation + StandardScaler(train) for the 858 features and the 31 new_features -> ZZ (889).
hp features = feat_subject(zz_s, 'hgb_prank') per subject in stream order (build_all(ZZ, order, 'hp'), 2667 dims;
row t uses the subject's rows <= t only).

Fixed hyper-parameters (set before running):
  Classifier make_clf('hgb') = HGB(200, depth3, lr .06, l2 1, rs0), no feature selection, threadpool_limits(8).
  all_prank : make_clf('hgb') on hp of all training rows (stream0 recomputed, asserted == npz max|d|<1e-9; streams 1/2
              read from depositioned_aperiodic_emg_oof.npz).
  expert_G  : per fold k and group G, make_clf('hgb') trained on hp rows of training subjects whose group is G.
  expert_rand: trained on n_low randomly chosen training subjects (n_low = #low training subjects of fold k;
              rng = default_rng([stream_seed, k]), choice without replacement) -- group-agnostic control.
  Test-side (ORACLE routing by true group):
    exp_oracle    = expert_{group(s)} score for each test subject s
    mix_oracle    = sig(mean logit(all_prank, exp_oracle))
    low_only      = expert_low for low subjects, all_prank otherwise
    rand_low_only = expert_rand for low subjects, all_prank otherwise
  whiten / fbank = stream{s}__resnet_whiten_logit_delta / stream{s}__resnet_fbank_logit_delta
                   (depositioned_resnet_timefreq_oof.npz). Fusion = sig(mean logit(., whiten, fbank)), threshold .5:
    fuseH_wf (all_prank), fuseX_wf (exp_oracle), fuseM_wf (mix_oracle), fuseL_wf (low_only), fuseR_wf (rand_low_only)
    fuseX_wf_thr = thr_mean256 of fuseX_wf logit (minus mean of same subject's previous <=256 fused logits; no shift if
                   fewer than 16 previous rows; == apply(l, order, 256, 'mean') of depositioned_adaptive_threshold.py).
    fuseH_wf_thr = same shift applied to fuseH_wf (reference; extra bootstrap fuseX_wf_thr vs fuseH_wf_thr).
Evaluation: all / mature (t>=120); AUC, subject-macro bACC, micro bACC; #subjects bACC>=.9 and max; paired subject
bootstrap (1000; smoke 100) of every variant vs fuseH_wf; group (high/mid/low) macro bACC mean and mean/median change
vs fuseH_wf.
Checks (assert): fold == npz; all_prank reproduction (stream0); test-label shuffle -> identical expert scores (first run
fold, every expert); subject isolation (hp of one subject built alone == pooled build; test and train side).
Run: python test/depositioned_group_expert_oracle.py [--smoke]   (smoke: stream0, fold0, bootstrap 100)
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from depositioned_history_delta import arrange, make_clf, boot, load_db  # noqa: E402
from depositioned_resnet_history_delta import stream_order, logit, sig  # noqa: E402
from depositioned_feature_multiscale import feat_subject  # noqa: E402
from depositioned_adaptive_threshold import apply as thr_apply  # noqa: E402
from depositioned_symmetric_contrast import impute_scale, build_all  # noqa: E402
from causal_kernel_state_mapping import metrics  # noqa: E402
from switch_dual_memory import Tee  # noqa: E402

MATURE, STREAM_SEEDS = 120, (0, 1, 2)
THR_K = 256
GROUPS = ('high', 'mid', 'low')
EXPERTS = GROUPS + ('rand',)
T0 = time.perf_counter()


def el():
    return f'elapsed={time.perf_counter() - T0:.1f}s'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); args = ap.parse_args()
    stem = 'depositioned_group_expert_oracle' + ('_smoke' if args.smoke else '')
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
    grp_of = {s: G for G in GROUPS for s in diag['diagnosis']['groups'][G]['subjects']}
    assert all(s in grp_of for s in subs), [s for s in subs if s not in grp_of]
    grp_row = np.array([grp_of[s] for s in g])
    print(f'[start] rows={len(y)} subjects={len(subs)} dims={Fc.shape[1]}+{Fn.shape[1]} smoke={args.smoke} '
          f'sseeds={sseeds} groups={ {G: sum(v == G for v in grp_of.values()) for G in GROUPS} } (ORACLE, leaky)',
          flush=True)

    Xa, ya, ga, _ = arrange(subs, data, 0, 'independent'); fold_of = {}
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Xa, ya, ga)):
        for s in np.unique(ga[te]): fold_of[s] = k
    fold = np.array([fold_of[s] for s in g]); assert np.array_equal(fold, A_['fold'])
    print('[check] fold == depositioned_aperiodic_emg_oof.npz fold', flush=True)
    run_folds = [0] if args.smoke else list(range(5))

    result = {'meta': {'oracle_leaky_upper_bound': True, 'mature_index': MATURE, 'threshold': .5, 'rows': int(len(y)),
                       'subjects': len(subs), 'stream_seeds': list(sseeds), 'smoke': args.smoke,
                       'thr': {'k': THR_K, 'min_past': 16, 'mode': 'mean'}, 'bootstrap_n': nb,
                       'dims': {'zz': 889, 'hp': 3 * 889}}, 'checks': {}, 'fold_train_counts': {}}
    checks = result['checks']
    oof = {'y': y, 'groups': g, 'fold': fold}
    sidx = {s: j for j, s in enumerate(subs)}
    for sd in sseeds:
        order, t = stream_order(g, subs, sd)
        S = {n: np.full(len(y), np.nan) for n in ('all_prank',) + tuple(f'expert_{e}' for e in EXPERTS)}
        with threadpool_limits(limits=8):
            for k in run_folds:
                tk = time.perf_counter()
                tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k); te_subs = list(np.unique(g[te]))
                tfo, tfn = impute_scale(Fc, tr), impute_scale(Fn, tr)
                ZZ = np.hstack([tfo(Fc), tfn(Fn)])
                t1 = time.perf_counter(); Fp = build_all(ZZ, order, 'hp')
                for s0 in te_subs[:2] + [np.unique(g[tr])[0]]:  # subject isolation (test + train side)
                    loc = order[sidx[s0]]
                    assert np.array_equal(feat_subject(ZZ[loc], 'hgb_prank'), Fp[loc])
                checks[f'isolation_s{sd}_f{k}'] = 'ok'
                print(f'[s{sd} fold {k + 1}/5] hp built dim={Fp.shape[1]} s={time.perf_counter() - t1:.1f} '
                      f'isolation ok {el()}', flush=True)
                if sd == 0:
                    t1 = time.perf_counter()
                    S['all_prank'][te] = make_clf('hgb').fit(Fp[tr], y[tr]).predict_proba(Fp[te])[:, 1]
                    dmax = float(np.abs(S['all_prank'][te] - A_[f'stream{sd}__all_prank'][te]).max())
                    assert dmax < 1e-9, dmax; checks[f'all_prank_reproduced_s{sd}_f{k}'] = dmax
                    print(f'[s{sd} fold {k + 1}/5] all_prank reproduced (maxabs {dmax:.1e}) '
                          f'fit_s={time.perf_counter() - t1:.1f} {el()}', flush=True)
                else:
                    S['all_prank'][te] = A_[f'stream{sd}__all_prank'][te]
                tr_subs = np.unique(g[tr])
                tr_grp = {G: [s for s in tr_subs if grp_of[s] == G] for G in GROUPS}
                n_low = len(tr_grp['low'])
                rand_subs = list(np.random.default_rng([sd, k]).choice(tr_subs, n_low, replace=False))
                n_rand_low = sum(grp_of[s] == 'low' for s in rand_subs)
                cnt = {G: len(v) for G, v in tr_grp.items()}; cnt['rand'] = n_low; cnt['rand_of_which_low'] = n_rand_low
                cnt['test'] = {G: sum(grp_of[s] == G for s in te_subs) for G in GROUPS}
                result['fold_train_counts'][f's{sd}_f{k}'] = cnt
                print(f'[s{sd} fold {k + 1}/5] train subjects per group {cnt}', flush=True)
                for e in EXPERTS:
                    t1 = time.perf_counter()
                    ss = rand_subs if e == 'rand' else tr_grp[e]
                    tre = tr[np.isin(g[tr], ss)]
                    clf = make_clf('hgb').fit(Fp[tre], y[tre]); S[f'expert_{e}'][te] = clf.predict_proba(Fp[te])[:, 1]
                    msg = ''
                    if k == run_folds[0]:  # test-label shuffle -> identical expert scores
                        ys = y.copy(); ys[te] = np.random.default_rng(k).permutation(y[te])
                        d_sh = float(np.abs(make_clf('hgb').fit(Fp[tre], ys[tre]).predict_proba(Fp[te])[:, 1]
                                            - S[f'expert_{e}'][te]).max())
                        assert d_sh == 0., d_sh; checks[f'label_shuffle_s{sd}_f{k}_{e}'] = d_sh; msg = ' shuffle ok'
                    print(f'[s{sd} fold {k + 1}/5] expert_{e:4s} n_subj={len(ss)} rows={len(tre)} '
                          f'fit_s={time.perf_counter() - t1:.1f}{msg} {el()}', flush=True)
                    del clf
                del ZZ, Fp
                print(f'[s{sd} fold {k + 1}/5] done fold_s={time.perf_counter() - tk:.1f} {el()}', flush=True)
        ev = np.isfinite(S['expert_low'])
        low = grp_row == 'low'
        Ex = {G: S[f'expert_{G}'] for G in GROUPS}
        S['exp_oracle'] = np.select([grp_row == G for G in GROUPS], [Ex[G] for G in GROUPS], np.nan)
        S['mix_oracle'] = sig(np.mean([logit(S['all_prank']), logit(S['exp_oracle'])], 0))
        S['low_only'] = np.where(low, S['expert_low'], S['all_prank'])
        S['rand_low_only'] = np.where(low, S['expert_rand'], S['all_prank'])
        W, FB = B_[f'stream{sd}__resnet_whiten_logit_delta'], B_[f'stream{sd}__resnet_fbank_logit_delta']
        L = {}
        for n, base in (('fuseH_wf', 'all_prank'), ('fuseX_wf', 'exp_oracle'), ('fuseM_wf', 'mix_oracle'),
                        ('fuseL_wf', 'low_only'), ('fuseR_wf', 'rand_low_only')):
            l_ = np.mean([logit(S[base]), logit(W), logit(FB)], 0); l_[~ev] = np.nan; L[n] = l_; S[n] = sig(l_)
        dH = float(np.nanmax(np.abs(S['fuseH_wf'][ev] - sig(np.mean(
            [logit(A_[f'stream{sd}__all_prank']), logit(W), logit(FB)], 0))[ev])))
        assert dH < 1e-9, dH; checks[f'fuseH_wf_matches_npz_s{sd}'] = dH
        S['fuseX_wf_thr'] = sig(thr_apply(L['fuseX_wf'], order, THR_K, 'mean'))
        S['fuseH_wf_thr'] = sig(thr_apply(L['fuseH_wf'], order, THR_K, 'mean'))
        for n in S: S[n] = np.where(ev, S[n], np.nan)
        M = {'all': ev, 'mature': ev & (t >= MATURE)}
        R = result[f'stream{sd}'] = {'methods': {}, 'bootstrap': {}, 'subjects': {}, 'groups': {}}
        ev_subs = list(np.unique(g[ev])); PS = {}
        for n, p in S.items():
            oof[f'stream{sd}__{n}'] = p
            r = R['methods'][n] = {s: metrics(y, g, p, M[s]) for s in M}
            PS[n] = {s: metrics(y, g, p, ev & (g == s))['subject_macro_bACC'] for s in ev_subs}
            ba = np.array([q for q in PS[n].values() if q is not None])
            R['subjects'][n] = {'n_ge_0.9': int((ba >= .9).sum()), 'max': float(ba.max()), 'n': int(len(ba))}
            print(f'[result stream {sd}] {n:16s} ' + ' | '.join(
                f'{s}: AUC={r[s]["AUC"]:.4f} mac={r[s]["subject_macro_bACC"]:.4f} mic={r[s]["micro_bACC"]:.4f}' for s in M) +
                f' | subj>=.9={R["subjects"][n]["n_ge_0.9"]} max={R["subjects"][n]["max"]:.3f}', flush=True)
        for n in S:
            R['groups'][n] = {}
            for G in GROUPS:
                ss = [s for s in ev_subs if grp_of.get(s) == G and PS[n][s] is not None]
                if not ss: continue
                a = np.array([PS[n][s] for s in ss]); b = np.array([PS['fuseH_wf'][s] for s in ss])
                R['groups'][n][G] = {'n': len(ss), 'median': float(np.median(a)), 'mean': float(a.mean()),
                                     'd_median_vs_fuseH_wf': float(np.median(a) - np.median(b)),
                                     'd_mean_vs_fuseH_wf': float(a.mean() - b.mean())}
            print(f'[group stream {sd}] {n:16s} ' + ' | '.join(
                f'{G}(n={q["n"]}): mean={q["mean"]:.4f} dmean={q["d_mean_vs_fuseH_wf"]:+.4f} '
                f'dmed={q["d_median_vs_fuseH_wf"]:+.4f}' for G, q in R['groups'][n].items()), flush=True)
        for a_, b_ in [(n, 'fuseH_wf') for n in S if n != 'fuseH_wf'] + [('fuseX_wf_thr', 'fuseH_wf_thr')]:
            for seg in M:
                bb = R['bootstrap'][f'{a_}__vs__{b_}__{seg}'] = boot(S[a_], S[b_], y, g, M[seg], nb)
                print(f'[boot stream {sd}] {a_:14s} vs {b_:12s} {seg:6s} '
                      f'dAUC={bb["dAUC"]["mean"]:+.4f}[{bb["dAUC"]["p2.5"]:+.4f},{bb["dAUC"]["p97.5"]:+.4f}] '
                      f'dmac={bb["dmacro_bACC"]["mean"]:+.4f}[{bb["dmacro_bACC"]["p2.5"]:+.4f},{bb["dmacro_bACC"]["p97.5"]:+.4f}] '
                      f'dmic={bb["dmicro_bACC"]["mean"]:+.4f}[{bb["dmicro_bACC"]["p2.5"]:+.4f},{bb["dmicro_bACC"]["p97.5"]:+.4f}]',
                      flush=True)
        (OUT / (stem + '.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    result['seconds'] = time.perf_counter() - T0
    (OUT / (stem + '.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    if not args.smoke: np.savez_compressed(OUT / (stem + '_oof.npz'), **oof)
    print(f'[done] all checks passed {el()} -> {stem}.json', flush=True)
    sys.stdout = tee.stream; tee.f.close()


if __name__ == '__main__':
    main()
