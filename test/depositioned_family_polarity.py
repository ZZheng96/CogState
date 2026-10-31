"""Direction G: per-feature-family unlabeled direction alignment (de-positioned streams).

Protocol: stream order = concat(relax,task) permuted by default_rng([stream_seed, subject_index]) (stream_order), stream
seeds 0/1/2; outer folds = subject->fold map stored in depositioned_feature_multiscale_oof.npz (SGKF(5, rs0)); fusionM_prank
and position-only scores reused from the same file. Threshold .5; paired subject bootstrap 1000 (boot).
Fixed hyperparameters (set before any run, not tuned):
  Features: per fold, global median impute (train fold) + StandardScaler(train fold) -> z; hgb_prank feature map
    [z, z-h32, prank128] (depositioned_feature_multiscale.feat_subject, causal within subject). Families = the 12
    FAMILY_RULES of depositioned_lowgroup_diagnosis.families; a family uses its 3 blocks (z, z-h32, prank) of its columns.
  Family models (cross-subject, training-fold subjects only): lr = LogisticRegression(C=.05, max_iter 2000) on
    StandardScaler(train) of the family block; hgb = make_clf('hgb') (HGB 200, depth 3, lr .06, l2 1, rs0).
    Family score s_f = decision_function; standardized by mean/std of s_f on the training rows -> z_f.
  Test-time alignment (unlabeled, causal; row t uses rows 0..t of its own subject stream only):
    none   : famalign = mean_f z_f.
    corr   : r_f(t) = Pearson corr over prefix rows 0..t of z_f vs mean of the other 11 z (unaligned); sign_f = -1 iff
             prefix length t+1 >= MATURE (=120, depositioned_feature_multiscale.MATURE) and r_f < 0, else +1.
    pca    : PC1 of the prefix covariance of the 12 z columns (rows 0..t), oriented so sum(loadings) >= 0;
             sign_f = sign(loading_f) (0 -> +1) iff t+1 >= MATURE, else +1.
    oracle : (UPPER BOUND, uses test labels) sign_f = sign(AUC(y_s, z_f over all windows of s) - .5), fixed per subject.
    famalign logit = mean_f sign_f * z_f.  fusionG = sigmoid(mean(logit fusionM_prank, famalign logit)).
Reported: per stream AUC / macro bACC (all, mature t>=120); bootstrap fusionG_* vs fusionM_prank; direction accuracy
(sign at final prefix and over mature rows vs oracle sign, overall / per family / on oracle-negative pairs; baseline =
all +1); per group (depositioned_subject_diagnosis.json high/mid/low) median subject bACC and median change vs
fusionM_prank; # subjects with bACC >= .9. Main pre-declared method: fusionG_lr_corr.
Checks (assert): position-only AUC ~.5; subject isolation (features alone == pooled; alignment is per-subject);
test-label shuffle -> identical family scores (lr refit); prefix invariance (future rows +100 -> prefix scores of all
variants unchanged, future changed).
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from depositioned_history_delta import make_clf, boot, load_db  # noqa: E402
from depositioned_resnet_history_delta import stream_order, logit, sig  # noqa: E402
from depositioned_feature_multiscale import feat_subject, feat_all, MATURE  # noqa: E402
from depositioned_lowgroup_diagnosis import families  # noqa: E402
from causal_kernel_state_mapping import metrics  # noqa: E402
from switch_dual_memory import Tee  # noqa: E402

STREAM_SEEDS, MODELS, ALIGNS = (0, 1, 2), ('lr', 'hgb'), ('none', 'corr', 'pca', 'oracle')
D0 = 858
T0 = time.perf_counter()


def el():
    return f'elapsed={time.perf_counter() - T0:.1f}s'


def new_model(m):
    return make_pipeline(StandardScaler(), LogisticRegression(C=.05, max_iter=2000)) if m == 'lr' else make_clf('hgb')


def fit_family(m, Xtr, ytr, Xte):
    with threadpool_limits(limits=2):
        mdl = new_model(m).fit(Xtr, ytr); s = mdl.decision_function(Xtr)
        return mdl, float(s.mean()), float(s.std() + 1e-9), mdl.decision_function(Xte)


def align(Zs):
    """Zs: n x F family z-scores of one subject in stream order. Returns {variant: (logit n, sign n x F)}."""
    n, F = Zs.shape; c = np.arange(1, n + 1)[:, None]; mature = (np.arange(n) + 1 >= MATURE)[:, None]
    out = {'none': (Zs.mean(1), np.ones((n, F)))}
    O = (Zs.sum(1, keepdims=True) - Zs) / (F - 1)
    cs = lambda A: np.cumsum(A, 0) / c  # noqa: E731
    mx, mo = cs(Zs), cs(O)
    cov = cs(Zs * O) - mx * mo; vx = cs(Zs * Zs) - mx ** 2; vo = cs(O * O) - mo ** 2
    with np.errstate(invalid='ignore', divide='ignore'):
        r = cov / np.sqrt(np.maximum(vx * vo, 0.))
    sg = np.where(mature & (r < 0), -1., 1.); out['corr'] = ((sg * Zs).mean(1), sg)
    M1 = np.cumsum(Zs, 0); M2 = np.cumsum(Zs[:, :, None] * Zs[:, None, :], 0); sp = np.ones((n, F))
    for t in range(MATURE - 1, n):
        mu = M1[t] / (t + 1); C = M2[t] / (t + 1) - np.outer(mu, mu)
        v = np.linalg.eigh(C)[1][:, -1]; v = v if v.sum() >= 0 else -v
        sp[t] = np.where(v < 0, -1., 1.)
    out['pca'] = ((sp * Zs).mean(1), sp)
    return out


def oracle_sign(ys, Zs):
    return np.array([1. if roc_auc_score(ys, Zs[:, f]) >= .5 else -1. for f in range(Zs.shape[1])])


def subj_scores(Zs, ys):
    A = align(Zs); so = oracle_sign(ys, Zs); A['oracle'] = ((so * Zs).mean(1), np.tile(so, (len(Zs), 1)))
    return A


def summarize(name, y, g, p, M, res):
    r = {seg: metrics(y, g, p, mk) for seg, mk in M.items()}
    ba = {s: metrics(y, g, p, M['all'] & (g == s))['subject_macro_bACC'] for s in np.unique(g[M['all']])}
    r['n_bACC_ge_0.9'] = int(sum(v >= .9 for v in ba.values())); r['subject_bACC'] = ba
    res[name] = r
    print(f'[result] {name:24s} ' + ' | '.join(f'{seg}: AUC={r[seg]["AUC"]:.4f} mac={r[seg]["subject_macro_bACC"]:.4f}'
                                              for seg in M) + f' | n>=.9={r["n_bACC_ge_0.9"]}', flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); args = ap.parse_args()
    stem = 'depositioned_family_polarity' + ('_smoke' if args.smoke else '')
    tee = Tee(sys.stdout, OUT / (stem + '.log')); sys.stdout = tee
    man, subs, data = load_db(); names = man['feature_names']; assert len(names) == D0
    Fc = np.vstack([data[s][0] for s in subs]); y = np.concatenate([data[s][1] for s in subs])
    g = np.concatenate([np.full(len(data[s][1]), s) for s in subs])
    ms = np.load(OUT / 'depositioned_feature_multiscale_oof.npz')
    assert np.array_equal(ms['y'], y) and np.array_equal(ms['groups'], g); fold = ms['fold']
    diag = json.loads((OUT / 'depositioned_subject_diagnosis.json').read_text(encoding='utf-8'))['diagnosis']
    grp_of = {s: G for G, v in diag['groups'].items() for s in v['subjects']}
    fam = families(names); FN = list(fam); NF = len(FN)
    cols = {f: np.concatenate([fam[f] + k * D0 for k in range(3)]) for f in FN}
    sseeds = STREAM_SEEDS[:1] if args.smoke else STREAM_SEEDS; run_folds = [0] if args.smoke else list(range(5))
    nb = 100 if args.smoke else 1000
    print(f'[start] rows={len(y)} subjects={len(subs)} smoke={args.smoke} sseeds={sseeds} folds={run_folds} '
          f'families={NF} models={MODELS} aligns={ALIGNS} mature={MATURE}', flush=True)
    res = {'meta': {'families': {f: len(fam[f]) for f in FN}, 'mature': MATURE, 'models': MODELS, 'aligns': ALIGNS,
                    'smoke': args.smoke, 'n_boot': nb, 'main': 'fusionG_lr_corr',
                    'fusion': 'sig(mean(logit fusionM_prank, mean_f sign_f z_f))'}, 'checks': {}}
    checks = res['checks']; oof = {'y': y, 'groups': g, 'fold': fold}; sidx = {s: j for j, s in enumerate(subs)}
    for sd in sseeds:
        order, t = stream_order(g, subs, sd); fm = ms[f'stream{sd}__fusionM_prank']
        ZF = {m: np.full((len(y), NF), np.nan) for m in MODELS}
        S = {f'fam_{m}_{a}': np.full(len(y), np.nan) for m in MODELS for a in ALIGNS}
        SIGN = {(m, a): np.full((len(y), NF), np.nan) for m in MODELS for a in ALIGNS}
        for k in run_folds:
            tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k); te_subs = list(np.unique(g[te]))
            assert not set(te_subs) & set(np.unique(g[tr]))
            med = np.nanmedian(np.where(np.isfinite(Fc[tr]), Fc[tr], np.nan), 0); med = np.where(np.isfinite(med), med, 0.)
            Fi = np.where(np.isfinite(Fc), Fc, med); Z = StandardScaler().fit(Fi[tr]).transform(Fi)
            t1 = time.perf_counter(); F = feat_all(Z, order, 'hgb_prank')
            print(f'[s{sd} fold {k + 1}/5] prank features {F.shape} build_s={time.perf_counter() - t1:.1f} '
                  f'test_subjects={len(te_subs)} {el()}', flush=True)
            for s0 in te_subs[:2] + [np.unique(g[tr])[0]]:  # isolation: subject alone == pooled
                loc = order[sidx[s0]]; assert np.array_equal(feat_subject(Z[loc], 'hgb_prank'), F[loc])
            checks[f'isolation_features_s{sd}_f{k}'] = True
            jobs = [(m, f) for m in ('hgb', 'lr') for f in FN]
            t1 = time.perf_counter()
            out = Parallel(n_jobs=8, backend='loky')(delayed(fit_family)(m, F[tr][:, cols[f]], y[tr], F[te][:, cols[f]])
                                                     for m, f in jobs)
            MD = {}
            for (m, f), (mdl, mu, sdv, ste) in zip(jobs, out):
                MD[(m, f)] = (mdl, mu, sdv); ZF[m][te, FN.index(f)] = (ste - mu) / sdv
            print(f'[s{sd} fold {k + 1}/5] {len(jobs)} family models fit_s={time.perf_counter() - t1:.1f} ' + ' '.join(
                f'{f}:{roc_auc_score(y[te], ZF["lr"][te, FN.index(f)]):.3f}' for f in FN) + f' (lr test AUC) {el()}', flush=True)
            if k == run_folds[0]:  # test-label shuffle -> identical lr family scores
                y2 = y.copy(); y2[te] = np.random.default_rng(1).permutation(y[te])
                for f in FN[:3]:
                    _, mu, sdv, ste = fit_family('lr', F[tr][:, cols[f]], y2[tr], F[te][:, cols[f]])
                    assert np.array_equal((ste - mu) / sdv, ZF['lr'][te, FN.index(f)])
                checks[f'test_label_shuffle_invariant_s{sd}'] = True
            for j, s in enumerate(te_subs):
                loc = order[sidx[s]]
                for m in MODELS:
                    A = subj_scores(ZF[m][loc], y[loc])
                    for a in ALIGNS: S[f'fam_{m}_{a}'][loc] = A[a][0]; SIGN[(m, a)][loc] = A[a][1]
                if j % 4 == 0 or j == len(te_subs) - 1:
                    print(f'[s{sd} fold {k + 1}/5 align {j + 1}/{len(te_subs)}] {s} {el()}', flush=True)
            for s in te_subs[:2]:  # prefix invariance (all legal variants, both models)
                loc = order[sidx[s]]; n = len(loc); cut = n // 2; alt = Z[loc].copy(); alt[cut:] += 100.
                Fa = feat_subject(alt, 'hgb_prank')
                assert np.array_equal(Fa[:cut], F[loc][:cut])
                for m in MODELS:
                    Za = np.stack([(MD[(m, f)][0].decision_function(Fa[:, cols[f]]) - MD[(m, f)][1]) / MD[(m, f)][2]
                                   for f in FN], 1)
                    Aa = align(Za)
                    for a in ('none', 'corr', 'pca'):
                        ref = S[f'fam_{m}_{a}'][loc]
                        dp = float(np.abs(Aa[a][0][:cut] - ref[:cut]).max()); df = float(np.abs(Aa[a][0][cut:] - ref[cut:]).max())
                        assert dp < 1e-9 and df > 1e-6, (m, a, dp, df)
                        checks[f'prefix_s{sd}_f{k}_{s}_{m}_{a}'] = {'prefix': dp, 'future': df}
            print(f'[s{sd} fold {k + 1}/5] checks ok (isolation/shuffle/prefix) {el()}', flush=True)
            del F, Z, Fi, MD, out
        ev = np.isfinite(S['fam_lr_none']); M = {'all': ev, 'mature': ev & (t >= MATURE)}
        R = res[f'stream{sd}'] = {'methods': {}, 'bootstrap': {}, 'sign_acc': {}, 'groups': {}, 'sanity': {}}
        for c in ('lr', 'hgb'):
            a_ = metrics(y, g, ms[f'stream{sd}__position_only__{c}'], ev)['AUC']; ok = abs(a_ - .5) < .03
            R['sanity'][f'position_only__{c}'] = {'AUC': a_, 'ok': ok}
            print(f'[sanity stream {sd}] position_only__{c} AUC={a_:.4f} -> {"OK" if ok else "FLAG"}', flush=True)
            if not args.smoke: assert ok
        P = {'fusionM_prank': np.where(ev, fm, np.nan)}
        for m in MODELS:
            for a in ALIGNS:
                P[f'fam_{m}_{a}'] = sig(S[f'fam_{m}_{a}'])
                P[f'fusionG_{m}_{a}'] = sig((logit(fm) + S[f'fam_{m}_{a}']) / 2)
        for n_, p in P.items(): summarize(f's{sd}_{n_}', y, g, p, M, R['methods'])
        for m in MODELS:
            for a in ALIGNS:
                for b_, seg in (('fusionM_prank', 'all'), ('fusionM_prank', 'mature'), (f'fusionG_{m}_none', 'all')):
                    bb = R['bootstrap'][f'fusionG_{m}_{a}__vs__{b_}__{seg}'] = boot(P[f'fusionG_{m}_{a}'], P[b_], y, g, M[seg], nb)
                    print(f'[boot stream {sd}] fusionG_{m}_{a:6s} vs {b_:18s} {seg:6s} '
                          f'dAUC={bb["dAUC"]["mean"]:+.4f}[{bb["dAUC"]["p2.5"]:+.4f},{bb["dAUC"]["p97.5"]:+.4f}] '
                          f'dmac={bb["dmacro_bACC"]["mean"]:+.4f}[{bb["dmacro_bACC"]["p2.5"]:+.4f},{bb["dmacro_bACC"]["p97.5"]:+.4f}]',
                          flush=True)
        # direction accuracy vs oracle sign
        es = list(np.unique(g[ev]))
        for m in MODELS:
            OS = np.array([SIGN[(m, 'oracle')][order[sidx[s]][0]] for s in es])  # subjects x F
            R['sign_acc'][m] = {'oracle_neg_frac': float((OS < 0).mean()),
                                'oracle_neg_frac_by_family': dict(zip(FN, (OS < 0).mean(0).tolist())),
                                'baseline_all_plus': float((OS > 0).mean())}
            for a in ('corr', 'pca'):
                FIN = np.array([SIGN[(m, a)][order[sidx[s]][-1]] for s in es])
                MAT = np.array([(SIGN[(m, a)][order[sidx[s]][MATURE - 1:]] == OS[i]).mean(0) for i, s in enumerate(es)])
                neg = OS < 0
                R['sign_acc'][m][a] = {'final_acc': float((FIN == OS).mean()), 'mature_rows_acc': float(MAT.mean()),
                                       'final_acc_on_oracle_neg': float((FIN[neg] == OS[neg]).mean()) if neg.any() else None,
                                       'final_flip_rate': float((FIN < 0).mean()),
                                       'final_acc_by_family': dict(zip(FN, (FIN == OS).mean(0).tolist()))}
                q = R['sign_acc'][m][a]
                print(f'[sign stream {sd}] {m} {a:4s} final_acc={q["final_acc"]:.3f} mature_rows_acc={q["mature_rows_acc"]:.3f} '
                      f'acc_on_oracle_neg={q["final_acc_on_oracle_neg"]} flip_rate={q["final_flip_rate"]:.3f} '
                      f'(baseline all+ {R["sign_acc"][m]["baseline_all_plus"]:.3f})', flush=True)
            print(f'[sign stream {sd}] {m} oracle negative frac by family: ' + ' '.join(
                f'{f}={v:.2f}' for f, v in R['sign_acc'][m]['oracle_neg_frac_by_family'].items()), flush=True)
        # groups
        base = R['methods'][f's{sd}_fusionM_prank']['subject_bACC']
        for n_ in P:
            ba = R['methods'][f's{sd}_{n_}']['subject_bACC']; R['groups'][n_] = {}
            for G in ('high', 'mid', 'low'):
                ss = [s for s in es if grp_of[s] == G]
                if not ss: continue
                R['groups'][n_][G] = {'n': len(ss), 'median_bACC': float(np.median([ba[s] for s in ss])),
                                      'median_delta_vs_fusionM': float(np.median([ba[s] - base[s] for s in ss]))}
            if n_.startswith('fusionG'):
                print(f'[group stream {sd}] {n_:20s} ' + ' '.join(
                    f'{G}: med={v["median_bACC"]:.3f} d={v["median_delta_vs_fusionM"]:+.3f}' for G, v in R['groups'][n_].items()),
                    flush=True)
        for n_, p in P.items(): oof[f'stream{sd}__{n_}'] = p
        (OUT / (stem + '.json')).write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding='utf-8')
    res['seconds'] = time.perf_counter() - T0
    (OUT / (stem + '.json')).write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding='utf-8')
    np.savez_compressed(OUT / (stem + '_oof.npz'), **oof)
    print(f'[done] {el()} -> {stem}.json', flush=True)
    sys.stdout = tee.stream; tee.f.close()


if __name__ == '__main__':
    main()
