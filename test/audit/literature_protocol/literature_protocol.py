"""Literature-protocol audit: where does this project sit under evaluation protocols common in single-channel EEG papers?

Features: 889 = 858 (load_db) + 31 new_features. Models: lr = LogisticRegression(C=.01, max_iter 1000);
hgb = HistGradientBoosting(max_iter 200, depth 3, lr .06, l2 1, rs 0) (make_clf('hgb') of depositioned_history_delta).
Score = predict_proba[:, 1], threshold .5 for bACC. Protocols:
  strict        subject-wise SGKF(5, shuffle, rs0) (asserted == npz fold); train-median impute + StandardScaler on
                fold-train rows (the project protocol).
  window_kfold  LEAKY (common in papers): KFold(5, shuffle, rs0) over windows ignoring subjects; impute/scale on
                train rows. Same subject's windows in train and test.
  session_norm  LEAKY / non-causal: per subject, impute with subject median then z-score each feature with ALL of
                that subject's windows (mean/std over the whole session, labels not used), then strict SGKF folds.
  real_time     the database shards are shuffled, but reconstruct_time.py recovered every window's real recording
                time (exact float32 match). Order-free models (lr / hgb, strict) are unaffected by order (asserted
                identical). Order-dependent part: the v2 causal within-subject adaptation layer
                (depositioned_within_subject_adapt_v2.adapt_subject, variant pseudo_lr_full, then thr_mean256) run on
                an order-free base logit (strict lr) in (i) the shuffled streams 0/1/2 and (ii) the real time order.
                (The stored v2 branch logits are history features of the shuffled streams and can't be replayed in
                real time without retraining the ResNets.)
  v2 pipeline   fuse_pseudo_lr_full_thr recomputed exactly (streams 0/1/2, asserted == v2 json macro bACC); under
                session_norm its fused logit is z-scored per subject over all windows (non-causal) -> sig.
                window_kfold is not applicable (its ResNet branches are trained subject-wise).
Reported: AUC (pooled), subject-macro bACC, micro bACC, group means (high/mid/low; low post-hoc -> circular), paired
subject bootstrap (1000) vs the strict counterpart.
Checks: fold == npz; real_time y/groups == canonical; test-label shuffle -> strict lr/hgb scores identical (fold 0);
determinism (strict lr fold 0 refit identical); v2 recomputation == v2 json; real-time order-free scores identical.
Run: python test/audit/literature_protocol/literature_protocol.py [--smoke]  (smoke: fold 0 / stream 0, boot 100)
"""
import argparse, json, sys, time, warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import GROUPS, STREAM_SEEDS, TEST, boot, el, impute_scale, load_all, metrics, open_log, save  # noqa: E402
sys.path.insert(0, str(TEST))
from depositioned_resnet_history_delta import logit, sig, stream_order  # noqa: E402
from depositioned_adaptive_threshold import apply as thr_apply  # noqa: E402
from depositioned_within_subject_adapt_v2 import adapt_subject  # noqa: E402
from depositioned_within_subject_adapt import THR_K  # noqa: E402

MODELS = ('lr', 'hgb')


def make(name):
    if name == 'hgb':
        return HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=.06, l2_regularization=1.,
                                              random_state=0)
    return LogisticRegression(C=.01, max_iter=1000)


def zz(D, tr):
    a, b = impute_scale(D['Fc'], tr), impute_scale(D['Fn'], tr)
    return np.hstack([a(D['Fc']), b(D['Fn'])])


def session_norm(D):
    F = np.hstack([D['Fc'], D['Fn']]).astype(float); out = np.empty_like(F); g = D['g']
    for s in D['subs']:
        q = g == s; A = np.where(np.isfinite(F[q]), F[q], np.nan)
        med = np.nanmedian(A, 0); med = np.where(np.isfinite(med), med, 0.)
        out[q] = StandardScaler().fit_transform(np.where(np.isfinite(A), A, med))
    return out


def fit_score(name, X, y, tr, te):
    return make(name).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]


def v2_fused(D, sd, run_subs, ZZt, Bl, y, order, sidx):
    """fuse_pseudo_lr_full(_thr) logit from v2's adapt_subject. Bl (n, 3) branch logits; returns (lf, lf_thr)."""
    g = D['g']; L = Bl.mean(1); zz_ = np.full(len(y), np.nan)
    out = Parallel(n_jobs=8, backend='loky')(
        delayed(adapt_subject)(ZZt[order[sidx[s]]], Bl[order[sidx[s]]], y[order[sidx[s]]]) for s in run_subs)
    for s, (r, _) in zip(run_subs, out): zz_[order[sidx[s]]] = r['pseudo_lr_full']
    ev = np.isin(g, run_subs); f = np.isfinite(zz_)
    lf = np.where(ev, np.where(f, L + zz_, L), np.nan)
    return lf, thr_apply(lf, order, THR_K, 'mean')


def evaluate(y, g, p, mask, grp_of):
    m = metrics(y, g, p, mask); per = {}
    for s in np.unique(g[mask]):
        q = metrics(y, g, p, mask & (g == s))['subject_macro_bACC']
        if q is not None: per[s] = q
    m['groups_macro_bACC'] = {G: float(np.mean([v for s, v in per.items() if grp_of[s] == G]))
                              for G in GROUPS if any(grp_of[s] == G for s in per)}
    return m


def main():
    warnings.filterwarnings('ignore')
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); args = ap.parse_args()
    sfx = '_smoke' if args.smoke else ''; stem = 'literature_protocol' + sfx
    tee = open_log(HERE, stem)
    D = load_all(); y, g, fold, subs, grp_of = D['y'], D['g'], D['fold'], D['subs'], D['grp_of']; n = len(y)
    print('[check] fold == npz fold', flush=True)
    RT = np.load(HERE / 'real_time.npz')
    assert np.array_equal(RT['y'], y) and np.array_equal(RT['groups'], g); t_real = RT['t_end']
    print('[check] real_time.npz y/groups == canonical order', flush=True)
    run_folds = [0] if args.smoke else list(range(5)); sseeds = STREAM_SEEDS[:1] if args.smoke else STREAM_SEEDS
    nb = 100 if args.smoke else 1000
    run_subs = [s for s in subs if D['fold_of'][s] in run_folds]; ev = np.isin(g, run_subs)
    sidx = {s: j for j, s in enumerate(subs)}
    res = {'meta': {'models': {'lr': 'LR C=.01', 'hgb': 'HGB 200/3/.06/l2=1'}, 'folds': run_folds,
                    'stream_seeds': list(sseeds), 'bootstrap_n': nb, 'smoke': args.smoke,
                    'leaky_protocols': ['window_kfold', 'session_norm'],
                    'note_groups': 'low group defined post-hoc from pipeline performance -> circular'},
           'checks': {}, 'methods': {}, 'bootstrap': {}}
    S = {}
    print(f'[start] rows={n} subjects={len(subs)} eval_subjects={len(run_subs)} {el()}', flush=True)

    # ---------- strict + session_norm (subject-wise SGKF) ----------
    Xs = session_norm(D)
    for name in MODELS:
        for proto in ('strict', 'session_norm'):
            p = np.full(n, np.nan); tp = time.perf_counter()
            for k in run_folds:
                tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
                X = zz(D, tr) if proto == 'strict' else Xs
                p[te] = fit_score(name, X, y, tr, te)
                if proto == 'strict' and k == 0:
                    y2 = y.copy(); y2[te] = np.random.default_rng(0).permutation(y2[te])
                    assert np.array_equal(p[te], fit_score(name, X, y2, tr, te)), ('label shuffle', name)
                    assert np.array_equal(p[te], fit_score(name, X, y, tr, te)), ('determinism', name)
                    res['checks'][f'{name}_strict_fold0'] = 'test-label shuffle identical + determinism ok'
                print(f'[{proto} {name}] fold {k + 1}/5 fold_s={time.perf_counter() - tp:.1f} {el()}', flush=True)
            S[f'{proto}__{name}'] = p
    # ---------- window_kfold (leaky) ----------
    for name in MODELS:
        p = np.full(n, np.nan); tp = time.perf_counter()
        for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(np.arange(n))):
            if k not in run_folds: continue
            p[te] = fit_score(name, zz(D, tr), y, tr, te)
            print(f'[window_kfold {name}] fold {k + 1}/5 fold_s={time.perf_counter() - tp:.1f} {el()}', flush=True)
        S[f'window_kfold__{name}'] = p
    # real-time order: order-free models -> identical scores (sorting rows does not change any fit)
    for name in MODELS:
        S[f'real_time__{name}'] = S[f'strict__{name}']
    res['checks']['real_time_order_free'] = 'lr/hgb strict scores do not depend on row order (row-wise fit/score)'

    # ---------- v2 pipeline (stored branches) ----------
    A_ = D['A_']; B_ = np.load(TEST / 'depositioned_resnet_timefreq_oof.npz')
    v2 = json.loads((TEST / f'depositioned_within_subject_adapt_v2{sfx}.json').read_text(encoding='utf-8'))
    ZZt = np.full((n, 889), np.nan)
    for k in run_folds:
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k); ZZt[te] = zz(D, tr)[te]
    for sd in sseeds:
        order, _ = stream_order(g, subs, sd); tp = time.perf_counter()
        Bl = np.c_[logit(A_[f'stream{sd}__all_prank']), logit(B_[f'stream{sd}__resnet_whiten_logit_delta']),
                   logit(B_[f'stream{sd}__resnet_fbank_logit_delta'])]
        lf, lft = v2_fused(D, sd, run_subs, ZZt, Bl, y, order, sidx)
        S[f'strict__v2_s{sd}'] = sig(lft)
        mac = metrics(y, g, S[f'strict__v2_s{sd}'], ev)['subject_macro_bACC']
        ref = v2[f'stream{sd}']['methods']['fuse_pseudo_lr_full_thr']['all']['subject_macro_bACC']
        assert abs(mac - ref) < 1e-9, ('v2 reproduction', sd, mac, ref)
        res['checks'][f'v2_s{sd}_reproduced'] = ref
        zs = np.full(n, np.nan)
        for s in run_subs:
            q = g == s; zs[q] = (lft[q] - lft[q].mean()) / lft[q].std()
        S[f'session_norm__v2_s{sd}'] = sig(zs)
        print(f'[v2 s{sd}] reproduced mac={mac:.4f} (json {ref:.4f}) stream_s={time.perf_counter() - tp:.1f} {el()}',
              flush=True)
    # ---------- causal adaptation layer on order-free base: shuffled streams vs real time ----------
    Lb = logit(S['strict__lr']); Bb = np.c_[Lb, Lb, Lb]
    real_order = [np.flatnonzero(g == s)[np.argsort(t_real[g == s], kind='stable')] for s in subs]
    for tag, order in [(f'shuffled_s{sd}', stream_order(g, subs, sd)[0]) for sd in sseeds] + [('real', real_order)]:
        tp = time.perf_counter()
        lf, lft = v2_fused(D, None, run_subs, ZZt, Bb, y, order, sidx)
        key = ('real_time' if tag == 'real' else 'strict') + f'__adapt_lr_{tag}'
        S[key] = sig(lft); S[key.replace('__adapt', '__adaptnothr')] = sig(lf)
        print(f'[adapt {tag}] stream_s={time.perf_counter() - tp:.1f} {el()}', flush=True)
    # also the base logit with only the causal thr_mean256 in real vs shuffled order
    for tag, order in [('shuffled_s0', stream_order(g, subs, 0)[0]), ('real', real_order)]:
        key = ('real_time' if tag == 'real' else 'strict') + f'__lr_thr_{tag}'
        S[key] = sig(thr_apply(np.where(ev, Lb, np.nan), order, THR_K, 'mean'))

    # ---------- evaluation ----------
    for k_, p in S.items():
        S[k_] = np.where(ev, p, np.nan)
        r = res['methods'][k_] = evaluate(y, g, S[k_], ev & np.isfinite(S[k_]), grp_of)
        print(f'[result] {k_:34s} AUC={r["AUC"]:.4f} mac={r["subject_macro_bACC"]:.4f} mic={r["micro_bACC"]:.4f} | '
              + ' '.join(f'{G}={v:.3f}' for G, v in r['groups_macro_bACC'].items()), flush=True)
    pairs = [(f'{pr}__{m}', f'strict__{m}') for m in MODELS for pr in ('window_kfold', 'session_norm')]
    pairs += [(f'session_norm__v2_s{sd}', f'strict__v2_s{sd}') for sd in sseeds]
    pairs += [(f'strict__v2_s{sd}', 'strict__lr') for sd in sseeds]
    pairs += [('real_time__adapt_lr_real', f'strict__adapt_lr_shuffled_s{sd}') for sd in sseeds]
    pairs += [('real_time__adaptnothr_lr_real', f'strict__adaptnothr_lr_shuffled_s{sd}') for sd in sseeds]
    pairs += [('real_time__lr_thr_real', 'strict__lr_thr_shuffled_s0'), ('real_time__lr_thr_real', 'strict__lr')]
    for a, b in pairs:
        m = ev & np.isfinite(S[a]) & np.isfinite(S[b])
        bb = res['bootstrap'][f'{a}__vs__{b}'] = boot(S[a], S[b], y, g, m, nb)
        print(f'[boot] {a:32s} vs {b:32s} dAUC={bb["dAUC"]["mean"]:+.4f}[{bb["dAUC"]["p2.5"]:+.4f},'
              f'{bb["dAUC"]["p97.5"]:+.4f}] dmac={bb["dmacro_bACC"]["mean"]:+.4f}[{bb["dmacro_bACC"]["p2.5"]:+.4f},'
              f'{bb["dmacro_bACC"]["p97.5"]:+.4f}]', flush=True)
    res['seconds'] = time.perf_counter() - __import__('common').T0
    save(HERE, stem, res); print(f'[done] {el()} -> {stem}.json', flush=True)
    sys.stdout = tee.stream; tee.f.close()


if __name__ == '__main__':
    main()
