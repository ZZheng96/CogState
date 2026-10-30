"""Direction B: causal test-time unlabeled self-training on top of fusionM_prank (de-positioned streams).

Protocol identical to depositioned_feature_multiscale.py (stream order = concat(relax,task) permuted by
default_rng([stream_seed, subject_index]); outer folds = stored fold map; z = StandardScaler(train fold) of median-imputed
858 features). Base scores (OOF, causal) reused from depositioned_feature_multiscale_oof.npz:
fusionM_prank = sigmoid(mean(logit hgb_prank, logit resnet_whiten_logit_delta)). No labels of test subjects are used.

Hyper-parameters (fixed before running, not tuned):
  MIN_PAST = 32       TTA only when the subject has >= 32 past stream windows; otherwise output = fusionM_prank
  REFIT    = 8        LR refit at t = 32, 40, 48, ... using only windows [0, t); windows t..t+7 use that model
  FRAC     = 0.30     pseudo-labels: top floor(.3 t) past windows by source score -> 1, bottom floor(.3 t) -> 0
                      (stable argsort for ties)
  LR       = LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000, lbfgs), no extra scaling
  CLIP     = 4        LR logit (decision_function) clipped to +-4
  PCA      = PCA(32, svd_solver='randomized', random_state=0) fit on training-fold rows only
Variants:
  tta_pseudo       source = fusionM_prank; X = PCA(z);
                   logit = .5 logit(fusionM_prank) + .5 clip(LR)
  tta_pseudo_full  source = fusionM_prank; X = PCA(hgb_prank features [z, z-h32, prank128], 2574 dims, unscaled);
                   logit = .5 logit(fusionM_prank) + .5 clip(LR)
  tta_cotrain      source = hgb_prank; X = PCA(z);
                   logit = .5 (.5 logit(hgb_prank) + .5 clip(LR)) + .5 logit(resnet_whiten_logit_delta)
Threshold .5. Paired subject bootstrap 1000 vs fusionM_prank, segments all and mature (t >= 120).
Checks (assert): stored position-only AUC ~.5; prefix invariance (future features+scores changed -> past outputs
identical, future changed); subject isolation (per-subject run == interleaved multi-subject stepwise run).
"""
import argparse, json, sys, time, warnings
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

warnings.filterwarnings('ignore', category=ConvergenceWarning)

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from depositioned_history_delta import boot, load_db  # noqa: E402
from depositioned_resnet_history_delta import stream_order, logit, sig  # noqa: E402
from depositioned_feature_multiscale import feat_subject, feat_all  # noqa: E402
from causal_kernel_state_mapping import metrics  # noqa: E402
from switch_dual_memory import Tee  # noqa: E402

MIN_PAST, REFIT, FRAC, C_LR, CLIP, NPC, MATURE = 32, 8, .30, .1, 4., 32, 120
VARIANTS = ('tta_pseudo', 'tta_pseudo_full', 'tta_cotrain')
STREAM_SEEDS = (0, 1, 2)
T0 = time.perf_counter()


def el():
    return f'elapsed={time.perf_counter() - T0:.1f}s'


def fit_lr(X, s):
    k = int(np.floor(FRAC * len(s))); idx = np.argsort(s, kind='stable')
    sel = np.concatenate([idx[:k], idx[-k:]]); lab = np.r_[np.zeros(k), np.ones(k)]
    return LogisticRegression(C=C_LR, class_weight='balanced', max_iter=2000).fit(X[sel], lab)


def lr_track(X, s):
    """X, s: one subject in stream order. out[t] = LR logit using model fit on rows [0, r), r = last refit <= t."""
    n = len(s); out = np.full(n, np.nan)
    for r in range(MIN_PAST, n, REFIT):
        m = fit_lr(X[:r], s[:r]); out[r:r + REFIT] = m.decision_function(X[r:r + REFIT])
    return out


def lr_track_interleaved(Xs, ss):
    """Naive stepwise reference: several subjects' streams interleaved round-robin, per-subject buffers."""
    buf = [[] for _ in Xs]; model = [None] * len(Xs); out = [np.full(len(s), np.nan) for s in ss]
    for t in range(max(len(s) for s in ss)):
        for j in range(len(Xs)):
            if t >= len(ss[j]): continue
            nb = len(buf[j])
            if nb >= MIN_PAST and (nb - MIN_PAST) % REFIT == 0:
                P = np.array([b[0] for b in buf[j]]); S = np.array([b[1] for b in buf[j]])
                model[j] = fit_lr(P, S)
            if nb >= MIN_PAST: out[j][t] = model[j].decision_function(Xs[j][t:t + 1])[0]
            buf[j].append((Xs[j][t], ss[j][t]))
    return out


def combine(v, lr, lf, lh, lw):
    c = np.clip(lr, -CLIP, CLIP)
    if v == 'tta_cotrain': new = .5 * (.5 * lh + .5 * c) + .5 * lw
    else: new = .5 * lf + .5 * c
    return sig(np.where(np.isfinite(lr), new, lf))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); args = ap.parse_args()
    stem = 'depositioned_tta_selftrain' + ('_smoke' if args.smoke else '')
    tee = Tee(sys.stdout, OUT / (stem + '.log')); sys.stdout = tee
    sseeds = STREAM_SEEDS[:1] if args.smoke else STREAM_SEEDS
    nb = 100 if args.smoke else 1000
    man, subs, data = load_db()
    Fc = np.vstack([data[s][0] for s in subs]); y = np.concatenate([data[s][1] for s in subs])
    g = np.concatenate([np.full(len(data[s][1]), s) for s in subs])
    prior = np.load(OUT / 'depositioned_feature_multiscale_oof.npz')
    assert np.array_equal(prior['y'], y) and np.array_equal(prior['groups'], g)
    fold = prior['fold']; sidx = {s: j for j, s in enumerate(subs)}
    run_folds = [0] if args.smoke else list(range(5))
    print(f'[start] rows={len(y)} subjects={len(subs)} smoke={args.smoke} sseeds={sseeds} variants={VARIANTS}', flush=True)
    result = {'meta': {'min_past': MIN_PAST, 'refit': REFIT, 'frac': FRAC, 'C': C_LR, 'clip': CLIP, 'n_pca': NPC,
                       'mature_index': MATURE, 'threshold': .5, 'stream_seeds': list(sseeds), 'smoke': args.smoke,
                       'base_source': 'depositioned_feature_multiscale_oof.npz'}, 'checks': {}}
    checks = result['checks']; oof = {'y': y, 'groups': g, 'fold': fold}
    # z and PCA(z) depend only on fold (not stream order)
    with threadpool_limits(limits=6):
        Zf, PZ = {}, {}
        for k in run_folds:
            tr = np.flatnonzero(fold != k)
            med = np.nanmedian(np.where(np.isfinite(Fc[tr]), Fc[tr], np.nan), 0); med = np.where(np.isfinite(med), med, 0.)
            Zi = np.where(np.isfinite(Fc), Fc, med)
            Zf[k] = StandardScaler().fit(Zi[tr]).transform(Zi)
            pz = PCA(NPC, svd_solver='randomized', random_state=0).fit(Zf[k][tr]); PZ[k] = (pz, pz.transform(Zf[k]))
            print(f'[fold {k + 1}/5] z + PCA(z) ready evr={pz.explained_variance_ratio_.sum():.3f} {el()}', flush=True)
        for sd in sseeds:
            order, t = stream_order(g, subs, sd)
            fM, hp, W = (prior[f'stream{sd}__{n}'] for n in ('fusionM_prank', 'hgb_prank', 'resnet_whiten_logit_delta'))
            assert np.abs(fM - sig((logit(hp) + logit(W)) / 2)).max() < 1e-9
            lf, lh, lw = logit(fM), logit(hp), logit(W)
            S = {v: np.full(len(y), np.nan) for v in VARIANTS}; LR = {v: np.full(len(y), np.nan) for v in VARIANTS}
            for k in run_folds:
                tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
                Z = Zf[k]; pz, XZ = PZ[k]
                t1 = time.perf_counter(); F = feat_all(Z, order, 'hgb_prank')
                pf = PCA(NPC, svd_solver='randomized', random_state=0).fit(F[tr]); XF = pf.transform(F)
                print(f'[s{sd} fold {k + 1}/5] prank feats {F.shape[1]} + PCA evr={pf.explained_variance_ratio_.sum():.3f} '
                      f'{time.perf_counter() - t1:.1f}s {el()}', flush=True)
                del F
                te_subs = list(np.unique(g[te]))
                spec = {'tta_pseudo': (XZ, fM), 'tta_pseudo_full': (XF, fM), 'tta_cotrain': (XZ, hp)}
                for v in VARIANTS:
                    X, src = spec[v]; t1 = time.perf_counter()
                    for i, s0 in enumerate(te_subs):
                        loc = order[sidx[s0]]; LR[v][loc] = lr_track(X[loc], src[loc])
                    S[v][te] = combine(v, LR[v][te], lf[te], lh[te], lw[te])
                    # subject isolation: per-subject run == interleaved stepwise run of 3 test subjects
                    locs = [order[sidx[s0]] for s0 in te_subs[:3]]
                    inter = lr_track_interleaved([X[l] for l in locs], [src[l] for l in locs])
                    di = max(float(np.nanmax(np.abs(a - LR[v][l]))) for a, l in zip(inter, locs))
                    assert all(np.array_equal(np.isnan(a), np.isnan(LR[v][l])) for a, l in zip(inter, locs)) and di < 1e-9, di
                    # prefix invariance: change future features (raw z +100 then rebuilt) and future source scores
                    rng = np.random.default_rng(k)
                    for s0 in te_subs[:2]:
                        loc = order[sidx[s0]]; cut = len(loc) // 2; alt = Z[loc].copy(); alt[cut:] += 100.
                        Xa = pf.transform(feat_subject(alt, 'hgb_prank')) if v == 'tta_pseudo_full' else pz.transform(alt)
                        assert np.allclose(Xa[:cut], X[loc][:cut], rtol=1e-9, atol=1e-9)  # BLAS-level noise only
                        Xa = np.vstack([X[loc][:cut], Xa[cut:]])
                        sa = src[loc].copy(); sa[cut:] = rng.random(len(loc) - cut)
                        la = lr_track(Xa, sa)
                        d_pre = float(np.nanmax(np.abs(la[:cut] - LR[v][loc][:cut]), initial=0.))
                        d_fut = float(np.nanmax(np.abs(la[cut:] - LR[v][loc][cut:])))
                        assert np.array_equal(np.isnan(la[:cut]), np.isnan(LR[v][loc][:cut])) and d_pre == 0. and d_fut > 1e-6
                        checks[f'causal_s{sd}_f{k}_{v}_{s0}'] = {'prefix_after_future_change': d_pre, 'future_changed': d_fut}
                    checks[f'isolation_s{sd}_f{k}_{v}'] = di
                    print(f'[s{sd} fold {k + 1}/5] {v:16s} subjects={len(te_subs)} fit_s={time.perf_counter() - t1:.1f} '
                          f'checks ok (isolation {di:.1e}, prefix) {el()}', flush=True)
            ev = np.isfinite(S[VARIANTS[0]])
            M = {'all': ev, 'mature': ev & (t >= MATURE)}
            R = result[f'stream{sd}'] = {'methods': {}, 'bootstrap': {}, 'subjects': {}, 'sanity': {}}
            for c in ('lr', 'hgb'):
                a = metrics(y, g, prior[f'stream{sd}__position_only__{c}'], ev)['AUC']
                R['sanity'][f'position_only__{c}'] = a; print(f'[sanity s{sd}] position_only__{c} AUC={a:.4f}', flush=True)
                if not args.smoke: assert abs(a - .5) < .03
            a = float(roc_auc_score(y[ev], t[ev])); R['sanity']['raw_position_AUC'] = a
            print(f'[sanity s{sd}] raw t AUC={a:.4f}', flush=True); assert abs(a - .5) < .03
            for v in VARIANTS:  # pseudo-label quality diagnostic (labels used only for reporting)
                m_ = ev & np.isfinite(LR[v]); R['sanity'][f'{v}__lr_logit_AUC'] = float(roc_auc_score(y[m_], LR[v][m_]))
            names = {'fusionM_prank': fM, 'hgb_prank': hp, 'resnet_whiten_logit_delta': W, **S}
            for n, p in names.items():
                p = np.where(ev, p, np.nan); oof[f'stream{sd}__{n}'] = p
                r = R['methods'][n] = {s: metrics(y, g, p, M[s]) for s in M}
                ba = np.array([q for s in np.unique(g[ev]) if (q := metrics(y, g, p, ev & (g == s))['subject_macro_bACC']) is not None])
                R['subjects'][n] = {'n_ge_0.9': int((ba >= .9).sum()), 'max': float(ba.max()), 'n': int(len(ba))}
                print(f'[result s{sd}] {n:26s} ' + ' | '.join(
                    f'{s}: AUC={r[s]["AUC"]:.4f} mac={r[s]["subject_macro_bACC"]:.4f} mic={r[s]["micro_bACC"]:.4f}' for s in M) +
                    f' | subj>=.9={R["subjects"][n]["n_ge_0.9"]} max={R["subjects"][n]["max"]:.3f}', flush=True)
            for v in VARIANTS:
                for seg in M:
                    bb = R['bootstrap'][f'{v}__vs__fusionM_prank__{seg}'] = boot(S[v], fM, y, g, M[seg], nb)
                    print(f'[boot s{sd}] {v:16s} vs fusionM_prank {seg:6s} '
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
