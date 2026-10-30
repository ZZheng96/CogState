"""Strong 858-dim features: current window minus subject's own causal past unlabeled mean.

Pre-registered variants (fixed before running):
  current_only = [z]; delta_k8 = [z, z-hist_8]; delta_k32 = [z, z-hist_32];
  delta_all = [z, z-hist_all(P=32)]; delta_multi = [z, z-hist_8, z-hist_all].
  hist_k(t) = (sum z[max(0,t-k):t]) / (n_past + P), P=8 (prior mean 0 after train standardization); current row excluded.
Classifiers: HGB(200, depth3, lr .06, l2 1, rs0) and LR(C=.05). Threshold .5 (+ inner-fold threshold).
Input features: highdim_deep_features.npz X only (no OOF scores).
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from causal_kernel_state_mapping import metrics  # noqa: E402
from switch_dual_memory import Tee, masks, boot  # noqa: E402

P_K, P_ALL = 8., 32.
VARIANTS = ('current_only', 'delta_k8', 'delta_k32', 'delta_all', 'delta_multi')
SEGS = ('all', 'switching', 'steady', 'history_comparable')


def make_clf(name):
    if name == 'hgb':
        return HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=.06,
                                              l2_regularization=1., random_state=0)
    return LogisticRegression(C=.05, max_iter=2000)


def hist_subject(z):
    """Row t uses only z[:t]. Exact stepwise: windowed sums recomputed from the slice; expanding sum accumulated in order."""
    n, d = z.shape
    h8, h32, ha = np.zeros((n, d)), np.zeros((n, d)), np.zeros((n, d))
    run = np.zeros(d)
    for t in range(n):
        lo8, lo32 = max(0, t - 8), max(0, t - 32)
        h8[t] = z[lo8:t].sum(0) / (t - lo8 + P_K)
        h32[t] = z[lo32:t].sum(0) / (t - lo32 + P_K)
        ha[t] = run / (t + P_ALL)
        run = run + z[t]
    return h8, h32, ha


def hist_ref(z, t, k, P):
    """Naive loop reference for a single row."""
    lo = 0 if k is None else max(0, t - k)
    s = np.zeros(z.shape[1])
    for j in range(lo, t):
        s = s + z[j]
    return s / (t - lo + P)


def build_subject(z, v, H=None):
    if v == 'current_only':
        return z.copy()
    h8, h32, ha = H if H is not None else hist_subject(z)
    return {'delta_k8': lambda: np.hstack([z, z - h8]), 'delta_k32': lambda: np.hstack([z, z - h32]),
            'delta_all': lambda: np.hstack([z, z - ha]),
            'delta_multi': lambda: np.hstack([z, z - h8, z - ha])}[v]()


def build_all(z, g, rows):
    """rows: global indices (time-ordered within subject). Returns dict variant -> feature matrix aligned with rows."""
    out = {v: None for v in VARIANTS}
    d = z.shape[1]; dims = {'current_only': 1, 'delta_k8': 2, 'delta_k32': 2, 'delta_all': 2, 'delta_multi': 3}
    for v in VARIANTS: out[v] = np.zeros((len(rows), d * dims[v]))
    gr = g[rows]
    for s in np.unique(gr):
        loc = np.flatnonzero(gr == s); zs = z[loc]; H = hist_subject(zs)
        for v in VARIANTS: out[v][loc] = build_subject(zs, v, H)
    return out


def macro_bacc(y, g, p, thr):
    from sklearn.metrics import balanced_accuracy_score
    per = [balanced_accuracy_score(y[g == s], p[g == s] >= thr) for s in np.unique(g) if np.unique(y[g == s]).size == 2]
    return float(np.mean(per))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--pca', type=int, default=0, help='0 = raw 858 dims; >0 = PCA dims fit on train before history')
    ap.add_argument('--no-inner', action='store_true')
    args = ap.parse_args()
    stem = 'strong_feature_history_delta' + (f'_pca{args.pca}' if args.pca else '') + ('_smoke' if args.smoke else '')
    tee = Tee(sys.stdout, OUT / (stem + '.log')); sys.stdout = tee
    start = time.perf_counter()
    d = np.load(OUT / 'highdim_deep_features.npz'); y, g, X = d['y'].astype(int), d['groups'], d['X'].astype(float)
    M = masks(y, g)
    mb = np.load(OUT / 'merge_best_window6_oof.npz')
    assert np.array_equal(mb['y'], y) and np.array_equal(mb['groups'], g)
    base = expit(mb['combo__C4_bb_resnet_eq'])
    print(f'[start] rows={len(y)} subjects={len(np.unique(g))} dim={X.shape[1]} pca={args.pca} smoke={args.smoke}', flush=True)
    keys = [f'{v}__{c}' for v in VARIANTS for c in ('hgb', 'lr')]
    scores = {k: np.full(len(y), np.nan) for k in keys}
    thr_scores = {k: np.full(len(y), np.nan) for k in keys}
    result = {'meta': {'P_k': P_K, 'P_all': P_ALL, 'pca': args.pca, 'variants': VARIANTS, 'threshold': .5,
                       'inner_threshold': not args.no_inner, 'baseline': 'merge_best_window6_oof combo__C4_bb_resnet_eq'},
              'folds': [], 'methods': {}, 'bootstrap': {}}
    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, g))
    with threadpool_limits(limits=4):
        for fold, (tr, te) in enumerate(splits):
            assert not set(g[tr]) & set(g[te])
            if args.smoke:
                tr = tr[np.isin(g[tr], np.unique(g[tr])[:5])]; te = te[np.isin(g[te], np.unique(g[te])[:1])]
            for ii in (tr, te):  # time order within subject preserved (indices sorted, rows ordered in time)
                assert np.all(np.diff(ii) > 0)
            med = np.nanmedian(np.where(np.isfinite(X[tr]), X[tr], np.nan), 0); med = np.where(np.isfinite(med), med, 0.)
            Xf = np.where(np.isfinite(X), X, med[None, :])
            sc = StandardScaler().fit(Xf[tr]); Z = sc.transform(Xf)
            if args.pca:
                pca = PCA(args.pca, random_state=0).fit(Z[tr]); Z = pca.transform(Z)
                Z = Z / Z[tr].std(0)  # keep train-standardized (prior mean 0 holds: PCA centered on train)
            t0 = time.perf_counter()
            Ftr, Fte = build_all(Z[tr], g[tr], np.arange(len(tr))), build_all(Z[te], g[te], np.arange(len(te)))
            # ---- causal checks on test subjects
            zte, gte = Z[te], g[te]
            for s in np.unique(gte)[:3]:
                loc = np.flatnonzero(gte == s); zs = zte[loc]; H = hist_subject(zs)
                for t in (0, 1, 5, 40, len(zs) - 1):
                    assert np.allclose(H[0][t], hist_ref(zs, t, 8, P_K), atol=1e-12)
                    assert np.allclose(H[1][t], hist_ref(zs, t, 32, P_K), atol=1e-12)
                    assert np.allclose(H[2][t], hist_ref(zs, t, None, P_ALL), atol=1e-10)
            for s in np.unique(gte):
                loc = np.flatnonzero(gte == s); zs = zte[loc]; cut = len(zs) // 2
                alt = zs.copy(); alt[cut:] += 100.
                Ha, Hp = hist_subject(alt), hist_subject(zs[:cut])
                for v in VARIANTS:
                    full = Fte[v][loc]
                    assert np.array_equal(build_subject(alt, v, Ha)[:cut], full[:cut])
                    assert np.array_equal(build_subject(zs[:cut], v, Hp), full[:cut])
            print(f'[fold {fold+1}/5] built features + checks ok ntr={len(tr)} nte={len(te)} '
                  f'build_s={time.perf_counter()-t0:.1f} elapsed={time.perf_counter()-start:.1f}s', flush=True)
            info = {'fold': fold, 'test_subjects': [str(s) for s in np.unique(gte)], 'thresholds': {}}
            # inner split on train subjects for threshold
            if not args.no_inner:
                itr, iva = next(StratifiedGroupKFold(4, shuffle=True, random_state=fold).split(tr, y[tr], g[tr]))
                assert not set(g[tr][itr]) & set(g[tr][iva])
            for v in VARIANTS:
                for c in ('hgb', 'lr'):
                    t1 = time.perf_counter(); k = f'{v}__{c}'
                    m = make_clf(c).fit(Ftr[v], y[tr]); p = m.predict_proba(Fte[v])[:, 1]
                    scores[k][te] = p
                    if not args.no_inner:
                        pi = make_clf(c).fit(Ftr[v][itr], y[tr][itr]).predict_proba(Ftr[v][iva])[:, 1]
                        grid = np.linspace(.05, .95, 91)
                        accs = [macro_bacc(y[tr][iva], g[tr][iva], pi, th) for th in grid]
                        thr = float(grid[int(np.argmax(accs))]); info['thresholds'][k] = thr
                        pc = np.clip(p, 1e-12, 1 - 1e-12)  # monotone shift: (p>=thr) <=> (shifted>=.5)
                        thr_scores[k][te] = expit(np.log(pc / (1 - pc)) - np.log(thr / (1 - thr)))
                    print(f'[fold {fold+1}/5] {k:22s} dim={Ftr[v].shape[1]} fit_s={time.perf_counter()-t1:.1f} '
                          f'thr={info["thresholds"].get(k, .5):.2f} elapsed={time.perf_counter()-start:.1f}s', flush=True)
            result['folds'].append(info)
            if args.smoke: break
    ev = np.isfinite(scores[keys[0]])
    allsc = {'baseline_merge_best': base, **scores}
    if not args.no_inner:
        allsc.update({f'{k}__innerthr': v for k, v in thr_scores.items()})
    for name, v in allsc.items():
        r = result['methods'][name] = {s: metrics(y, g, v, ev & M[s]) for s in SEGS}
        print(f'[result] {name:34s} ' + ' | '.join(
            f'{s}: AUC={r[s]["AUC"] or float("nan"):.4f} mac={r[s]["subject_macro_bACC"] or float("nan"):.4f} mic={r[s]["micro_bACC"]:.4f}'
            for s in SEGS), flush=True)
    nb = 100 if args.smoke else 1000
    for k in keys:
        for ref in ('baseline_merge_best', f'current_only__{k.split("__")[1]}'):
            if ref == k: continue
            for seg in ('all', 'history_comparable'):
                mk = ev & M[seg]
                if np.unique(g[mk]).size < 2: continue
                key = f'{k}_vs_{ref}__{seg}'; bb = result['bootstrap'][key] = boot(allsc[k], allsc[ref], y, g, mk, nb)
                print(f'[boot] {key:70s} dAUC={bb["dAUC"]["mean"]:+.4f}[{bb["dAUC"]["p2.5"]:+.4f},{bb["dAUC"]["p97.5"]:+.4f}] '
                      f'dmac={bb["dmacro_bACC"]["mean"]:+.4f}[{bb["dmacro_bACC"]["p2.5"]:+.4f},{bb["dmacro_bACC"]["p97.5"]:+.4f}] '
                      f'elapsed={time.perf_counter()-start:.1f}s', flush=True)
    # per-subject on history_comparable for best model (selected by macro bACC on that segment; reporting only)
    best = max(keys, key=lambda k: result['methods'][k]['history_comparable']['subject_macro_bACC'] or -1)
    per = {}
    for nm in (best, 'baseline_merge_best'):
        mk = ev & M['history_comparable']
        per[nm] = {str(s): metrics(y, g, allsc[nm], mk & (g == s))['subject_macro_bACC'] for s in np.unique(g[mk])}
    result['per_subject_history_comparable'] = per; result['best_history_comparable'] = best
    for nm, dct in per.items():
        vals = np.array([v for v in dct.values() if v is not None])
        print(f'[subject] {nm}: n_subj={len(vals)} >=0.9: {(vals >= .9).sum()} mean={vals.mean():.4f} '
              + ' '.join(f'{s}:{v:.3f}' for s, v in sorted(dct.items()) if v is not None), flush=True)
    result['seconds'] = time.perf_counter() - start
    np.savez_compressed(OUT / (stem + '_oof.npz'), y=y, groups=g, eval_mask=ev, baseline=base, **scores,
                        **{f'{k}__innerthr': v for k, v in thr_scores.items()}, **{f'mask_{k}': v for k, v in M.items()})
    (OUT / (stem + '_results.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[done] elapsed={result["seconds"]:.1f}s', flush=True)
    sys.stdout = tee.stream; tee.f.close()


if __name__ == '__main__':
    main()
