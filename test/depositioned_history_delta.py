"""De-positioned causal unlabeled history delta on database/ (no timestamps; random per-subject stream order).

Stream order per subject: concat(relax, task) then np.random.default_rng(seed).permutation (seed 0 main; 1, 2 robustness),
so order carries no label / schedule / position information.
Variants (fixed): current_only [z]; delta_k32 [z, z-mean(past 32, no prior; 0 if no past)];
  delta_all [z, z-sum(past)/(t+32)]; delta_multi [z, z-hist_8 (no prior), z-hist_all].
Controls: position_only [t]; position_plus_current [z, t].
Classifiers: HGB(200, depth3, lr .06, l2 1, rs0), LR(C=.05). Threshold .5.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
DB = OUT.parent / 'database'
sys.path.insert(0, str(OUT))
from causal_kernel_state_mapping import metrics  # noqa: E402
from switch_dual_memory import Tee  # noqa: E402

P_ALL, MATURE = 32., 120
DELTAS = ('delta_k32', 'delta_all', 'delta_multi')
VARIANTS = ('current_only',) + DELTAS + ('position_only', 'position_plus_current')
SEGS = ('all', 'mature')


def make_clf(name):
    if name == 'hgb':
        return HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=.06,
                                              l2_regularization=1., random_state=0)
    return LogisticRegression(C=.05, max_iter=2000)


def load_db():
    man = json.loads((DB / 'manifest.json').read_text(encoding='utf-8'))
    subs = sorted(p.stem for p in (DB / 'windows' / 'relax').glob('*.npz'))
    assert subs == sorted(p.stem for p in (DB / 'windows' / 'task').glob('*.npz'))
    data = {}
    for s in subs:
        parts = [np.load(DB / 'windows' / k / f'{s}.npz') for k in ('relax', 'task')]
        f = np.vstack([p['f'] for p in parts]).astype(float)
        lab = np.concatenate([p['label'] for p in parts]).astype(int)
        assert set(np.unique(parts[0]['label'])) == {0} and set(np.unique(parts[1]['label'])) == {1}
        sid = np.unique(np.concatenate([p['subject'] for p in parts])); assert sid.size == 1
        data[s] = (f, lab)
    return man, subs, data


def arrange(subs, data, seed, mode):
    """Per-subject fixed random permutation. shared: default_rng(seed) for every subject (label sequences correlate
    across subjects -> position leaks); independent: default_rng([seed, subject_index])."""
    X, y, g, t = [], [], [], []
    for j, s in enumerate(subs):
        f, lab = data[s]
        rng = np.random.default_rng(seed if mode == 'shared' else [seed, j]); perm = rng.permutation(len(lab))
        X.append(f[perm]); y.append(lab[perm]); g.append(np.full(len(lab), s)); t.append(np.arange(len(lab)))
    return np.vstack(X), np.concatenate(y), np.concatenate(g), np.concatenate(t)


def hist_subject(z):
    """Row t uses only z[:t]. Exact stepwise: windowed means from slices; expanding sum accumulated in order."""
    n, d = z.shape
    h8, h32, ha = np.zeros((n, d)), np.zeros((n, d)), np.zeros((n, d))
    run = np.zeros(d)
    for t in range(n):
        if t > 0:
            h8[t] = z[max(0, t - 8):t].mean(0); h32[t] = z[max(0, t - 32):t].mean(0)
        ha[t] = run / (t + P_ALL)
        run = run + z[t]
    return h8, h32, ha


def hist_ref(z, t, k, P):
    lo = 0 if k is None else max(0, t - k)
    if t == lo: return np.zeros(z.shape[1])
    s = np.zeros(z.shape[1])
    for j in range(lo, t): s = s + z[j]
    return s / ((t - lo) + P)


def build_subject(z, ts, v, H=None):
    if v == 'current_only': return z.copy()
    if v == 'position_only': return ts[:, None].astype(float)
    if v == 'position_plus_current': return np.hstack([z, ts[:, None]])
    h8, h32, ha = H if H is not None else hist_subject(z)
    return {'delta_k32': lambda: np.hstack([z, z - h32]), 'delta_all': lambda: np.hstack([z, z - ha]),
            'delta_multi': lambda: np.hstack([z, z - h8, z - ha])}[v]()


def build_all(z, g, ts, variants):
    gr = np.asarray(g); out = {}
    blocks = {v: [] for v in variants}; order = []
    for s in np.unique(gr):
        loc = np.flatnonzero(gr == s); assert np.array_equal(ts[loc], np.arange(len(loc)))
        H = hist_subject(z[loc]) if any(v in DELTAS for v in variants) else None
        for v in variants: blocks[v].append(build_subject(z[loc], ts[loc], v, H))
        order.append(loc)
    order = np.concatenate(order)
    for v in variants:
        F = np.vstack(blocks[v]); out[v] = np.empty_like(F); out[v][order] = F
    return out


# ---------------- vectorized paired subject bootstrap ----------------
def subj_stats(y, g, p, mask, subs):
    """Per-subject confusion counts at .5 + per-subject bACC (nan if single class)."""
    c = np.zeros((len(subs), 4))  # TP FN TN FP
    for j, s in enumerate(subs):
        q = mask & (g == s); yy, pp = y[q], p[q] >= .5
        c[j] = [(yy & pp).sum(), (yy & ~pp).sum(), (~yy & ~pp).sum(), (~yy & pp).sum()]
    with np.errstate(invalid='ignore', divide='ignore'):
        ba = .5 * (c[:, 0] / (c[:, 0] + c[:, 1]) + c[:, 2] / (c[:, 2] + c[:, 3]))
    return c, ba


def auc_mats(y, g, p, mask, subs):
    """Per-subject positive/negative counts per unique score value."""
    m = mask; u, inv = np.unique(p[m], return_inverse=True)
    sj = np.searchsorted(subs, g[m]); yy = y[m].astype(bool)
    A_pos = np.zeros((len(subs), len(u))); A_neg = np.zeros((len(subs), len(u)))
    np.add.at(A_pos, (sj[yy], inv[yy]), 1.); np.add.at(A_neg, (sj[~yy], inv[~yy]), 1.)
    return A_pos, A_neg


def weighted_auc(W, A_pos, A_neg):
    pos, neg = W @ A_pos, W @ A_neg
    below = np.cumsum(neg, 1) - neg
    return (pos * (below + .5 * neg)).sum(1) / (pos.sum(1) * neg.sum(1))


def boot(a, b, y, g, mask, n=1000):
    yb = y.astype(bool); subs = np.unique(g[mask])
    W = np.zeros((n, len(subs))); rng = np.random.default_rng(0)
    for i in range(n): W[i] = np.bincount(rng.integers(0, len(subs), len(subs)), minlength=len(subs))
    res = {}
    out = []
    for p in (a, b):
        c, ba = subj_stats(yb, g, p, mask, subs); ok = np.isfinite(ba)
        auc = weighted_auc(W, *auc_mats(yb, g, p, mask, subs))
        mac = (W[:, ok] @ ba[ok]) / W[:, ok].sum(1)
        C = W @ c; mic = .5 * (C[:, 0] / (C[:, 0] + C[:, 1]) + C[:, 2] / (C[:, 2] + C[:, 3]))
        out.append(np.c_[auc, mac, mic])
    d = out[0] - out[1]
    for j, k in enumerate(('dAUC', 'dmacro_bACC', 'dmicro_bACC')):
        res[k] = {'mean': float(d[:, j].mean()), 'p2.5': float(np.percentile(d[:, j], 2.5)),
                  'p97.5': float(np.percentile(d[:, j], 97.5))}
    return res


def run_seed(seed, subs, data, variants, clfs, args, start, result):
    X, y, g, ts = arrange(subs, data, seed, args.seed_mode)
    rho = spearmanr(ts, y).correlation
    print(f'[seed {seed}] rows={len(y)} subjects={len(np.unique(g))} dim={X.shape[1]} '
          f'spearman(t,y)={rho:+.4f} variants={variants} clfs={clfs}', flush=True)
    keys = [f'{v}__{c}' for v in variants for c in clfs]
    scores = {k: np.full(len(y), np.nan) for k in keys}
    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, g))
    folds = []
    with threadpool_limits(limits=4):
        for fold, (tr, te) in enumerate(splits):
            assert not set(g[tr]) & set(g[te])
            if args.smoke:
                tr = tr[np.isin(g[tr], np.unique(g[tr])[:5])]; te = te[np.isin(g[te], np.unique(g[te])[:1])]
            med = np.nanmedian(np.where(np.isfinite(X[tr]), X[tr], np.nan), 0); med = np.where(np.isfinite(med), med, 0.)
            Xf = np.where(np.isfinite(X), X, med[None, :])
            Z = StandardScaler().fit(Xf[tr]).transform(Xf)
            t0 = time.perf_counter()
            Ftr, Fte = build_all(Z[tr], g[tr], ts[tr], variants), build_all(Z[te], g[te], ts[te], variants)
            zte, gte, tte = Z[te], g[te], ts[te]
            # exact stepwise vs naive reference
            for s in np.unique(gte)[:3]:
                zs = zte[gte == s]; H = hist_subject(zs)
                for t in (0, 1, 5, 40, len(zs) - 1):
                    assert np.allclose(H[0][t], hist_ref(zs, t, 8, 0.), atol=1e-12)
                    assert np.allclose(H[1][t], hist_ref(zs, t, 32, 0.), atol=1e-12)
                    assert np.allclose(H[2][t], hist_ref(zs, t, None, P_ALL), atol=1e-10)
            # subject isolation (subject built alone == built inside pool) + prefix invariance
            for s in np.unique(gte):
                loc = np.flatnonzero(gte == s); zs, tss = zte[loc], tte[loc]; cut = len(zs) // 2
                alone = build_all(zs, gte[loc], tss, variants)
                alt = zs.copy(); alt[cut:] += 100.
                Ha, Hp = hist_subject(alt), hist_subject(zs[:cut])
                for v in variants:
                    full = Fte[v][loc]
                    assert np.array_equal(alone[v], full)
                    assert np.array_equal(build_subject(alt, tss, v, Ha)[:cut], full[:cut])
                    assert np.array_equal(build_subject(zs[:cut], tss[:cut], v, Hp), full[:cut])
            if fold == 0:  # train-side isolation check on one train subject
                s0 = np.unique(g[tr])[0]; loc = np.flatnonzero(g[tr] == s0)
                alone = build_all(Z[tr][loc], g[tr][loc], ts[tr][loc], variants)
                for v in variants: assert np.array_equal(alone[v], Ftr[v][loc])
            print(f'[seed {seed} fold {fold+1}/5] features + checks ok ntr={len(tr)} nte={len(te)} '
                  f'build_s={time.perf_counter()-t0:.1f} elapsed={time.perf_counter()-start:.1f}s', flush=True)
            folds.append({'fold': fold, 'test_subjects': [str(s) for s in np.unique(gte)]})
            for v in variants:
                for c in clfs:
                    t1 = time.perf_counter(); k = f'{v}__{c}'
                    scores[k][te] = make_clf(c).fit(Ftr[v], y[tr]).predict_proba(Fte[v])[:, 1]
                    print(f'[seed {seed} fold {fold+1}/5] {k:30s} dim={Ftr[v].shape[1]} fit_s={time.perf_counter()-t1:.1f} '
                          f'elapsed={time.perf_counter()-start:.1f}s', flush=True)
            if args.smoke: break
    ev = np.isfinite(scores[keys[0]]); M = {'all': ev, 'mature': ev & (ts >= MATURE)}
    R = result[f'seed{seed}'] = {'spearman_t_y': float(rho), 'folds': folds, 'methods': {}, 'bootstrap': {}}
    for k in keys:
        r = R['methods'][k] = {s: metrics(y, g, scores[k], M[s]) for s in SEGS}
        print(f'[result seed {seed}] {k:30s} ' + ' | '.join(
            f'{s}: n={r[s]["n"]} AUC={r[s]["AUC"] or float("nan"):.4f} mac={r[s]["subject_macro_bACC"] or float("nan"):.4f} '
            f'mic={r[s]["micro_bACC"]:.4f}' for s in SEGS), flush=True)
    for c in clfs:
        k = f'position_only__{c}'
        if k in R['methods']:
            auc = R['methods'][k]['all']['AUC']
            flag = 'OK' if auc is not None and abs(auc - .5) < .03 else 'FLAG: position carries label info!'
            print(f'[sanity seed {seed}] {k} AUC={auc:.4f} -> {flag}', flush=True)
            R.setdefault('sanity', {})[k] = {'AUC': auc, 'ok': flag == 'OK'}
    nb = 100 if args.smoke else 1000
    for k in keys:
        v, c = k.split('__'); ref = f'current_only__{c}'
        if v == 'current_only' or ref not in scores: continue
        for seg in SEGS:
            if np.unique(g[M[seg]]).size < 1: continue
            bb = R['bootstrap'][f'{k}__{seg}'] = boot(scores[k], scores[ref], y, g, M[seg], nb)
            print(f'[boot seed {seed}] {k:30s} vs {ref:18s} {seg:6s} '
                  f'dAUC={bb["dAUC"]["mean"]:+.4f}[{bb["dAUC"]["p2.5"]:+.4f},{bb["dAUC"]["p97.5"]:+.4f}] '
                  f'dmac={bb["dmacro_bACC"]["mean"]:+.4f}[{bb["dmacro_bACC"]["p2.5"]:+.4f},{bb["dmacro_bACC"]["p97.5"]:+.4f}] '
                  f'elapsed={time.perf_counter()-start:.1f}s', flush=True)
    return y, g, ts, ev, scores


def per_subject_count(R, y, g, ev, scores, k):
    subs = np.unique(g[ev])
    vals = {str(s): metrics(y, g, scores[k], ev & (g == s))['subject_macro_bACC'] for s in subs}
    arr = np.array([v for v in vals.values() if v is not None])
    print(f'[subject] {k}: n_subj={len(arr)} >=0.9: {(arr >= .9).sum()} mean={arr.mean():.4f}', flush=True)
    return {'per_subject': vals, 'n_ge_0.9': int((arr >= .9).sum()), 'n_subj': len(arr)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--seed-mode', choices=('independent', 'shared'), default='independent'); args = ap.parse_args()
    stem = 'depositioned_history_delta' + ('_sharedseed' if args.seed_mode == 'shared' else '') + ('_smoke' if args.smoke else '')
    tee = Tee(sys.stdout, OUT / (stem + '.log')); sys.stdout = tee
    start = time.perf_counter()
    man, subs, data = load_db()
    n = sum(len(v[1]) for v in data.values())
    print(f'[start] db rows={n} subjects={len(subs)} note="{man.get("note")}" smoke={args.smoke}', flush=True)
    if n != 16067: print(f'[warn] expected 16067 rows, got {n}', flush=True)
    result = {'meta': {'P_all': P_ALL, 'mature_index': MATURE, 'variants': VARIANTS, 'threshold': .5,
                       'rows': n, 'subjects': len(subs), 'hist_k_prior': 0, 'seed_mode': args.seed_mode}}
    if args.seed_mode == 'independent':  # cross-subject label-sequence correlation by stream index (should be ~0)
        _, y0, g0, _ = arrange(subs, data, 0, 'independent'); m = min(len(data[s][1]) for s in subs)
        S = np.array([y0[g0 == s][:m] for s in subs], float); C = np.corrcoef(S)
        result['meta']['cross_subject_label_corr'] = float((C.sum() - len(subs)) / (len(subs) * (len(subs) - 1)))
        print(f'[check] mean cross-subject label-sequence corr (first {m}) = {result["meta"]["cross_subject_label_corr"]:+.4f}', flush=True)
    y, g, ts, ev, sc0 = run_seed(0, subs, data, VARIANTS, ('hgb', 'lr'), args, start, result)
    R0 = result['seed0']
    # best delta variant: max subject macro bACC (all rows) among delta variants x classifier
    cand = [f'{v}__{c}' for v in DELTAS for c in ('hgb', 'lr')]
    best = max(cand, key=lambda k: R0['methods'][k]['all']['subject_macro_bACC'] or -1)
    best_hgb = max(DELTAS, key=lambda v: R0['methods'][f'{v}__hgb']['all']['subject_macro_bACC'] or -1)
    result['best_delta_seed0'] = best; result['best_delta_hgb_seed0'] = best_hgb
    print(f'[best] seed0 best delta={best}; best HGB delta={best_hgb}', flush=True)
    result['subject_counts_seed0'] = {k: per_subject_count(R0, y, g, ev, sc0, k)
                                      for k in dict.fromkeys([best, 'current_only__hgb', 'current_only__lr'])}
    oof = {f'seed0__{k}': v for k, v in sc0.items()}
    oof.update(y_seed0=y, groups_seed0=g, t_seed0=ts, eval_seed0=ev)
    for seed in (1, 2):
        ys, gs, tss, evs, scs = run_seed(seed, subs, data, ('current_only', best_hgb), ('hgb',), args, start, result)
        oof.update({f'seed{seed}__{k}': v for k, v in scs.items()})
        oof.update({f'y_seed{seed}': ys, f'groups_seed{seed}': gs, f't_seed{seed}': tss, f'eval_seed{seed}': evs})
    result['seconds'] = time.perf_counter() - start
    np.savez_compressed(OUT / (stem + '_oof.npz'), **oof)
    (OUT / (stem + '_results.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[done] elapsed={result["seconds"]:.1f}s', flush=True)
    sys.stdout = tee.stream; tee.f.close()


if __name__ == '__main__':
    main()
