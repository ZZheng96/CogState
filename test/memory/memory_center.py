"""Experiment 2: continuously-forgetting centring reference for Zc (ana_bc pipeline), real recording order.

ana_bc (realorder_v2) uses Zc_t = Z_t - median([16 x mu0, Z_0..Z_{t-1}]) (all past, never forgets). Here the reference is
replaced by a forgetting one; the prior mu0 (fold-train mean of Z) is a pseudo-sample at t=-1 with weight 16 and decays /
leaves the window exactly like a past row:
  all        median, all past (== ana_bc; reproduction asserted)
  emed{h}    exponentially weighted median, weight 2^(-age/h)       h in 32/64/128/256 windows
  emean{h}   exponentially weighted mean                             h in 32/64/128/256
  win{W}     median of the last W windows (+ prior while inside)     W in 64/128/256
Per outer fold k and variant: Zc -> StandardScaler(fold-train) -> c = LR(.01), bc_sel = realorder_v2's selected config of
fold k (cfg, C re-used, not re-searched, to bound cost) -> a = realorder_v2 all_prank HGB logit (does not use Zc) ->
ana = L + pseudo-adapt(Zc, L), L = (a + bc)/2. Inner OOF on TRAINING subjects (same SGKF(4) split as realorder_v2) gives
the inner macro bACC used to select (variant, smoothing half-life h in HS) jointly; 'center' = best variant with h=0,
'center+smooth' = joint best. No test labels/blocks are used.
Run: python test/memory/memory_center.py [--smoke]
"""
# --- test layout bootstrap (generated 2026-11-02): locate test/ root, shared modules and cache ---
import sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_TEST = _HERE
while _TEST.name != "test" and _TEST.parent != _TEST:
    _TEST = _TEST.parent
CACHE_DIR = _TEST / "cache"
if str(_TEST) not in _sys.path:
    _sys.path.append(str(_TEST))
for _d in sorted(_TEST.iterdir()):
    if _d.is_dir() and _d.name not in {"cache", "results", "pretrained_cache", "reports", "experiments", "_orphans", "__pycache__"}:
        if str(_d) not in _sys.path:
            _sys.path.append(str(_d))
# --- end bootstrap ---

import argparse, json

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from _mem import (HERE, HS, TEST, boot, ema, el, evaluate, fmt, fmt_boot, hl_time, label_shuffle_metric, load_oof,
                  open_log, save, sig)

sys_path = str(TEST / 'realorder')
import sys; sys.path.insert(0, sys_path)  # noqa: E402
import realorder_v2 as RV  # noqa: E402

PRIOR, NJ = RV.PRIOR, 16
VARIANTS = ('all',) + tuple(f'emed{h}' for h in (32, 64, 128, 256)) + tuple(f'emean{h}' for h in (32, 64, 128, 256)) + \
    tuple(f'win{w}' for w in (64, 128, 256))


def _wmedian(V, w):
    """column-wise weighted median of V (m, d) with weights w (m,), interpolated (== np.median for equal weights)."""
    o = np.argsort(V, 0, kind='stable'); Vs = np.take_along_axis(V, o, 0); cw = np.cumsum(w[o], 0); half = cw[-1] / 2
    lo = (cw < half - 1e-12).sum(0); hi = (cw <= half + 1e-12).sum(0)
    hi = np.minimum(hi, len(w) - 1); lo = np.minimum(lo, len(w) - 1)
    c = np.arange(V.shape[1])
    return .5 * (Vs[lo, c] + Vs[hi, c])


def center_var(X, mu0, var):
    """one subject, real order -> X_t - ref_t (ref from prior + rows < t only)."""
    n = len(X); out = np.empty_like(X)
    if var.startswith('emean'):
        h = int(var[5:]); dcy = 2. ** (-1. / h); S = PRIOR * mu0.copy(); W = float(PRIOR)
        for t in range(n):
            out[t] = X[t] - S / W; S = dcy * S + X[t]; W = dcy * W + 1.
        return out
    for t in range(n):
        if var == 'all':
            V = np.vstack([mu0[None], X[:t]]); w = np.r_[PRIOR, np.ones(t)]
        elif var.startswith('emed'):
            h = int(var[4:]); age = np.arange(t, 0, -1, dtype=float)  # row i has age t-i
            V = np.vstack([mu0[None], X[:t]]); w = np.r_[PRIOR * 2. ** (-(t + 1) / h), 2. ** (-age / h)]
        else:
            W_ = int(var[3:]); lo = max(0, t - W_)
            if t + 1 <= W_:
                V = np.vstack([mu0[None], X[lo:t]]); w = np.r_[PRIOR, np.ones(t - lo)]
            else:
                V = X[lo:t]; w = np.ones(t - lo)
        out[t] = X[t] - _wmedian(V, w)
    return out


def _cv(X, mu0, var):
    with threadpool_limits(limits=1): return center_var(X, mu0, var)


def _adapt1(X, L):
    with threadpool_limits(limits=1): return RV.adapt(X, L)


def _fit_dec(cfg, C, X, y, g, blk, itr, iva):
    with threadpool_limits(limits=1):
        T, m = RV.fit_model(cfg, C, X, y, g, blk, itr); return m.decision_function(RV.apply_T(T, X[iva]))


def build_Z(D, tr):
    Z = []
    for F in (D['Fc'], D['Fn']):
        med = np.nanmedian(np.where(np.isfinite(F[tr]), F[tr], np.nan), 0); med = np.where(np.isfinite(med), med, 0.)
        A = np.where(np.isfinite(F), F, med); Z.append(StandardScaler().fit(A[tr]).transform(A))
    return np.hstack(Z)


def fused(Zc, L, locs):
    zs = Parallel(n_jobs=NJ, backend='loky')(delayed(_adapt1)(Zc[loc], L[loc]) for loc in locs)
    out = np.full(len(L), np.nan)
    for loc, z in zip(locs, zs): out[loc] = np.where(np.isfinite(z), L[loc] + z, L[loc])
    return out


def inner_score(l, locs, y, hs=HS):
    return {h: float(np.mean([.5 * ((ema(l[loc], h)[y[loc] == 1] >= 0).mean() + (ema(l[loc], h)[y[loc] == 0] < 0).mean())
                              for loc in locs])) for h in hs}


def run_fold(O, D, k, cfgC, blk, y, variants, tag=''):
    g = O['g']; tr = np.flatnonzero(O['fold'] != k); te_subs = [s for s in O['subs'] if O['fold_of'][s] == k]
    tr_subs = [s for s in O['subs'] if O['fold_of'][s] != k]
    loc_tr = [O['order'][s] for s in tr_subs]; loc_te = [O['order'][s] for s in te_subs]
    Z = build_Z(D, tr); mu0 = Z[tr].mean(0); cfg, C = cfgC
    spl = [(tr[a_], tr[b_]) for a_, b_ in StratifiedGroupKFold(4, shuffle=True, random_state=0).split(tr, y[tr], g[tr])]
    a_out, a_in = O['oof__a'], O[f'inner{k}__a']
    res = {}
    for var in variants:
        t1 = time.perf_counter()
        locs = [O['order'][s] for s in O['subs']]
        cz = Parallel(n_jobs=NJ, backend='loky')(delayed(_cv)(Z[loc], mu0, var) for loc in locs)
        Zc = np.empty_like(Z)
        for loc, r in zip(locs, cz): Zc[loc] = r
        cmu, csd = Zc[tr].mean(0), Zc[tr].std(0); csd = np.where(csd > 1e-12, csd, 1.); Zc = (Zc - cmu) / csd
        te = np.concatenate(loc_te)
        jobs = [(tr, te)] + spl
        dec = Parallel(n_jobs=5, backend='loky')(delayed(_fit_dec)(cfg, C, Zc, y, g, blk, a_, b_) for a_, b_ in jobs)
        bc_out = np.full(len(y), np.nan); bc_out[te] = dec[0]; bc_in = np.full(len(y), np.nan)
        for (_, b_), d in zip(spl, dec[1:]): bc_in[b_] = d
        f_out = fused(Zc, (a_out + bc_out) / 2, loc_te); f_in = fused(Zc, (a_in + bc_in) / 2, loc_tr)
        res[var] = {'out': f_out, 'in': f_in, 'inner': inner_score(f_in, loc_tr, y),
                    'Zc': (Zc, cmu, csd) if var == variants[0] else None}
        print(f'[fold {k}{tag} {var:8s}] inner h0={res[var]["inner"][0]:.4f} best h='
              f'{max(HS, key=lambda h: res[var]["inner"][h])} s={time.perf_counter() - t1:.1f} {el()}', flush=True)
    return res, Z, mu0, loc_tr, loc_te


def select(res):
    keys = [(v, h) for v in res for h in HS]
    best_c = max(res, key=lambda v: (res[v]['inner'][0], -list(res).index(v)))
    best_j = max(keys, key=lambda vh: (res[vh[0]]['inner'][vh[1]], -list(res).index(vh[0]), -vh[1]))
    return best_c, best_j


import time  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); ap.add_argument('--boot', type=int)
    ap.add_argument('--variants', help='comma list (default: all VARIANTS); first must be "all"')
    a = ap.parse_args(); nb = a.boot or (100 if a.smoke else 1000)
    stem = 'memory_center' + ('_smoke' if a.smoke else ''); tee = open_log(stem)
    O = load_oof(a.smoke); D = O['D']; O['fold_of'] = D['fold_of']; y, g, ev, grp_of = O['y'], O['g'], O['ev'], O['grp_of']
    info = json.loads((HERE / ('realorder_v2_saveoof' + ('_smoke' if a.smoke else '') + '.json')).read_text('utf-8'))
    info = info['info']['real']
    RT = np.load(TEST / 'audit' / 'literature_protocol' / 'real_time.npz'); blk = RT['trial'].astype(int)
    base = O['oof__ana_bc']
    variants = tuple(a.variants.split(',')) if a.variants else VARIANTS
    assert variants[0] == 'all'
    print(f'[start] variants={list(variants)} HS={list(HS)} boot={nb} ' +
          ' '.join(f'{v}:' + hl_time(int(''.join(c for c in v if c.isdigit()))) for v in variants[1:]) + f' {el()}',
          flush=True)
    # weighted-median sanity: equal weights == np.median; 'all' == realorder_v2.center_median
    rng = np.random.default_rng(0); V = rng.normal(size=(7, 5)); assert np.allclose(_wmedian(V, np.ones(7)), np.median(V, 0))
    V = rng.normal(size=(8, 5)); assert np.allclose(_wmedian(V, np.ones(8)), np.median(V, 0))
    R = {'meta': {'variants': list(variants), 'HS': list(HS), 'prior_weight': PRIOR,
                  'note': 'h / W >= 32 windows (>=192 s) approach the 300 s R/T block: exploits paradigm autocorrelation'},
         'select': {}, 'fixed': {}, 'selected': {}, 'checks': {}}
    OUTV = {v: np.full(len(y), np.nan) for v in variants}
    out_c, out_j = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for k in [int(x) for x in O['folds']]:
        cfgC = tuple(info[str(k)]['bc_sel'])
        res, Z, mu0, loc_tr, loc_te = run_fold(O, D, k, cfgC, blk, y, variants)
        if k == int(O['folds'][0]):
            s0 = loc_te[0]
            d = np.abs(center_var(Z[s0], mu0, 'all') - RV.center_median(Z[s0], mu0)).max()
            assert d < 1e-9, d; R['checks']['all_eq_center_median'] = float(d)
            d = max(np.abs(center_var(Z[s0][:len(s0) // 2], mu0, v) - center_var(Z[s0], mu0, v)[:len(s0) // 2]).max()
                    for v in variants)
            assert d < 1e-12, d; R['checks']['prefix_centre'] = float(d)
            print(f'[check] all==center_median, centring prefix invariance ok (max|d|={d:.1e})', flush=True)
        te = np.concatenate(loc_te)
        d = float(np.nanmax(np.abs(res['all']['out'][te] - base[te])))
        R['checks'].setdefault('all_reproduces_ana_bc', {})[k] = d
        di = float(np.nanmax(np.abs(res['all']['in'][np.concatenate(loc_tr)] - O[f'inner{k}__ana_bc'][np.concatenate(loc_tr)])))
        R['checks'].setdefault('all_reproduces_inner_ana_bc', {})[k] = di
        dd = np.abs(res['all']['out'][te] - base[te]); dm = float(np.nanmean(dd))
        sg = float(np.mean(np.sign(res['all']['out'][te]) == np.sign(base[te])))
        R['checks'].setdefault('all_reproduces_ana_bc_mean_sign', {})[k] = [dm, sg]
        print(f'[check f{k}] variant all reproduces ana_bc raw max|d|={d:.2e} mean|d|={dm:.2e} sign agree={sg:.4f} '
              f'inner max|d|={di:.2e}', flush=True)
        # outer: realorder_v2 fit with 8 BLAS threads vs 1 here -> lbfgs stop-point noise (max ~.15 logit seen in f2)
        assert dm < 1e-2 and sg > .99 and di < 1e-6, (d, dm, sg, di)
        for v in variants: OUTV[v][te] = res[v]['out'][te]
        bc, (bj, hj) = select(res)
        R['select'][k] = {'center': bc, 'joint': [bj, hj], 'inner': {v: res[v]['inner'] for v in res}}
        print(f'[select f{k}] center={bc} inner={res[bc]["inner"][0]:.4f} | joint={bj}+h{hj} ({hl_time(hj)}) '
              f'inner={res[bj]["inner"][hj]:.4f} {el()}', flush=True)
        for loc in loc_te: out_c[loc] = res[bc]['out'][loc]; out_j[loc] = ema(res[bj]['out'][loc], hj)
        if k == int(O['folds'][0]):
            y2 = label_shuffle_metric(y, g, base, np.isin(np.arange(len(y)), te), O['fold'])
            sc2 = {v: inner_score(res[v]['in'], loc_tr, y2) for v in res}
            same = all(sc2[v] == res[v]['inner'] for v in res); assert same
            R['checks']['test_label_perm_selection_identical'] = True
            print('[check] test-label permutation -> identical inner scores / selection', flush=True)

    def report(name, l, store):
        m = evaluate(y, g, l, ev, grp_of); bb = boot(sig(l), sig(base), y, g, ev, nb)
        store[name] = {'metrics': m, 'boot_vs_raw': bb}
        print(f'[result] {name:14s} {fmt(m)} | vs raw {fmt_boot(bb)}', flush=True)

    for v in variants: report(v, OUTV[v], R['fixed'])
    report('center', out_c, R['selected']); report('center+smooth', out_j, R['selected'])
    y3 = label_shuffle_metric(y, g, base, ev, O['fold'], seed=1); R['checks']['label_shuffle'] = {}
    for nm, l in (('center', out_c), ('center+smooth', out_j)):
        m = evaluate(y3, g, l, ev, grp_of); R['checks']['label_shuffle'][nm] = m
        print(f'[check] label shuffle {nm}: AUC={m["AUC"]:.4f} mac={m["subject_macro_bACC"]:.4f} (expect ~.5)', flush=True)
    np.savez(HERE / (stem + '_oof.npz'), y=y, g=g, fold=O['fold'], ev=ev, base=base, sel__center=out_c,
             sel__center_smooth=out_j, **{f'var__{v}': OUTV[v] for v in variants})
    save(stem, R, tee)


if __name__ == '__main__': main()
