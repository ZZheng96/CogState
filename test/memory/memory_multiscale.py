"""Experiment 3: history_leakfree five-scale (50/100/200/400/800) history features under the strict fold + real order.

Per outer fold k: Z = 889 (train-median impute + StandardScaler(fold-train)), clipped to +-8. Per subject in REAL order,
mean/std of the previous min(k, t) windows (current excluded) by exact int64 cumulative sums (Q = 1e5, |Z|<=8 -> no
overflow). Cold start: prior pseudo-sample of weight P0 = 16 with mean mu0 / variance v0 = fold-train within-subject
statistics; mean_t = (S1 + P0 mu0)/(n + P0), var_t = (S2 + P0 (v0 + mu0^2))/(n + P0) - mean_t^2 (integer part exact).
Feature = round((Z - mean)/max(sd, .1), 6). HGB(300, .06, depth 3, l2 1) per scale on ALL training rows (no 120-window
drop); D_hgb = mean probability over scales. Fusion with experiment 1/2 = equal-weight mean of logits (fixed, not tuned).
Reports: all windows, and pos>=120 (comparable with history_leakfree.py: eval >=120, bAmed = past-median threshold).
Controls: position-only HGB (56/57 subjects start with relax -> position alone carries label information),
prefix invariance of history features, label shuffle.
Run: python test/memory/memory_multiscale.py [--smoke]
"""
import argparse, json

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from _mem import (HERE, TEST, boot, el, evaluate, fmt, fmt_boot, hl_time, label_shuffle_metric, load_oof, open_log,
                  save, sig)

import sys; sys.path.insert(0, str(TEST))  # noqa: E402
from memory_stream_window6 import subject_median_threshold  # noqa: E402

SCALES, FLOOR, Q, P0, CLIPZ, K0 = (50, 100, 200, 400, 800), .1, 10 ** 5, 16, 8., 120


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def hgb():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_depth=3, l2_regularization=1.,
                                          random_state=0)


def build_Z(D, tr):
    Z = []
    for F in (D['Fc'], D['Fn']):
        med = np.nanmedian(np.where(np.isfinite(F[tr]), F[tr], np.nan), 0); med = np.where(np.isfinite(med), med, 0.)
        A = np.where(np.isfinite(F), F, med); Z.append(StandardScaler().fit(A[tr]).transform(A))
    return np.clip(np.hstack(Z), -CLIPZ, CLIPZ)


def hist_feats(Zs, k, mu0, v0):
    """one subject, real order: exact causal window stats (previous min(k,t) rows) + prior pseudo-sample."""
    zi = np.round(Zs * Q).astype(np.int64); d = Zs.shape[1]
    cs = np.vstack([np.zeros((1, d), np.int64), np.cumsum(zi, 0)])
    cs2 = np.vstack([np.zeros((1, d), np.int64), np.cumsum(zi * zi, 0)])
    t = np.arange(len(Zs)); lo = np.maximum(t - k, 0); n = (t - lo)[:, None].astype(np.float64)
    S1 = (cs[t] - cs[lo]).astype(np.float64) / Q; S2 = (cs2[t] - cs2[lo]).astype(np.float64) / Q / Q
    mean = (S1 + P0 * mu0) / (n + P0)
    var = (S2 + P0 * (v0 + mu0 ** 2)) / (n + P0) - mean ** 2
    sd = np.sqrt(np.maximum(var, 0.))
    return np.round((Zs - mean) / np.maximum(sd, FLOOR), 6)


def feats(Z, locs, k, mu0, v0):
    F = np.empty_like(Z)
    for loc in locs: F[loc] = hist_feats(Z[loc], k, mu0, v0)
    return F


def bamed(l, y, g, mask):
    """history_leakfree bAmed: threshold = median of own past >=20 scores (else .5 on probability)."""
    o = np.lexsort((POS, g)); inv = np.empty_like(o); inv[o] = np.arange(len(o))  # rows in real order per subject
    p = sig(l); thr = subject_median_threshold(p[o], g[o], mask[o])[inv]
    per = []
    for s in np.unique(g[mask]):
        q = mask & (g == s); yh = p[q] >= thr[q]; yy = y[q]
        if np.unique(yy).size == 2: per.append(.5 * (yh[yy == 1].mean() + (~yh[yy == 0]).mean()))
    return float(np.mean(per))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); ap.add_argument('--boot', type=int)
    a = ap.parse_args(); nb = a.boot or (100 if a.smoke else 1000)
    stem = 'memory_multiscale' + ('_smoke' if a.smoke else ''); tee = open_log(stem)
    O = load_oof(a.smoke); D = O['D']; y, g, ev, grp_of, fold = O['y'], O['g'], O['ev'], O['grp_of'], O['fold']
    pos = O['pos']; base = O['oof__ana_bc']; global POS; POS = pos
    sm = np.load(HERE / ('memory_smooth' + ('_smoke' if a.smoke else '') + '_oof.npz'))
    ce = np.load(HERE / ('memory_center' + ('_smoke' if a.smoke else '') + '_oof.npz'))
    assert np.array_equal(sm['y'], y) and np.array_equal(ce['y'], y)
    E1, E2 = sm['sel__auto'], ce['sel__center_smooth']
    print(f'[start] scales={list(SCALES)} (' + ' '.join(hl_time(s) for s in SCALES) + f') P0={P0} Q={Q} boot={nb} {el()}',
          flush=True)
    R = {'meta': {'scales': list(SCALES), 'scale_time': {s: hl_time(s) for s in SCALES}, 'prior_weight': P0, 'Q': Q,
                  'floor': FLOOR, 'clip': CLIPZ,
                  'note': 'scales >= 50 windows (300 s) span >= one R/T block: features exploit the paradigm '
                          'block structure; 56/57 subjects start with relax'}, 'checks': {}}
    P = {f'd{s}': np.full(len(y), np.nan) for s in SCALES}; P['pos_only'] = np.full(len(y), np.nan)
    locs = [O['order'][s] for s in O['subs']]
    for k in [int(x) for x in O['folds']]:
        t1 = time.perf_counter(); tr = np.flatnonzero(fold != k); te = np.flatnonzero(fold == k)
        Z = build_Z(D, tr); mu0 = Z[tr].mean(0)
        v0 = np.mean([Z[O['order'][s]].var(0) for s in O['subs'] if D['fold_of'][s] != k], 0)
        with threadpool_limits(limits=16):
            P['pos_only'][te] = hgb().fit(pos[tr, None].astype(float), y[tr]).predict_proba(pos[te, None].astype(float))[:, 1]
            for s in SCALES:
                F = feats(Z, locs, s, mu0, v0)
                if k == int(O['folds'][0]) and s == SCALES[0]:
                    loc = O['order'][[x for x in O['subs'] if D['fold_of'][x] == k][0]]; cut = len(loc) // 2
                    dd = float(np.abs(hist_feats(Z[loc][:cut], s, mu0, v0) - F[loc][:cut]).max())
                    assert dd == 0., dd
                    R['checks']['prefix_feature_maxdiff'] = dd
                    print(f'[check] history-feature prefix invariance max|d|={dd:.1e}', flush=True)
                P[f'd{s}'][te] = hgb().fit(F[tr], y[tr]).predict_proba(F[te])[:, 1]
                print(f'[fold {k} d{s}] fitted {el()}', flush=True)
        print(f'[fold {k}] done s={time.perf_counter() - t1:.1f} {el()}', flush=True)
    L = {f'd{s}': logit(P[f'd{s}']) for s in SCALES}
    L['D_hgb'] = logit(np.mean([P[f'd{s}'] for s in SCALES], 0)); L['pos_only'] = logit(P['pos_only'])
    L['ana_bc_raw'] = base; L['smooth_auto'] = E1; L['center_smooth'] = E2
    L['D_hgb+smooth'] = (L['D_hgb'] + E1) / 2; L['D_hgb+center_smooth'] = (L['D_hgb'] + E2) / 2
    L['D_hgb+ana_bc_raw'] = (L['D_hgb'] + base) / 2
    ev120 = ev & (pos >= K0); R['all'] = {}; R['eval120'] = {}
    for nm, l in L.items():
        for tag, mask, store in (('all', ev, R['all']), ('ev120', ev120, R['eval120'])):
            m = evaluate(y, g, l, mask, grp_of); m['bAmed'] = bamed(l, y, g, mask)
            bb = boot(sig(l), sig(base), y, g, mask, nb); store[nm] = {'metrics': m, 'boot_vs_raw': bb}
            print(f'[result {tag:5s}] {nm:20s} {fmt(m)} bAmed={m["bAmed"]:.4f} | vs raw {fmt_boot(bb)}', flush=True)
    y3 = label_shuffle_metric(y, g, base, ev, fold, seed=1); R['checks']['label_shuffle'] = {}
    for nm in ('D_hgb', 'D_hgb+smooth', 'D_hgb+center_smooth'):
        m = evaluate(y3, g, L[nm], ev, grp_of); R['checks']['label_shuffle'][nm] = m
        print(f'[check] label shuffle {nm}: AUC={m["AUC"]:.4f} mac={m["subject_macro_bACC"]:.4f} (expect ~.5)', flush=True)
    old = TEST / 'history_leakfree_results.json'
    if old.exists():
        R['old_history_leakfree_D_hgb'] = json.loads(old.read_text('utf-8'))['fusion'].get('D_hgb')
    np.savez(HERE / (stem + '_oof.npz'), y=y, g=g, fold=fold, ev=ev, pos=pos, **L)
    save(stem, R, tee)


import time  # noqa: E402

if __name__ == '__main__': main()
