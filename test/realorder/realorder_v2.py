"""Real-order v2 analogue + causal 889-feature median centring + block-robust training + causal EMA smoothing.

Protocol: outer SGKF(5, shuffle, rs0) by subject (== npz fold, asserted); 6 s windows; every per-subject statistic is
causal in the evaluated order. Orders: 'real' (recovered recording order, the deployment order) and shuffled stream
seeds s0/s1/s2 (control; == depositioned_resnet_history_delta.stream_order).
Per outer fold k:
  Z   = 889 (858 Fc + 31 Fn), train-median imputation + StandardScaler(fold-train rows)  (== v2 ZZ).
  Zc  = per subject, in the evaluated order: Z_t - median([16 x mu0, Z_0..Z_{t-1}]), mu0 = mean Z over fold-train rows
        (theory/subject_offset_ea.center 'median'), then StandardScaler(fold-train rows of Zc).
  Branches (logits): a = all_prank HGB (stored npz for s0..s2; RECOMPUTED for real order: feat_subject(Z,'hgb_prank') in
        real order, make_clf('hgb')), w/f = stored ResNet whiten/fbank logit_delta (s0..s2 only; NOT recomputed: GPU runs
        3879 s + 2637 s with order-dependent inputs/head, > 1 h), r = LR(C=.01) on Z (strict_LR889), c = LR(C=.01) on Zc.
  Block-robust (block ids of TRAINING subjects only): drift_{s,b} = mean_{s,b}(X) - mean_{s,class(b)}(X).
        scrN: |mean sign(task - relax)| over adjacent block pairs of all training subjects, top N features.
        pcK : remove top-K right singular vectors of the drift matrix (all rows).  bm: subtract drift from training rows.
        bz_scr250 / bz_pc5 / bz_bm (C=.01, on Z); *_sel: inner SGKF(4, shuffle, rs0) over training subjects, grid
        C {.003,.01,.03} x {none, scr100/250/500, pc2/5/10, bm, bm_pc5}, criterion mean within-subject AUC (first max).
        bz_sel on Z, bc_sel on Zc. (inner folds reuse the outer-train imputation / scaler / mu0: label-free.)
  Pseudo within-subject LR (== v2 pseudo_lr_full): refits t0 = 48, 64, ...; pseudo labels = top/bottom 30% of past L;
        StandardScaler(past X) -> LR(C=.01); z clipped +-3; fused = L + z (L where z undefined), L = mean of branches.
Tiers: raw | thr (fused - mean of previous <=256, >=16 rows; == v2 *_thr) | ema{h} (causal EMA half-life h windows,
  then thr) | emasel (h in {0=thr,2,4,8,16,32} chosen per outer fold by inner-OOF macro bACC of LR(C=.01) on Zc, ties ->
  smaller h; no test data used).
Checks: fold == npz; v2 reproduction (fuse_pseudo_lr_full_thr macro bACC vs v2 json, |d|<1e-9); strict_LR889 AUC vs theory
  json; all_prank recomputation == npz (s0 fold 0); prefix invariance (future Z rows + branch logits perturbed -> every
  output prefix identical, future changed); test-label + test-block shuffle (fold 0) -> identical outputs.
Run: python test/realorder/realorder_v2.py [--smoke] [--orders real,s0,...]
"""
import argparse, json, sys, time, warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.signal import lfilter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
TEST = HERE.parent
sys.path.insert(0, str(TEST / 'audit')); sys.path.insert(0, str(TEST))
from common import GROUPS, boot, load_all, metrics, open_log, save  # noqa: E402
from depositioned_history_delta import make_clf  # noqa: E402
from depositioned_feature_multiscale import feat_subject  # noqa: E402
from depositioned_adaptive_threshold import shift_subject  # noqa: E402
from depositioned_within_subject_adapt import pseudo_labels, R_STEP, MIN_PAST, THR_K  # noqa: E402

PRIOR, CLIP, EPS, NJ = 16, 3., 1e-6, 16
HS = (0, 2, 4, 8, 16, 32)
CS = (.003, .01, .03)
CFGS = ('none', 'scr100', 'scr250', 'scr500', 'pc2', 'pc5', 'pc10', 'bm', 'bm_pc5')
FIXED = {'bz_scr250': 'scr250', 'bz_pc5': 'pc5', 'bz_bm': 'bm'}
SINGLES = ('r', 'c', 'bz_scr250', 'bz_pc5', 'bz_bm', 'bz_sel', 'bc_sel', 'a')
PIPES_SH = {'v2': (('a', 'w', 'f'), 'Z'), 'v2c_in': (('a', 'w', 'f'), 'Zc'), 'v2c_br': (('a', 'w', 'f', 'c'), 'Z'),
            'v2c_both': (('a', 'w', 'f', 'c'), 'Zc'), 'v2c_bc': (('a', 'w', 'f', 'bc_sel'), 'Zc')}
PIPES_ALL = {'ana0': (('a', 'r'), 'Z'), 'ana_c': (('a', 'c'), 'Zc'), 'ana_bc': (('a', 'bc_sel'), 'Zc'),
             'lrc_ps': (('c',), 'Zc')}
TIERS = ('raw', 'thr') + tuple(f'ema{h}' for h in HS[1:]) + ('emasel',)
T0 = time.perf_counter()


def el():
    return f'elapsed={time.perf_counter() - T0:.1f}s'


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def sig(z):
    return 1. / (1. + np.exp(-z))


def stream_order(g, subs, seed):
    return [np.flatnonzero(g == s)[np.random.default_rng([seed, j]).permutation(int((g == s).sum()))]
            for j, s in enumerate(subs)]


def lr(C=.01, it=1000):
    return LogisticRegression(C=C, max_iter=it)


def center_median(X, mu0):
    out = np.empty_like(X); pri = np.repeat(mu0[None], PRIOR, 0)
    for i in range(len(X)): out[i] = X[i] - np.median(np.vstack([pri, X[:i]]), 0)
    return out


def _cm(X, mu0):
    with threadpool_limits(limits=1): return center_median(X, mu0)


def ema(l, h):
    a = 1. - 2. ** (-1. / h)
    return lfilter([a], [1., -(1. - a)], l, zi=[(1. - a) * l[0]])[0]


def tier_all(l, hsel):
    out = {'raw': l, 'thr': shift_subject(l, THR_K, 'mean')}
    for h in HS[1:]: out[f'ema{h}'] = shift_subject(ema(l, h), THR_K, 'mean')
    out['emasel'] = out['thr'] if hsel == 0 else out[f'ema{hsel}']
    return out


def zc(sp, sn):
    return np.clip((sn - sp.mean()) / max(sp.std(), EPS), -CLIP, CLIP)


def adapt(X, L):
    n = len(L); z = np.full(n, np.nan)
    for t0 in range(MIN_PAST, n, R_STEP):
        t1 = min(n, t0 + R_STEP); sc = StandardScaler().fit(X[:t0]); xp, xn = sc.transform(X[:t0]), sc.transform(X[t0:t1])
        idx, yl = pseudo_labels(L[:t0]); m = lr().fit(xp[idx], yl)
        z[t0:t1] = zc(m.decision_function(xp), m.decision_function(xn))
    return z


# ---------------------------------------------------------------- block-robust transforms (training rows only)
def drift(X, y, g, blk, rows):
    Dm, Dr = [], np.zeros((len(rows), X.shape[1])); pos = {r: i for i, r in enumerate(rows)}
    gr = g[rows]
    for s in np.unique(gr):
        rs = rows[gr == s]; cm = {c: X[rs[y[rs] == c]].mean(0) for c in (0, 1)}
        for b in np.unique(blk[rs]):
            rb = rs[blk[rs] == b]; d = X[rb].mean(0) - cm[int(y[rb[0]])]; Dm.append(d)
            Dr[[pos[r] for r in rb]] = d
    return np.array(Dm), Dr


def scr_scores(X, y, g, blk, rows):
    acc, cnt = np.zeros(X.shape[1]), 0; gr = g[rows]
    for s in np.unique(gr):
        rs = rows[gr == s]; bs = np.unique(blk[rs])
        mb = {b: (X[rs[blk[rs] == b]].mean(0), int(y[rs[blk[rs] == b]][0])) for b in bs}
        for b0, b1 in zip(bs[:-1], bs[1:]):
            (m0, l0), (m1, l1) = mb[b0], mb[b1]
            if l0 == l1: continue
            acc += np.sign(m1 - m0 if l1 == 1 else m0 - m1); cnt += 1
    return np.abs(acc / cnt)


def fit_T(cfg, X, y, g, blk, rows):
    """-> (apply-transform (picklable, small), per-train-row drift or None)."""
    if cfg == 'none': return ('none',), None
    if cfg.startswith('scr'):
        return ('cols', np.argsort(-scr_scores(X, y, g, blk, rows), kind='stable')[:int(cfg[3:])]), None
    Dm, Dr = drift(X, y, g, blk, rows)
    if cfg == 'bm': return ('none',), Dr
    K = int(cfg.split('pc')[1]); V = np.linalg.svd(Dm, full_matrices=False)[2][:K].T
    return ('proj', V), (Dr if cfg.startswith('bm') else None)


def apply_T(T, X):
    if T[0] == 'cols': return X[:, T[1]]
    if T[0] == 'proj': return X - (X @ T[1]) @ T[1].T
    return X


def fit_model(cfg, C, X, y, g, blk, rows):
    T, Dr = fit_T(cfg, X, y, g, blk, rows); Xr = X[rows] if Dr is None else X[rows] - Dr
    return T, lr(C, 2000).fit(apply_T(T, Xr), y[rows])


def _inner_job(cfg, C, X, y, g, blk, itr, iva):
    with threadpool_limits(limits=1):
        T, m = fit_model(cfg, C, X, y, g, blk, itr); return m.decision_function(apply_T(T, X[iva]))


def inner_select(X, y, g, blk, rows, tag):
    t1 = time.perf_counter()
    spl = [(rows[a], rows[b]) for a, b in StratifiedGroupKFold(4, shuffle=True, random_state=0).split(
        rows, y[rows], g[rows])]
    jobs = [(cfg, C, fi) for cfg in CFGS for C in CS for fi in range(len(spl))]
    res = Parallel(n_jobs=NJ, backend='loky')(delayed(_inner_job)(cfg, C, X, y, g, blk, *spl[fi]) for cfg, C, fi in jobs)
    oof = {}
    for (cfg, C, fi), d in zip(jobs, res):
        o = oof.setdefault((cfg, C), np.full(len(y), np.nan)); o[spl[fi][1]] = d
    gr = g[rows]; best, bv, crit = None, -1, {}
    for cfg in CFGS:
        for C in CS:
            o = oof[(cfg, C)]
            v = float(np.mean([roc_auc_score(y[rs], o[rs]) for s in np.unique(gr) if (rs := rows[gr == s]).size]))
            crit[f'{cfg}|C{C}'] = v
            if v > bv: best, bv = (cfg, C), v
    print(f'[inner {tag}] best={best[0]} C={best[1]} withinAUC={bv:.4f} (none C.01={crit["none|C0.01"]:.4f}) '
          f'jobs={len(jobs)} s={time.perf_counter() - t1:.1f} {el()}', flush=True)
    return best, crit, oof[('none', .01)]


def select_h(l, order_tr):
    """inner-OOF logits l of training subjects (lists of row indices in evaluated order) -> h (ties smaller)."""
    best, bv, sc = 0, -1, {}
    for h in HS:
        bs = []
        for loc in order_tr:
            q = l[loc]; q = shift_subject(q, THR_K, 'mean') if h == 0 else shift_subject(ema(q, h), THR_K, 'mean')
            yy = Y_GLOBAL[loc]; yh = q >= 0
            bs.append(.5 * (yh[yy == 1].mean() + (~yh[yy == 0]).mean()))
        sc[h] = float(np.mean(bs))
        if sc[h] > bv + 1e-12: best, bv = h, sc[h]
    return best, sc


# ---------------------------------------------------------------- per test subject
def process_subject(ctx, Zs, ext):
    """One test subject: Zs (n, 889) rows in evaluated order, ext {branch: logits} -> {(output, tier): logits}."""
    with threadpool_limits(limits=1):
        Zc = (center_median(Zs, ctx['mu0']) - ctx['cmu']) / ctx['csd']
        br = {'r': ctx['mr'].decision_function(Zs), 'c': ctx['mc'].decision_function(Zc)}
        for nm, (T, m) in ctx['bz'].items(): br[nm] = m.decision_function(apply_T(T, Zs))
        T, m = ctx['bc_sel']; br['bc_sel'] = m.decision_function(apply_T(T, Zc))
        br.update(ext)
        outs = {nm: br[nm] for nm in SINGLES}
        pipes = dict(PIPES_SH, **PIPES_ALL) if 'w' in ext else PIPES_ALL
        for pn, (bs, inp) in pipes.items():
            L = np.mean([br[b] for b in bs], 0); z = adapt(Zs if inp == 'Z' else Zc, L)
            outs[pn] = np.where(np.isfinite(z), L + z, L)
        return {(nm, t): v for nm, l in outs.items() for t, v in tier_all(l, ctx['hsel']).items()}


# ---------------------------------------------------------------- fold-level
def fold_common(D, k, y, blk):
    tr = np.flatnonzero(D['fold'] != k)
    Z = []
    for F in (D['Fc'], D['Fn']):
        med = np.nanmedian(np.where(np.isfinite(F[tr]), F[tr], np.nan), 0); med = np.where(np.isfinite(med), med, 0.)
        A = np.where(np.isfinite(F), F, med); Z.append(StandardScaler().fit(A[tr]).transform(A))
    Z = np.hstack(Z)
    with threadpool_limits(limits=8):
        mr = lr(.01, 2000).fit(Z[tr], y[tr])
        bz = {nm: fit_model(cfg, .01, Z, y, D['g'], blk, tr) for nm, cfg in FIXED.items()}
    (cfg, C), crit, _ = inner_select(Z, y, D['g'], blk, tr, f'bz f{k}')
    with threadpool_limits(limits=8): bz['bz_sel'] = fit_model(cfg, C, Z, y, D['g'], blk, tr)
    return dict(Z=Z, tr=tr, mr=mr, bz=bz, bz_sel_cfg=[cfg, C], bz_crit=crit)


def a_feats(Z, order):
    Fp = np.empty((len(Z), 3 * Z.shape[1]))
    for loc in order: Fp[loc] = feat_subject(Z[loc], 'hgb_prank')
    return Fp


def fold_order(D, FC, k, oname, order, y, blk, stored):
    g, Z, tr = D['g'], FC['Z'], FC['tr']; te_subs = [j for j, s in enumerate(D['subs']) if D['fold_of'][s] == k]
    mu0 = Z[tr].mean(0)
    res = Parallel(n_jobs=NJ, backend='loky')(delayed(_cm)(Z[loc], mu0) for loc in order)
    Zc = np.empty_like(Z)
    for loc, r in zip(order, res): Zc[loc] = r
    cmu, csd = Zc[tr].mean(0), Zc[tr].std(0); csd = np.where(csd > 1e-12, csd, 1.); Zc = (Zc - cmu) / csd
    with threadpool_limits(limits=8): mc = lr(.01, 2000).fit(Zc[tr], y[tr])
    (cfg, C), crit, oof_c = inner_select(Zc, y, g, blk, tr, f'bc {oname} f{k}')
    with threadpool_limits(limits=8): bc = fit_model(cfg, C, Zc, y, g, blk, tr)
    trset = set(tr.tolist()); order_tr = [loc for loc in order if loc[0] in trset]
    hsel, hsc = select_h(oof_c, order_tr)
    print(f'[emasel {oname} f{k}] h={hsel} inner macro=' + ' '.join(f'{h}:{v:.4f}' for h, v in hsc.items()), flush=True)
    ctx = dict(mu0=mu0, cmu=cmu, csd=csd, mr=FC['mr'], mc=mc, bz=FC['bz'], bc_sel=bc, hsel=hsel)
    hgb = None
    if stored is None:  # real order: recompute all_prank in this order
        t1 = time.perf_counter(); Fp = a_feats(Z, order)
        with threadpool_limits(limits=8): hgb = make_clf('hgb').fit(Fp[tr], y[tr])
        a_te = np.full(len(y), np.nan); te = np.flatnonzero(D['fold'] == k)
        a_te[te] = logit(hgb.predict_proba(Fp[te])[:, 1]); del Fp
        print(f'[hgb {oname} f{k}] all_prank recomputed s={time.perf_counter() - t1:.1f} {el()}', flush=True)
        ext_of = lambda loc: {'a': a_te[loc]}
    else:
        ext_of = lambda loc: {b: stored[b][loc] for b in ('a', 'w', 'f')}
    t1 = time.perf_counter()
    outs = Parallel(n_jobs=NJ, backend='loky')(delayed(process_subject)(ctx, Z[order[j]], ext_of(order[j]))
                                               for j in te_subs)
    print(f'[subjects {oname} f{k}] {len(te_subs)} processed s={time.perf_counter() - t1:.1f} {el()}', flush=True)
    info = {'bc_sel': [cfg, C], 'bc_crit': crit, 'hsel': hsel, 'h_inner_macro': hsc}
    return {j: o for j, o in zip(te_subs, outs)}, info, ctx, hgb


def prefix_check(ctx, hgb, Zs, ext, j):
    def run(Z_, e_):
        if hgb is not None: e_ = {'a': logit(hgb.predict_proba(feat_subject(Z_, 'hgb_prank'))[:, 1])}
        return process_subject(ctx, Z_, e_)
    n = len(Zs); cut = MIN_PAST + R_STEP * ((n - MIN_PAST) // (2 * R_STEP)) + 5
    rng = np.random.default_rng(j); Z1 = Zs.copy(); Z1[cut:] += rng.normal(0, 2, Z1[cut:].shape)
    e1 = {b: v.copy() for b, v in ext.items()}
    for b in e1: e1[b][cut:] += rng.normal(0, 2, n - cut)
    r0, r1 = run(Zs, ext), run(Z1, e1)
    for key in r0:
        assert np.array_equal(r0[key][:cut], r1[key][:cut]), ('prefix', key)
        assert np.abs(r0[key][cut:] - r1[key][cut:]).max() > 0, ('future unchanged', key)
    return r0, cut


# ---------------------------------------------------------------- evaluation
def evaluate(y, g, l, ev, grp_of):
    p = sig(l); m = metrics(y, g, p, ev); per = {}
    for s in np.unique(g[ev]):
        q = ev & (g == s); yh = p[q] >= .5; yy = y[q]
        per[s] = .5 * (yh[yy == 1].mean() + (~yh[yy == 0]).mean())
    m['groups'] = {G: float(np.mean([v for s, v in per.items() if grp_of[s] == G])) for G in GROUPS
                   if any(grp_of[s] == G for s in per)}
    m['n_ge_0.9'] = int(sum(v >= .9 for v in per.values())); m['n_subj'] = len(per)
    return m


def main():
    global Y_GLOBAL
    warnings.filterwarnings('ignore')
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); ap.add_argument('--orders', default='')
    args = ap.parse_args()
    stem = 'realorder_v2' + ('_smoke' if args.smoke else ''); tee = open_log(HERE, stem)
    D = load_all(); y, g, fold, subs, grp_of = D['y'], D['g'], D['fold'], D['subs'], D['grp_of']; Y_GLOBAL = y
    RT = np.load(TEST / 'audit' / 'literature_protocol' / 'real_time.npz')
    assert np.array_equal(RT['y'], y) and np.array_equal(RT['groups'], g)
    tt, blk = RT['t_end'], RT['trial'].astype(int)
    for s in subs:
        q = g == s
        for b in np.unique(blk[q]): assert np.unique(y[q & (blk == b)]).size == 1
    print('[check] fold == npz fold (load_all); blocks single-label', flush=True)
    A_ = D['A_']; B_ = np.load(TEST / 'depositioned_resnet_timefreq_oof.npz')
    for q in ('y', 'groups', 'fold'): assert np.array_equal(A_[q], B_[q]), q
    folds = [0] if args.smoke else list(range(5)); nb = 100 if args.smoke else 1000
    onames = args.orders.split(',') if args.orders else (['s0'] if args.smoke else ['real', 's0', 's1', 's2'])
    orders, stored = {}, {}
    for o in onames:
        if o == 'real':
            orders[o] = [np.flatnonzero(g == s)[np.argsort(tt[g == s], kind='stable')] for s in subs]; stored[o] = None
        else:
            sd = int(o[1:]); orders[o] = stream_order(g, subs, sd)
            stored[o] = {'a': logit(A_[f'stream{sd}__all_prank']), 'w': logit(B_[f'stream{sd}__resnet_whiten_logit_delta']),
                         'f': logit(B_[f'stream{sd}__resnet_fbank_logit_delta'])}
    print(f'[start] rows={len(y)} subjects={len(subs)} folds={folds} orders={onames} boot={nb} jobs={NJ} {el()}', flush=True)
    ev = np.isin(fold, folds); sidx = {s: j for j, s in enumerate(subs)}
    OUT = {o: {} for o in onames}; INFO = {o: {} for o in onames}
    res = {'meta': {'orders': onames, 'folds': folds, 'bootstrap_n': nb, 'prior': PRIOR, 'H': list(HS), 'C_grid': list(CS),
                    'cfgs': list(CFGS), 'recomputed': {'a(all_prank)': 'real order (HGB refit per fold)',
                                                       'r/c/bz/bc/pseudo/thr/ema': 'all orders',
                                                       'w,f(ResNet)': 'stored npz, shuffled orders only'}},
           'checks': {}, 'info': {}, 'methods': {}, 'bootstrap': {}}
    chk = res['checks']
    for k in folds:
        tk = time.perf_counter(); FC = fold_common(D, k, y, blk)
        res['info'][f'bz_f{k}'] = {'bz_sel': FC['bz_sel_cfg']}
        for o in onames:
            O, info, ctx, hgb = fold_order(D, FC, k, o, orders[o], y, blk, stored[o])
            INFO[o][k] = info
            for j, r in O.items():
                for key, v in r.items(): OUT[o].setdefault(key, np.full(len(y), np.nan))[orders[o][j]] = v
            if k == folds[0]:
                j = next(iter(O)); loc = orders[o][j]
                ext = {'a': OUT[o][('a', 'raw')][loc]} if stored[o] is None else {b: stored[o][b][loc] for b in 'awf'}
                r0, cut = prefix_check(ctx, hgb, FC['Z'][loc], ext, j)
                for key in r0: assert np.allclose(r0[key], OUT[o][key][loc], atol=1e-9, rtol=0), ('isolation', key)
                chk[f'prefix_{o}_{subs[j]}'] = {'cut': int(cut), 'n_outputs': len(r0), 'status': 'ok'}
                print(f'[check {o}] prefix invariance + isolation ok subject={subs[j]} cut={cut} outputs={len(r0)} {el()}',
                      flush=True)
                if o == onames[0]:  # test-label + test-block shuffle: refit fold, outputs must be identical
                    te = np.flatnonzero(fold == k); rng = np.random.default_rng(k)
                    y2, b2 = y.copy(), blk.copy(); y2[te] = rng.permutation(y2[te]); b2[te] = rng.permutation(b2[te])
                    Y_GLOBAL = y2; FC2 = fold_common(D, k, y2, b2)
                    O2, _, _, _ = fold_order(D, FC2, k, o, orders[o], y2, b2, stored[o]); Y_GLOBAL = y; del FC2
                    for jj in O:
                        for key in O[jj]: assert np.array_equal(O[jj][key], O2[jj][key]), ('label shuffle', key)
                    chk[f'label_block_shuffle_{o}_f{k}'] = 'identical'
                    print(f'[check {o}] test-label + test-block shuffle -> identical outputs {el()}', flush=True)
                if o == 's0' and k == 0:
                    t1 = time.perf_counter(); Fp = a_feats(FC['Z'], orders[o]); te = np.flatnonzero(fold == k)
                    with threadpool_limits(limits=8):
                        pa = make_clf('hgb').fit(Fp[FC['tr']], y[FC['tr']]).predict_proba(Fp[te])[:, 1]
                    dmax = float(np.abs(pa - A_['stream0__all_prank'][te]).max()); del Fp
                    assert dmax < 1e-9, dmax; chk['all_prank_recompute_s0_f0_maxabs'] = dmax
                    print(f'[check s0] all_prank recomputation == npz max|d|={dmax:.2e} s={time.perf_counter() - t1:.1f}',
                          flush=True)
        del FC
        print(f'[fold {k}] done s={time.perf_counter() - tk:.1f} {el()}', flush=True)
    res['info'].update({o: {str(k): v for k, v in INFO[o].items()} for o in onames})
    # ---------------- metrics
    for o in onames:
        M = res['methods'][o] = {}
        for (nm, t), l in OUT[o].items():
            M[f'{nm}|{t}'] = evaluate(y, g, l, ev, grp_of)
        for nm in sorted({k[0] for k in OUT[o]}, key=lambda s: (s not in SINGLES, s)):
            print(f'[result {o}] {nm:10s} ' + ' '.join(
                f'{t}:AUC={M[f"{nm}|{t}"]["AUC"]:.4f}/mac={M[f"{nm}|{t}"]["subject_macro_bACC"]:.4f}/n9={M[f"{nm}|{t}"]["n_ge_0.9"]}'
                for t in ('raw', 'thr', 'emasel')) + ' | ema mac ' +
                ' '.join(f'{h}:{M[f"{nm}|ema{h}"]["subject_macro_bACC"]:.4f}' for h in HS[1:]), flush=True)
        if o.startswith('s'):
            sd = int(o[1:]); vj = json.loads((TEST / ('depositioned_within_subject_adapt_v2' + ('_smoke' if args.smoke else '')
                                                      + '.json')).read_text(encoding='utf-8'))
            if f'stream{sd}' in vj:
                ref = vj[f'stream{sd}']['methods']['fuse_pseudo_lr_full_thr']['all']['subject_macro_bACC']
                got = M['v2|thr']['subject_macro_bACC']; assert abs(got - ref) < 1e-9, ('v2 reproduction', got, ref)
                chk[f'v2_reproduced_{o}'] = {'ref': ref, 'got': got}
                print(f'[check {o}] v2 fuse_pseudo_lr_full_thr reproduced mac={got:.6f} (ref {ref:.6f})', flush=True)
    tj = TEST / 'theory' / ('subject_offset_ea' + ('_smoke' if args.smoke else '') + '_results.json')
    if tj.exists():
        ref = json.loads(tj.read_text(encoding='utf-8'))['methods']['strict_LR889']['AUC']
        got = res['methods'][onames[0]]['r|raw']['AUC']; chk['strict_LR889_vs_theory'] = {'ref': ref, 'got': got}
        assert abs(got - ref) < 1e-4, ('strict_LR889', got, ref)
        print(f'[check] strict_LR889 AUC={got:.5f} vs theory {ref:.5f}', flush=True)
    # ---------------- bootstrap
    jobs = []
    for o in onames:
        names = sorted({k[0] for k in OUT[o]})
        pr = [(f'{nm}|{t}', 'r|raw') for nm in names for t in ('raw', 'thr', 'emasel')]
        pr += [(f'c|raw', 'r|raw'), ('ana_c|thr', 'ana0|thr'), ('ana_c|emasel', 'ana0|emasel'), ('ana_bc|thr', 'ana_c|thr'),
               ('bc_sel|raw', 'c|raw'), ('lrc_ps|thr', 'c|thr')]
        pr += [(f'{nm}|emasel', f'{nm}|thr') for nm in names]
        if 'v2' in names:
            pr += [(f'{v}|{t}', f'v2|{t}') for v in PIPES_SH if v != 'v2' for t in ('thr', 'emasel')]
            pr += [('v2c_bc|thr', 'v2c_both|thr')]
        for a_, b_ in dict.fromkeys(pr):
            if a_ != b_: jobs.append((o, a_, b_))
    t1 = time.perf_counter()

    def key(s): n_, t = s.split('|'); return (n_, t)
    bres = Parallel(n_jobs=NJ, backend='loky')(delayed(boot)(sig(OUT[o][key(a_)]), sig(OUT[o][key(b_)]), y, g, ev, nb)
                                               for o, a_, b_ in jobs)
    for (o, a_, b_), bb in zip(jobs, bres):
        res['bootstrap'][f'{o}::{a_}__vs__{b_}'] = bb
        print(f'[boot {o}] {a_:16s} vs {b_:14s} dAUC={bb["dAUC"]["mean"]:+.4f}[{bb["dAUC"]["p2.5"]:+.4f},'
              f'{bb["dAUC"]["p97.5"]:+.4f}] dmac={bb["dmacro_bACC"]["mean"]:+.4f}[{bb["dmacro_bACC"]["p2.5"]:+.4f},'
              f'{bb["dmacro_bACC"]["p97.5"]:+.4f}]', flush=True)
    print(f'[boot] {len(jobs)} comparisons s={time.perf_counter() - t1:.1f} {el()}', flush=True)
    # ---------------- shuffled seed mean + summary
    sh = [o for o in onames if o.startswith('s')]
    if sh:
        SM = res['methods']['shuffled_mean'] = {}
        for mk in res['methods'][sh[0]]:
            ms = [res['methods'][o][mk] for o in sh]
            SM[mk] = {'AUC': float(np.mean([m['AUC'] for m in ms])),
                      'subject_macro_bACC': float(np.mean([m['subject_macro_bACC'] for m in ms])),
                      'groups': {G: float(np.mean([m['groups'][G] for m in ms])) for G in ms[0]['groups']},
                      'n_ge_0.9': float(np.mean([m['n_ge_0.9'] for m in ms]))}
    print('[summary] order | output|tier | AUC mac high mid low n>=.9', flush=True)
    for o in (['shuffled_mean'] if sh else []) + [o for o in onames if o == 'real']:
        for mk in sorted(res['methods'][o]):
            if mk.split('|')[1] not in ('raw', 'thr', 'emasel'): continue
            m = res['methods'][o][mk]
            print(f'[summary] {o:13s} {mk:18s} {m["AUC"]:.4f} {m["subject_macro_bACC"]:.4f} ' +
                  ' '.join(f'{m["groups"].get(G, float("nan")):.4f}' for G in GROUPS) + f' {m["n_ge_0.9"]}', flush=True)
    res['seconds'] = time.perf_counter() - T0
    save(HERE, stem, res)
    print(f'[done] all checks passed {el()} -> {stem}.json', flush=True)
    sys.stdout = tee.stream; tee.f.close()


Y_GLOBAL = None
if __name__ == '__main__':
    main()
