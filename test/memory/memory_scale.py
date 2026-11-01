"""Experiment: causal online scale adaptation after centring (ana_bc pipeline, real recording order), h=0 only.

Zc_t = (Z_t - m_t) / max(s_t, floor) per feature, m_t = memory_center reference (median of prior mu0 [weight PRIOR] + past
rows inside the reference window), s_t = (w0 * s0 + n * s_past) / (w0 + n): s_past = 1.4826 * median|past - m_t| (mad)
or std(past) (std) over the n past rows in the window (n < 2 -> s0), s0 = fold-train prior scale (median over training
subjects of their per-subject MAD*1.4826 / std of Z), w0 = PRIOR while the prior pseudo-sample is inside the window (as
in centring), floor = FLOOR * s0. Only rows < t are used for m_t, s_t (strictly causal; window 6 s, no score smoothing).
Variants: c_win64 (centre only == memory_center win64), mad_win64, mad_win128, mad_all, std_win64.
Stage 1 (per outer fold k): every variant with realorder_v2's bc_sel config of fold k (reused). Stage 2: the TOPK
variants by inner h=0 macro bACC get bc_sel re-searched exactly like realorder_v2.inner_select (training subjects only,
same SGKF(4) split, within-subject AUC). Final = argmax inner h=0 macro bACC over all stage-1/2 candidates.
No test labels are used for any choice.
Run: python test/memory/memory_scale.py [--smoke]
"""
import argparse, json, sys, time

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from _mem import HERE, TEST, boot, el, evaluate, fmt, fmt_boot, label_shuffle_metric, load_oof, open_log, save, sig
import memory_center as MC

sys.path.insert(0, str(TEST / 'realorder'))
import realorder_v2 as RV  # noqa: E402

PRIOR, NJ, FLOOR = MC.PRIOR, 16, .1
VARIANTS = ('c_win64', 'mad_win64', 'mad_win128', 'mad_all', 'std_win64')


def parse(var):
    kind, ref = var.split('_'); return (None if kind == 'c' else kind), ref


def center_scale(X, mu0, s0, ref, kind):
    """one subject, real order -> (X_t - m_t) / max(s_t, floor), m_t, s_t from prior + rows < t only."""
    if kind is None: return MC.center_var(X, mu0, ref)
    n = len(X); out = np.empty_like(X); W = None if ref == 'all' else int(ref[3:]); fl = np.maximum(FLOOR * s0, 1e-6)
    for t in range(n):
        lo = 0 if W is None else max(0, t - W); P = X[lo:t]; pri = W is None or t + 1 <= W
        if pri: m = MC._wmedian(np.vstack([mu0[None], P]), np.r_[PRIOR, np.ones(len(P))])
        else: m = MC._wmedian(P, np.ones(len(P)))
        if len(P) >= 2:
            sp = 1.4826 * np.median(np.abs(P - m), 0) if kind == 'mad' else P.std(0)
            w0 = PRIOR if pri else 0.; s = (w0 * s0 + len(P) * sp) / (w0 + len(P))
        else: s = s0
        out[t] = (X[t] - m) / np.maximum(s, fl)
    return out


def _cs(X, mu0, s0, ref, kind):
    with threadpool_limits(limits=1): return center_scale(X, mu0, s0, ref, kind)


def _fit_tm(cfg, C, X, y, g, blk, itr):
    with threadpool_limits(limits=1): return RV.fit_model(cfg, C, X, y, g, blk, itr)


def prior_scale(Z, locs_tr):
    mad = np.median([1.4826 * np.median(np.abs(Z[l] - np.median(Z[l], 0)), 0) for l in locs_tr], 0)
    sd = np.median([Z[l].std(0) for l in locs_tr], 0)
    return {'mad': mad, 'std': sd, None: None}


def make_Zc(Z, mu0, s0s, var, locs, tr):
    kind, ref = parse(var)
    cz = Parallel(n_jobs=NJ, backend='loky')(delayed(_cs)(Z[l], mu0, s0s[kind], ref, kind) for l in locs)
    Zc = np.empty_like(Z)
    for l, r in zip(locs, cz): Zc[l] = r
    cmu, csd = Zc[tr].mean(0), Zc[tr].std(0); csd = np.where(csd > 1e-12, csd, 1.)
    return (Zc - cmu) / csd, cmu, csd


def score0(l, locs, y):
    return MC.inner_score(l, locs, y, hs=(0,))[0]


def prefix_check(Zs, mu0, s0s, var, cmu, csd, Tm, a_s, full):
    """one test subject: recompute the whole chain on the first half only -> must equal full output prefix."""
    kind, ref = parse(var); T, m = Tm; h = len(Zs) // 2

    def run(Z_, a_):
        with threadpool_limits(limits=1):
            Zc = (center_scale(Z_, mu0, s0s[kind], ref, kind) - cmu) / csd
            L = (a_ + m.decision_function(RV.apply_T(T, Zc))) / 2; z = RV.adapt(Zc, L)
            return np.where(np.isfinite(z), L + z, L)
    f_full, f_half = run(Zs, a_s), run(Zs[:h], a_s[:h])
    return float(np.abs(f_half - f_full[:h]).max()), float(np.abs(f_full - full).max())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); ap.add_argument('--boot', type=int)
    ap.add_argument('--topk', type=int, default=2)
    a = ap.parse_args(); nb = a.boot or (100 if a.smoke else 1000)
    stem = 'memory_scale' + ('_smoke' if a.smoke else ''); tee = open_log(stem)
    O = load_oof(a.smoke); D = O['D']; O['fold_of'] = D['fold_of']; y, g, ev, grp_of = O['y'], O['g'], O['ev'], O['grp_of']
    info = json.loads((HERE / ('realorder_v2_saveoof' + ('_smoke' if a.smoke else '') + '.json')).read_text('utf-8'))
    info = info['info']['real']
    RT = np.load(TEST / 'audit' / 'literature_protocol' / 'real_time.npz'); blk = RT['trial'].astype(int)
    base = O['oof__ana_bc']
    variants = ('c_win64', 'mad_win64') if a.smoke else VARIANTS; topk = 1 if a.smoke else a.topk
    MCO = np.load(HERE / ('memory_center' + ('_smoke' if a.smoke else '') + '_oof.npz'))['var__win64']
    print(f'[start] variants={list(variants)} topk={topk} prior={PRIOR} floor={FLOOR}*s0 boot={nb} h=0 only {el()}',
          flush=True)
    R = {'meta': {'variants': list(variants), 'topk': topk, 'prior_weight': PRIOR, 'floor_frac': FLOOR,
                  'cfgs': list(RV.CFGS), 'Cs': list(RV.CS)}, 'select': {}, 'fixed': {}, 'selected': {}, 'checks': {}}
    OUTV = {v: np.full(len(y), np.nan) for v in variants}
    out_re, out_fin = np.full(len(y), np.nan), np.full(len(y), np.nan)
    locs = [O['order'][s] for s in O['subs']]
    for k in [int(x) for x in O['folds']]:
        tf = time.perf_counter()
        tr = np.flatnonzero(O['fold'] != k); te_subs = [s for s in O['subs'] if O['fold_of'][s] == k]
        tr_subs = [s for s in O['subs'] if O['fold_of'][s] != k]
        loc_tr = [O['order'][s] for s in tr_subs]; loc_te = [O['order'][s] for s in te_subs]; te = np.concatenate(loc_te)
        Z = MC.build_Z(D, tr); mu0 = Z[tr].mean(0); s0s = prior_scale(Z, loc_tr)
        spl = [(tr[a_], tr[b_]) for a_, b_ in StratifiedGroupKFold(4, shuffle=True, random_state=0).split(tr, y[tr], g[tr])]
        a_out, a_in = O['oof__a'], O[f'inner{k}__a']; cfg0, C0 = tuple(info[str(k)]['bc_sel'])
        print(f'[fold {k}] reused bc_sel={cfg0} C={C0} test subj={len(te_subs)} {el()}', flush=True)

        def branch(Zc, cfg, C, bc_in=None):
            jobs = [(tr, te)] + ([] if bc_in is not None else spl)
            res = Parallel(n_jobs=5, backend='loky')(delayed(_fit_tm)(cfg, C, Zc, y, g, blk, a_) for a_, _ in jobs)
            T, m = res[0]; bc_out = np.full(len(y), np.nan); bc_out[te] = m.decision_function(RV.apply_T(T, Zc[te]))
            if bc_in is None:
                bc_in = np.full(len(y), np.nan)
                for (_, b_), (Ti, mi) in zip(spl, res[1:]): bc_in[b_] = mi.decision_function(RV.apply_T(Ti, Zc[b_]))
            f_out = MC.fused(Zc, (a_out + bc_out) / 2, loc_te); f_in = MC.fused(Zc, (a_in + bc_in) / 2, loc_tr)
            return f_out, f_in, (T, m)

        cand, ZC = {}, {}
        for var in variants:
            t1 = time.perf_counter(); Zc, cmu, csd = make_Zc(Z, mu0, s0s, var, locs, tr); ZC[var] = (Zc, cmu, csd)
            f_out, f_in, Tm = branch(Zc, cfg0, C0)
            cand[(var, 'reuse')] = {'out': f_out, 'in': f_in, 'inner': score0(f_in, loc_tr, y), 'cfg': [cfg0, C0]}
            OUTV[var][te] = f_out[te]
            print(f'[fold {k} s1 {var:10s}] inner h0={cand[(var, "reuse")]["inner"]:.4f} s={time.perf_counter() - t1:.1f} '
                  f'{el()}', flush=True)
            if k == int(O['folds'][0]):
                s0_ = loc_te[0]; kind, ref = parse(var); hh = len(s0_) // 2
                dc = np.abs(center_scale(Z[s0_][:hh], mu0, s0s[kind], ref, kind) -
                            center_scale(Z[s0_], mu0, s0s[kind], ref, kind)[:hh]).max()
                dp, df = prefix_check(Z[s0_], mu0, s0s, var, cmu, csd, Tm, a_out[s0_], f_out[s0_])
                R['checks'].setdefault('prefix', {})[var] = {'centre_scale': float(dc), 'chain': dp, 'chain_vs_run': df}
                print(f'[check {var}] prefix invariance centre/scale max|d|={dc:.1e} full chain max|d|={dp:.1e} '
                      f'(chain vs run {df:.1e})', flush=True)
                assert dc < 1e-12 and dp < 1e-9, (dc, dp)
        dm = float(np.nanmean(np.abs(OUTV['c_win64'][te] - MCO[te])))
        sg = float(np.mean(np.sign(OUTV['c_win64'][te]) == np.sign(MCO[te])))
        R['checks'].setdefault('c_win64_reproduces_memory_center', {})[k] = [dm, sg]
        print(f'[check f{k}] c_win64 vs memory_center win64 mean|d|={dm:.2e} sign agree={sg:.4f}', flush=True)
        assert sg > .99, sg
        top = sorted(variants, key=lambda v: (-cand[(v, 'reuse')]['inner'], variants.index(v)))[:topk]
        for var in top:
            t1 = time.perf_counter(); Zc = ZC[var][0]; tag = f'{var} f{k}'
            (cfg, C), crit, _ = RV.inner_select(Zc, y, g, blk, tr, tag)
            f_out, f_in, _ = branch(Zc, cfg, C, bc_in=RV.INNER_BEST[tag])
            cand[(var, 'research')] = {'out': f_out, 'in': f_in, 'inner': score0(f_in, loc_tr, y), 'cfg': [cfg, C],
                                       'crit_best': crit[f'{cfg}|C{C}']}
            print(f'[fold {k} s2 {var:10s}] re-searched bc_sel={cfg} C={C} inner h0={cand[(var, "research")]["inner"]:.4f}'
                  f' (reused {cand[(var, "reuse")]["inner"]:.4f}) s={time.perf_counter() - t1:.1f} {el()}', flush=True)
        keys = list(cand); fin = max(keys, key=lambda kk: (cand[kk]['inner'], -keys.index(kk)))
        out_re[te] = cand[(top[0], 'research')]['out'][te]; out_fin[te] = cand[fin]['out'][te]
        R['select'][k] = {'top': top, 'final': list(fin), 'final_cfg': cand[fin]['cfg'],
                          'inner': {f'{v}|{m}': c['inner'] for (v, m), c in cand.items()},
                          'cfg': {f'{v}|{m}': c['cfg'] for (v, m), c in cand.items()}}
        print(f'[select f{k}] top={top} final={fin[0]}|{fin[1]} cfg={cand[fin]["cfg"]} inner h0={cand[fin]["inner"]:.4f} '
              f'fold s={time.perf_counter() - tf:.1f} {el()}', flush=True)
        if k == int(O['folds'][0]):
            y2 = label_shuffle_metric(y, g, base, np.isin(np.arange(len(y)), te), O['fold'])
            same = all(score0(c['in'], loc_tr, y2) == c['inner'] for c in cand.values()); assert same
            R['checks']['test_label_perm_selection_identical'] = True
            print('[check] test-label permutation -> identical inner scores / selection', flush=True)

    ref_c = OUTV['c_win64']

    def report(name, l, store):
        m = evaluate(y, g, l, ev, grp_of); b1 = boot(sig(l), sig(base), y, g, ev, nb); b2 = boot(sig(l), sig(ref_c), y, g, ev, nb)
        store[name] = {'metrics': m, 'boot_vs_ana_bc': b1, 'boot_vs_c_win64': b2}
        print(f'[result] {name:12s} {fmt(m)} | vs ana_bc {fmt_boot(b1)} | vs c_win64 {fmt_boot(b2)}', flush=True)

    for v in variants: report(v, OUTV[v], R['fixed'])
    report('top1_research', out_re, R['selected']); report('final', out_fin, R['selected'])
    y3 = label_shuffle_metric(y, g, base, ev, O['fold'], seed=1); R['checks']['label_shuffle'] = {}
    for nm, l in (('top1_research', out_re), ('final', out_fin)):
        m = evaluate(y3, g, l, ev, grp_of); R['checks']['label_shuffle'][nm] = m
        print(f'[check] label shuffle {nm}: AUC={m["AUC"]:.4f} mac={m["subject_macro_bACC"]:.4f} (expect ~.5)', flush=True)
    np.savez(HERE / (stem + '_oof.npz'), y=y, g=g, fold=O['fold'], ev=ev, base=base, sel__final=out_fin,
             sel__top1_research=out_re, **{f'var__{v}': OUTV[v] for v in variants})
    save(stem, R, tee)


if __name__ == '__main__': main()
