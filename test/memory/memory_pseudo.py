"""Experiment 5: real-order causal within-subject pseudo-label adaptation on top of the branch-stack seed.

Seed s_t = memory_branchstack 'auto' logit (causal multi-h EMA branch stack; outer OOF for test subjects, nested
GroupKFold OOF for the training subjects of each fold). Features Z = 889 (fold-train median impute + StandardScaler,
clip +-8, as memory_multiscale.build_Z). For one subject in real order, every REFIT=10 windows at t0 (t0 >= MINPAST):
  confident past rows P = {i < t0 : |s_i| >= q-quantile of |s_j| among j < t0 with the same sign} (per-sign -> balanced),
  pseudo label sign(s_i); ridge (dual form, lambda) on Z[P] -> z_t for t0 <= t < t0+REFIT, divided by the std of its own
  fitted values on P (scale-free). Output f_t = s_t + w z_t (f_t = s_t before the first refit).
(q, lambda, w) chosen per outer fold by the SAME procedure simulated on the TRAINING subjects (their nested seeds),
max inner macro bACC; w=0 (== seed) is in the grid. Test labels never used; position/blocks never used.
Diagnostics: seed pseudo-label accuracy vs true labels (all past windows / confident subset) per group.
Controls: prefix invariance, test-label permutation, bootstrap vs seed and vs raw ana_bc, rotated-stream control
(test stream circularly rotated; method rerun) and the position-only LR from memory_branchstack.
Run: python test/memory/memory_pseudo.py [--smoke]
"""
import argparse, time

import numpy as np
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from _mem import (GROUPS, HERE, boot, el, evaluate, fmt, fmt_boot, label_shuffle_metric, load_oof, open_log,
                  per_subject_bacc, save, sig)
from memory_multiscale import build_Z
from memory_branchstack import CANDS, feats, fit_cand, model, rot_offsets  # noqa: F401

REFIT, MINPAST = 10, 20
QS, LAMS, WS = (0., .3, .6), (100., 1000.), (0., .25, .5, 1., 2.)


def adapt(Zs, s, qs=QS, lams=LAMS):
    """one subject, real order. -> dict (q, lam) -> z (nan before first refit). Gram precomputed (n x n)."""
    n = len(s); K = Zs @ Zs.T; out = {(q, l): np.full(n, np.nan) for q in qs for l in lams}
    for t0 in range(MINPAST, n, REFIT):
        sp = s[:t0]; a = np.abs(sp); pos = sp >= 0; t1 = min(n, t0 + REFIT)
        for q in qs:
            sel = np.zeros(t0, bool)
            for m in (pos, ~pos):
                if m.sum() < 3: continue
                sel |= m & (a >= np.quantile(a[m], q))
            if (sel & pos).sum() < 3 or (sel & ~pos).sum() < 3: continue
            P = np.flatnonzero(sel); yp = np.where(pos[P], 1., -1.); yc = yp - yp.mean()
            Kpp = K[np.ix_(P, P)]; kc_mean = Kpp.mean(0)
            # centred kernel (feature centring on P)
            Kc = Kpp - kc_mean[None] - kc_mean[:, None] + kc_mean.mean()
            Kt = K[t0:t1][:, P]; Ktc = Kt - Kt.mean(1, keepdims=True) - kc_mean[None] + kc_mean.mean()
            for l in lams:
                alpha = np.linalg.solve(Kc + l * np.eye(len(P)), yc)
                fit = Kc @ alpha; sd = fit.std()
                if sd < 1e-9: continue
                out[(q, l)][t0:t1] = (Ktc @ alpha + yp.mean()) / sd
    return out


def _adapt(Zs, s):
    with threadpool_limits(limits=1): return adapt(Zs, s)


def combine(s, z, w):
    return s + w * np.nan_to_num(z)


def run_adapt(Z, S, locs, nj):
    return Parallel(n_jobs=nj, backend='loky')(delayed(_adapt)(Z[loc], S[loc]) for loc in locs)


def inner_select(Ain, S, locs, y):
    sc = {}
    for q in QS:
        for l in LAMS:
            for w in WS:
                v = []
                for A, loc in zip(Ain, locs):
                    f = combine(S[loc], A[(q, l)], w); yy = y[loc]
                    v.append(.5 * ((f[yy == 1] >= 0).mean() + (f[yy == 0] < 0).mean()))
                sc[(q, l, w)] = float(np.mean(v))
    best = max(sc, key=lambda k: (sc[k], -k[2], -k[0], k[1]))
    return best, sc


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); ap.add_argument('--boot', type=int)
    ap.add_argument('--jobs', type=int, default=12)
    a = ap.parse_args(); nb = a.boot or (100 if a.smoke else 1000)
    stem = 'memory_pseudo' + ('_smoke' if a.smoke else ''); tee = open_log(stem)
    O = load_oof(a.smoke); D = O['D']; O['fold_of'] = D['fold_of']
    y, g, ev, grp_of, fold = O['y'], O['g'], O['ev'], O['grp_of'], O['fold']
    base = O['oof__ana_bc']
    BS = np.load(HERE / ('memory_branchstack' + ('_smoke' if a.smoke else '') + '_oof.npz'))
    assert np.array_equal(BS['y'], y)
    seed_out, seed_in = BS['sel__auto'], BS['seed_inner']; pos_only = BS['pos_only']
    print(f'[start] REFIT={REFIT} MINPAST={MINPAST} QS={QS} LAMS={LAMS} WS={WS} boot={nb} jobs={a.jobs} {el()}', flush=True)
    R = {'meta': {'refit': REFIT, 'min_past': MINPAST, 'qs': QS, 'lams': LAMS, 'ws': WS, 'seed': 'memory_branchstack auto'},
         'select': {}, 'checks': {}, 'seed_acc': {}}
    out = np.full(len(y), np.nan); rot = {rk: np.full(len(y), np.nan) for rk in ('random', 'same_phase', 'flip_phase')}
    rot_seed = {rk: np.full(len(y), np.nan) for rk in rot}
    rng = np.random.default_rng(123)
    for k in [int(x) for x in O['folds']]:
        t1 = time.perf_counter(); tr = np.flatnonzero(fold != k)
        Z = build_Z(D, tr)
        tr_subs = [s for s in O['subs'] if O['fold_of'][s] != k]; te_subs = [s for s in O['subs'] if O['fold_of'][s] == k]
        loc_tr = [O['order'][s] for s in tr_subs]; loc_te = [O['order'][s] for s in te_subs]
        Sin = seed_in[k]; assert all(np.isfinite(Sin[loc]).all() for loc in loc_tr)
        Ain = run_adapt(Z, Sin, loc_tr, a.jobs)
        print(f'[fold {k}] inner adapt done ({len(loc_tr)} subjects) {el()}', flush=True)
        best, sc = inner_select(Ain, Sin, loc_tr, y)
        R['select'][k] = {'best': list(best), 'inner_best': sc[best], 'inner_seed': sc[(QS[0], LAMS[0], 0.)],
                          'inner': {str(kk): v for kk, v in sc.items()}}
        print(f'[select f{k}] q={best[0]} lam={best[1]} w={best[2]} inner={sc[best]:.4f} seed={sc[(QS[0], LAMS[0], 0.)]:.4f} '
              f'{el()}', flush=True)
        q, l, w = best
        Ate = run_adapt(Z, seed_out, loc_te, a.jobs)
        for A, loc in zip(Ate, loc_te): out[loc] = combine(seed_out[loc], A[(q, l)], w)
        # rotated-stream control: seed recomputed on the rotated stream (branch-stack model of this fold), then adapt
        mdl = fit_cand(O, y, f'inner{k}', loc_tr, BS_BEST[k], nested=False)[0]
        jobs = []
        for loc in loc_te:
            for rk, r in rot_offsets(y[loc], rng).items():
                if r is None: continue
                lr = np.roll(loc, -r); sr = mdl.decision_function(feats(O, 'oof', lr, BS_BEST[k])); jobs.append((rk, lr, sr))
        As = Parallel(n_jobs=a.jobs, backend='loky')(delayed(_adapt)(Z[lr], sr) for _, lr, sr in jobs)
        for (rk, lr, sr), A in zip(jobs, As): rot[rk][lr] = combine(sr, A[(q, l)], w); rot_seed[rk][lr] = sr
        # seed pseudo-label accuracy (diagnostic, true labels): all windows / confident (q) subset of the final past
        for s, loc, A in zip(te_subs, loc_te, Ate):
            ss = seed_out[loc]; yy = y[loc]; aa = np.abs(ss); m = np.zeros(len(ss), bool)
            for sg in (ss >= 0, ss < 0):
                if sg.sum(): m |= sg & (aa >= np.quantile(aa[sg], q))
            R['seed_acc'][s] = {'all': float(((ss >= 0) == (yy == 1)).mean()), 'conf': float(((ss[m] >= 0) == (yy[m] == 1)).mean()),
                                'group': grp_of[s]}
        if k == int(O['folds'][0]):
            loc = loc_te[0]; cut = len(loc) // 2
            Ap = adapt(Z[loc][:cut], seed_out[loc][:cut])
            d = float(np.max(np.abs(combine(seed_out[loc][:cut], Ap[(q, l)], w) - out[loc][:cut])))
            assert d < 1e-8, d; R['checks']['prefix'] = {'subject': te_subs[0], 'cut': cut, 'max_abs_diff': d}
            print(f'[check] prefix invariance ok subject={te_subs[0]} cut={cut} max|d|={d:.1e}', flush=True)
            y2 = label_shuffle_metric(y, g, base, np.isin(g, te_subs), fold)
            b2, sc2 = inner_select(Ain, Sin, loc_tr, y2)
            assert b2 == best and all(sc2[kk] == sc[kk] for kk in sc); R['checks']['test_label_perm_selection_identical'] = True
            print('[check] test-label permutation -> identical inner scores / selection', flush=True)
        print(f'[fold {k}] done s={time.perf_counter() - t1:.1f} {el()}', flush=True)
    assert np.isfinite(out[ev]).all()

    def report(name, l, store, ref, refname, mask=ev):
        m = evaluate(y, g, l, mask, grp_of); bb = boot(sig(l), sig(ref), y, g, mask, nb)
        store[name] = {'metrics': m, f'boot_vs_{refname}': bb}
        print(f'[result] {name:12s} {fmt(m)} | vs {refname} {fmt_boot(bb)}', flush=True)

    R['results'] = {}
    report('ana_bc_raw', base, R['results'], base, 'raw'); report('pos_only', pos_only, R['results'], base, 'raw')
    report('seed', seed_out, R['results'], base, 'raw'); report('pseudo', out, R['results'], base, 'raw')
    R['results_vs_seed'] = {}; report('pseudo', out, R['results_vs_seed'], seed_out, 'seed')
    R['rotation'] = {}
    for rk in rot:
        mk = ev & np.isfinite(rot[rk])
        ms, mp = evaluate(y, g, rot_seed[rk], mk, grp_of), evaluate(y, g, rot[rk], mk, grp_of)
        R['rotation'][rk] = {'seed': ms, 'pseudo': mp}
        print(f'[control] rotated {rk:10s} seed {fmt(ms)} | pseudo {fmt(mp)}', flush=True)
    for G in GROUPS:
        v = [d for d in R['seed_acc'].values() if d['group'] == G]
        if v:
            R['seed_acc_group'] = R.get('seed_acc_group', {})
            R['seed_acc_group'][G] = {'all': float(np.mean([d['all'] for d in v])), 'conf': float(np.mean([d['conf'] for d in v])),
                                      'min_all': float(np.min([d['all'] for d in v])), 'n': len(v)}
            print(f'[diag] seed pseudo-label acc {G}: all={R["seed_acc_group"][G]["all"]:.4f} '
                  f'conf={R["seed_acc_group"][G]["conf"]:.4f} min={R["seed_acc_group"][G]["min_all"]:.4f} n={len(v)}',
                  flush=True)
    y3 = label_shuffle_metric(y, g, base, ev, fold, seed=1)
    m = evaluate(y3, g, out, ev, grp_of); R['checks']['label_shuffle'] = m
    print(f'[check] label shuffle pseudo: AUC={m["AUC"]:.4f} mac={m["subject_macro_bACC"]:.4f} (expect ~.5)', flush=True)
    R['per_subject'] = {s: [float(v), float(per_subject_bacc(y, g, seed_out, ev)[s])]
                        for s, v in per_subject_bacc(y, g, out, ev).items()}
    np.savez(HERE / (stem + '_oof.npz'), y=y, g=g, fold=fold, ev=ev, base=base, seed=seed_out, pseudo=out,
             **{f'rot__{rk}': rot[rk] for rk in rot}, **{f'rotseed__{rk}': rot_seed[rk] for rk in rot})
    save(stem, R, tee)


BS_BEST = {}

if __name__ == '__main__':
    import json
    _j = json.loads((HERE / ('memory_branchstack' + ('_smoke' if '--smoke' in __import__('sys').argv else '') + '.json'))
                    .read_text('utf-8'))
    BS_BEST.update({int(k): v['best'] for k, v in _j['select'].items()})
    main()
