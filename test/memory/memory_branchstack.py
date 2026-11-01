"""Experiment 4: multi-branch causal EMA stack, real recording order.

Features per window t (one subject, real order) = causal multi-h EMA (HS from _mem, h=0 raw) of EACH branch logit:
  ana  : ana_bc only                       (10 features == memory_smooth 'stack')
  br3  : a, bc_sel, c                      (30)
  br4  : a, bc_sel, c, ana_bc              (40)
Per outer fold k: StandardScaler + LR(C=1) fitted on inner-OOF branch logits (inner{k}__*) of the TRAINING subjects only;
candidate chosen by nested GroupKFold(4)-over-training-subjects macro bACC; applied to outer-OOF logits (oof__*) of the
test subjects. Centring variant (win64-centred branch logit) is NOT used: memory_center saved only outer logits, inner OOF
of the win64 variant would require re-running the bc_sel config search inside each fold; we do not reuse realorder_v2's
per-fold selection here.
Controls: position-only LR (pos, pos/50 sin/cos at block period 50 / 100 windows) fitted on training subjects = leakage
ceiling; rotation control = every test stream circularly rotated by a random offset before applying the (time-invariant)
method: a position-exploiting method collapses, a causal-memory method keeps its score up to the wrap discontinuity.
Checks: prefix invariance, test-label permutation (selection identical, metrics vs permuted labels ~ .5), bootstrap.
Run: python test/memory/memory_branchstack.py [--smoke]
"""
import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from _mem import (HERE, HS, boot, ema_stack, el, evaluate, fmt, fmt_boot, label_shuffle_metric, load_oof, open_log,
                  per_subject_bacc, save, sig)

CANDS = {'ana': ('ana_bc',), 'br3': ('a', 'bc_sel', 'c'), 'br4': ('a', 'bc_sel', 'c', 'ana_bc')}
PERIOD = 50  # windows per R/T block (300 s / 6 s)


HSHORT = tuple(h for h in HS if h <= 16)  # reference variant without long (>= 192 s) memory, not in selection


def feats(O, key, loc, cand):
    """one subject (row indices in real order) -> causal multi-h EMA of each branch; key = 'oof' or 'inner{k}'."""
    if cand == 'br4short': return np.hstack([ema_stack(O[f'{key}__{b}'][loc], HSHORT) for b in CANDS['br4']])
    return np.hstack([ema_stack(O[f'{key}__{b}'][loc]) for b in CANDS[cand]])


def rot_offsets(yl, rng):
    """CONTROL ONLY (uses labels to build the control stream, never in the method): random offset, offset at a block
    onset with the same label as the first window (start phase kept), and at an onset of the other label (flipped)."""
    n = len(yl); on = np.flatnonzero(np.diff(yl) != 0) + 1; mid = on[(on >= n // 4) & (on <= 3 * n // 4)]
    if len(mid) == 0: mid = on
    same = mid[yl[mid] == yl[0]]; flip = mid[yl[mid] != yl[0]]
    pick = lambda v: int(rng.choice(v)) if len(v) else None  # noqa: E731
    return {'random': int(rng.integers(n // 4, 3 * n // 4)), 'same_phase': pick(same), 'flip_phase': pick(flip)}


def model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=3000))


def _bacc(yy, d):
    return .5 * ((d[yy == 1] >= 0).mean() + (d[yy == 0] < 0).mean())


def fit_cand(O, y, key, locs, cand, nested=True):
    """fit on locs (training subjects); returns (model, nested macro, nested OOF logits per loc)."""
    X = np.vstack([feats(O, key, loc, cand) for loc in locs]); yy = np.concatenate([y[loc] for loc in locs])
    gg = np.concatenate([np.full(len(loc), j) for j, loc in enumerate(locs)])
    oof = np.full(len(yy), np.nan); sc = []
    if nested:
        for a_, b_ in GroupKFold(4).split(X, yy, gg):
            oof[b_] = model().fit(X[a_], yy[a_]).decision_function(X[b_])
        sc = [_bacc(yy[gg == j], oof[gg == j]) for j in range(len(locs))]
    m = model().fit(X, yy)
    cuts = np.cumsum([0] + [len(loc) for loc in locs])
    return m, float(np.mean(sc)) if sc else np.nan, [oof[cuts[j]:cuts[j + 1]] for j in range(len(locs))]


def select_fold(O, y, k):
    tr_subs = [s for s in O['subs'] if O['fold_of'][s] != k]; locs = [O['order'][s] for s in tr_subs]
    for b in ('a', 'bc_sel', 'c', 'ana_bc'): assert all(np.isfinite(O[f'inner{k}__{b}'][loc]).all() for loc in locs)
    res = {c: fit_cand(O, y, f'inner{k}', locs, c) for c in CANDS}
    best = max(CANDS, key=lambda c: (res[c][1], -list(CANDS).index(c)))
    return res, best, tr_subs, locs


def pos_feats(n):
    p = np.arange(n, dtype=float)
    return np.column_stack([p / 100.] + [f(2 * np.pi * p / P) for P in (2 * PERIOD, PERIOD) for f in (np.sin, np.cos)])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); ap.add_argument('--boot', type=int)
    a = ap.parse_args(); nb = a.boot or (100 if a.smoke else 1000)
    stem = 'memory_branchstack' + ('_smoke' if a.smoke else ''); tee = open_log(stem)
    O = load_oof(a.smoke); O['fold_of'] = O['D']['fold_of']
    y, g, ev, grp_of = O['y'], O['g'], O['ev'], O['grp_of']
    base = O['oof__ana_bc']
    for b in ('a', 'bc_sel', 'c', 'ana_bc'): assert np.isfinite(O[f'oof__{b}'][ev]).all()
    sm = np.load(HERE / ('memory_smooth' + ('_smoke' if a.smoke else '') + '_oof.npz')); smooth_stack = sm['sel__stack']
    assert np.array_equal(sm['y'], y)
    print(f'[start] cands={ {c: list(v) for c, v in CANDS.items()} } HS={list(HS)} boot={nb} {el()}', flush=True)
    R = {'meta': {'cands': {c: list(v) for c, v in CANDS.items()}, 'HS': list(HS),
                  'centring_variant': 'not used (inner OOF of win64 centring unavailable without re-searching bc_sel '
                                      'inside each fold; not re-searched)'},
         'select': {}, 'fixed': {}, 'selected': {}, 'checks': {}}
    out = {c: np.full(len(y), np.nan) for c in CANDS}; out['auto'] = np.full(len(y), np.nan)
    out['br4short'] = np.full(len(y), np.nan)
    RK, RM = ('random', 'same_phase', 'flip_phase'), ('auto', 'ana', 'br4short', 'pos_only', 'ana_bc_raw')
    rot = {(rk, c): np.full(len(y), np.nan) for rk in RK for c in RM}
    pos_out = np.full(len(y), np.nan); inner_seed = {}
    rng = np.random.default_rng(123)
    for k in [int(x) for x in O['folds']]:
        res, best, tr_subs, locs = select_fold(O, y, k)
        res['br4short'] = fit_cand(O, y, f'inner{k}', locs, 'br4short', nested=False)
        R['select'][k] = {'best': best, 'nested_inner_macro': {c: res[c][1] for c in CANDS}}
        print(f'[select f{k}] best={best} nested inner ' + ' '.join(f'{c}:{res[c][1]:.4f}' for c in CANDS) + f' {el()}',
              flush=True)
        # nested-OOF seed of the selected candidate on training subjects (used by memory_pseudo)
        for s, l in zip(tr_subs, res[best][2]): inner_seed[(k, s)] = l
        te_subs = [s for s in O['subs'] if O['fold_of'][s] == k]
        Xp = np.vstack([pos_feats(len(loc)) for loc in locs]); yp = np.concatenate([y[loc] for loc in locs])
        pm = LogisticRegression(C=1., max_iter=3000).fit(Xp, yp)
        for s in te_subs:
            loc = O['order'][s]; n = len(loc)
            for c in list(CANDS) + ['br4short']: out[c][loc] = res[c][0].decision_function(feats(O, 'oof', loc, c))
            out['auto'][loc] = out[best][loc]
            pos_out[loc] = pm.decision_function(pos_feats(n))
            for rk, r in rot_offsets(y[loc], rng).items():
                if r is None: continue
                lr = np.roll(loc, -r)  # rotated stream; lr[j] = loc[(j+r)%n]
                for c, cc in (('auto', best), ('ana', 'ana'), ('br4short', 'br4short')):
                    rot[(rk, c)][lr] = res[cc][0].decision_function(feats(O, 'oof', lr, cc))
                rot[(rk, 'pos_only')][lr] = pm.decision_function(pos_feats(n)); rot[(rk, 'ana_bc_raw')][lr] = base[lr]
        if k == int(O['folds'][0]):
            s = te_subs[0]; loc = O['order'][s]; cut = len(loc) // 2
            d = float(np.max(np.abs(res[best][0].decision_function(feats(O, 'oof', loc[:cut], best)) - out['auto'][loc][:cut])))
            assert d < 1e-9, d; R['checks']['prefix'] = {'subject': s, 'cut': cut, 'max_abs_diff': d}
            print(f'[check] prefix invariance ok subject={s} cut={cut} max|d|={d:.1e}', flush=True)
            y2 = label_shuffle_metric(y, g, base, np.isin(g, te_subs), O['fold'])
            res2, best2, _, _ = select_fold(O, y2, k)
            same = best2 == best and all(np.allclose(res2[c][0][-1].coef_, res[c][0][-1].coef_) for c in CANDS)
            assert same; R['checks']['test_label_perm_selection_identical'] = True
            print(f'[check] test-label permutation -> identical selection {el()}', flush=True)
    for c in out: assert np.isfinite(out[c][ev]).all()

    def report(name, l, store, ref=base, refname='raw'):
        m = evaluate(y, g, l, ev, grp_of); bb = boot(sig(l), sig(ref), y, g, ev, nb)
        store[name] = {'metrics': m, f'boot_vs_{refname}': bb}
        print(f'[result] {name:16s} {fmt(m)} | vs {refname} {fmt_boot(bb)}', flush=True)
        return m

    report('ana_bc_raw', base, R['fixed']); report('smooth_stack', smooth_stack, R['fixed'])
    report('pos_only', pos_out, R['fixed'])
    for c in list(CANDS) + ['br4short']: report(c, out[c], R['fixed'])
    report('auto', out['auto'], R['selected'])
    report('auto', out['auto'], R['selected'].setdefault('vs_smooth', {}), smooth_stack, 'smooth_stack')
    R['rotation'] = {}
    for (rk, nm), l in rot.items():
        mk = ev & np.isfinite(l)
        m = evaluate(y, g, l, mk, grp_of); R['rotation'][f'{rk}/{nm}'] = m
        print(f'[control] rotated {rk:10s} {nm:10s} {fmt(m)} (subjects={m["n_subj"]})', flush=True)
    y3 = label_shuffle_metric(y, g, base, ev, O['fold'], seed=1); R['checks']['label_shuffle'] = {}
    for nm in ('auto', 'br4', 'br3'):
        m = evaluate(y3, g, out[nm], ev, grp_of); R['checks']['label_shuffle'][nm] = m
        print(f'[check] label shuffle {nm}: AUC={m["AUC"]:.4f} mac={m["subject_macro_bACC"]:.4f} (expect ~.5)', flush=True)
    per = per_subject_bacc(y, g, out['auto'], ev)
    R['per_subject_auto'] = {s: float(v) for s, v in per.items()}
    seed_in = np.full((5, len(y)), np.nan)
    for (k, s), l in inner_seed.items(): seed_in[k, O['order'][s]] = l
    np.savez(HERE / (stem + '_oof.npz'), y=y, g=g, fold=O['fold'], ev=ev, base=base, pos_only=pos_out,
             seed_inner=seed_in, **{f'sel__{c}': out[c] for c in out}, **{f'rot__{rk}__{c}': rot[(rk, c)] for rk, c in rot})
    save(stem, R, tee)


if __name__ == '__main__': main()
