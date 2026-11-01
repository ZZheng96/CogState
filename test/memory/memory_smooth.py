"""Experiment 1: causal EMA directly on the raw ana_bc fused logit (no mean subtraction), real recording order.

Candidates: half-life h in HS windows (0 = raw) + multi-scale EMA LR stack fitted on training subjects.
Per outer fold k the half-life / stack is chosen by inner SGKF(4) OOF of the TRAINING subjects only (inner{k}__ana_bc);
test subjects never contribute labels. Fixed-h results are reported as reference (not selected on test).
Checks: prefix invariance (output on a truncated stream == prefix of full output), test-label permutation
(selection identical when test labels permuted; metrics against permuted labels ~ .5), subject bootstrap vs raw.
Run: python test/memory/memory_smooth.py [--smoke]
"""
import argparse

import numpy as np

from _mem import (BLOCK_S, HERE, HS, apply_smooth, boot, ema, el, evaluate, fmt, fmt_boot, hl_time, load_oof,
                  open_log, per_subject, save, select_smooth, sig, label_shuffle_metric)

BASE = 'ana_bc'


def run_select(O, y, k):
    Li = O[f'inner{k}__{BASE}']
    tr_subs = [s for s in O['subs'] if O['fold_of'][s] != k]
    locs = [O['order'][s] for s in tr_subs]
    assert all(np.isfinite(Li[loc]).all() for loc in locs)
    return select_smooth(Li, locs, y)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); ap.add_argument('--boot', type=int)
    a = ap.parse_args(); nb = a.boot or (100 if a.smoke else 1000)
    stem = 'memory_smooth' + ('_smoke' if a.smoke else ''); tee = open_log(stem)
    O = load_oof(a.smoke); O['fold_of'] = O['D']['fold_of']
    y, g, ev, grp_of = O['y'], O['g'], O['ev'], O['grp_of']
    base = O[f'oof__{BASE}']; assert np.isfinite(base[ev]).all()
    print(f'[start] base={BASE} HS={list(HS)} boot={nb} block={BLOCK_S:.0f}s ' +
          ' '.join(hl_time(h) for h in HS[1:]) + f' {el()}', flush=True)
    R = {'meta': {'base': BASE, 'HS': list(HS), 'step_s': 6.0, 'half_life_time': {h: hl_time(h) for h in HS},
                  'note': 'h>=32 windows (>=192 s) approaches the 300 s R/T block: gains there exploit the '
                          'paradigm temporal autocorrelation (labels constant within 5-min blocks)'},
         'select': {}, 'fixed': {}, 'selected': {}, 'checks': {}}
    out = {m: np.full(len(y), np.nan) for m in ('single', 'stack', 'auto')}
    for k in O['folds']:
        k = int(k)
        sel = run_select(O, y, k)
        R['select'][k] = {kk: v for kk, v in sel.items() if kk != 'stack'}
        print(f'[select f{k}] h={hl_time(sel["h"])} mode={sel["mode"]} stack_inner={sel["stack_inner_macro"]:.4f} '
              'inner ' + ' '.join(f'{h}:{v:.4f}' for h, v in sel['inner_macro'].items()) + f' {el()}', flush=True)
        te_subs = [s for s in O['subs'] if O['fold_of'][s] == k]
        for s in te_subs:
            loc = O['order'][s]; r = apply_smooth(sel, base[loc])
            for m in out: out[m][loc] = r[m]
        if k == int(O['folds'][0]):
            # prefix invariance on a test subject
            s = te_subs[0]; loc = O['order'][s]; cut = len(loc) // 2
            full = apply_smooth(sel, base[loc]); pre = apply_smooth(sel, base[loc][:cut])
            d = max(float(np.max(np.abs(full[m][:cut] - pre[m]))) for m in full)
            assert d < 1e-9, d
            R['checks']['prefix'] = {'subject': s, 'cut': cut, 'max_abs_diff': d}
            print(f'[check] prefix invariance ok subject={s} cut={cut} max|d|={d:.1e}', flush=True)
            # test-label permutation: selection must not change
            y2 = label_shuffle_metric(y, g, base, np.isin(g, te_subs), O['fold'])
            sel2 = run_select(O, y2, k)
            same = sel2['h'] == sel['h'] and sel2['mode'] == sel['mode'] and \
                np.allclose(sel2['stack_coef'], sel['stack_coef'])
            assert same; R['checks']['test_label_perm_selection_identical'] = True
            print(f'[check] test-label permutation -> identical selection {el()}', flush=True)
    for m in out: assert np.isfinite(out[m][ev]).all()
    fixed = {h: per_subject(lambda l, h=h: ema(l, h), base, O['order']) for h in HS}

    def report(name, l, store):
        m = evaluate(y, g, l, ev, grp_of); bb = boot(sig(l), sig(base), y, g, ev, nb)
        store[name] = {'metrics': m, 'boot_vs_raw': bb}
        print(f'[result] {name:12s} {fmt(m)} | vs raw {fmt_boot(bb)}', flush=True)

    for h in HS: report(f'h{h}', fixed[h], R['fixed']); R['fixed'][f'h{h}']['time'] = hl_time(h)
    for mname in out: report(mname, out[mname], R['selected'])
    # label permutation on all evaluated subjects: metrics of the selected outputs vs permuted labels
    y3 = label_shuffle_metric(y, g, base, ev, O['fold'], seed=1); R['checks']['label_shuffle'] = {}
    for mname in ('auto', 'single', 'stack'):
        m = evaluate(y3, g, out[mname], ev, grp_of)
        R['checks']['label_shuffle'][mname] = m
        print(f'[check] label shuffle {mname}: AUC={m["AUC"]:.4f} mac={m["subject_macro_bACC"]:.4f} (expect ~.5)',
              flush=True)
    np.savez(HERE / (stem + '_oof.npz'),
             y=y, g=g, fold=O['fold'], ev=ev, base=base, **{f'sel__{m}': out[m] for m in out},
             **{f'h{h}': fixed[h] for h in HS})
    save(stem, R, tee)


if __name__ == '__main__': main()
