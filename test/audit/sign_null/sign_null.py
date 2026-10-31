"""Sign-consistency null for the anchor-direction diagnostic (depositioned_anchor_direction.py, diagnostics block).

Observed statistic (reproduced exactly, asserted against depositioned_anchor_direction.json): ZZ (889) standardised on
fold-0 training rows; per subject s and feature j, Cohen's d_sj (task - relax; cohen_d of the anchor script); global
median m_j over all 57 subjects; group consistency C_G = mean_{s in G, j} [sign d_sj == sign m_j].
Nulls:
  label_perm   within EVERY subject permute the labels (N_PERM permutations, seed default_rng([1000, i]) per perm,
               spawned per subject), recompute d, global median and C_G for each group (+ 'all', + per subject).
               p_low = P(C_null <= C_obs) (is the group LESS consistent than chance?), p_high = P(C_null >= C_obs).
  random_subset (circularity check, no labels permuted): C of random subject subsets of the same size as each group
               (N_PERM draws, rng 2000) vs the observed group value -> is the low group special among subsets,
               given it was defined post-hoc from pipeline performance?
  split_half   group-free: random half split A/B of subjects (N_SPLIT splits, rng 3000); median from A only,
               consistency of B subjects vs that median; observed mean over splits vs label_perm null of the same
               statistic (perm i uses split i % N_SPLIT). Avoids both the post-hoc grouping and self-inclusion in
               the median.
Checks: observed C_G == anchor json; determinism (two observed computations identical).
Run: python test/audit/sign_null/sign_null.py [--smoke]   (smoke: 20 permutations, 20 splits)
"""
import argparse, json, sys, time, warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import GROUPS, TEST, el, load_all, open_log, save, zz_fold  # noqa: E402
sys.path.insert(0, str(TEST))
from depositioned_anchor_direction import cohen_d  # noqa: E402


def all_d(Z, ylist, locs):
    return np.stack([cohen_d(Z[l], yy) for l, yy in zip(locs, ylist)])


def cons_stats(Dm, subs, grp_of):
    med = np.sign(np.median(Dm, 0)); agree = np.sign(Dm) == med
    per = agree.mean(1)
    out = {'all': float(agree.mean())}
    for G in GROUPS:
        ix = [i for i, s in enumerate(subs) if grp_of[s] == G]
        out[G] = float(agree[ix].mean())
    return out, per


def split_stat(Dm, splits):
    v = []
    for A, B in splits:
        med = np.sign(np.median(Dm[A], 0)); v.append(float((np.sign(Dm[B]) == med).mean()))
    return float(np.mean(v))


def interval(obs, null):
    null = np.asarray(null)
    return {'observed': obs, 'null_mean': float(null.mean()), 'null_sd': float(null.std()),
            'null_95': [float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))],
            'p_low': float((1 + (null <= obs).sum()) / (1 + len(null))),
            'p_high': float((1 + (null >= obs).sum()) / (1 + len(null)))}


def main():
    warnings.filterwarnings('ignore')
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); args = ap.parse_args()
    stem = 'sign_null' + ('_smoke' if args.smoke else '')
    tee = open_log(HERE, stem)
    D = load_all(); y, g, fold, subs, grp_of = D['y'], D['g'], D['fold'], D['subs'], D['grp_of']
    print('[check] fold == npz fold', flush=True)
    n_perm = 20 if args.smoke else 200; n_split = 20 if args.smoke else 200
    Z = zz_fold(D, np.flatnonzero(fold != 0))
    locs = [np.flatnonzero(g == s) for s in subs]; ys = [y[l] for l in locs]
    Dobs = all_d(Z, ys, locs); obs, per_obs = cons_stats(Dobs, subs, grp_of)
    Dobs2 = all_d(Z, ys, locs); assert np.array_equal(Dobs, Dobs2), 'determinism'
    ref = json.loads((TEST / 'depositioned_anchor_direction.json').read_text(encoding='utf-8'))['diagnostics']
    for G in GROUPS:
        assert abs(ref[G]['sign_consistency_vs_global_median'] - obs[G]) < 1e-12, (G, obs[G])
    print(f'[obs] reproduces anchor json: ' + ' '.join(f'{k}={v:.4f}' for k, v in obs.items()) + f' {el()}', flush=True)
    sizes = {G: sum(grp_of[s] == G for s in subs) for G in GROUPS}
    r2 = np.random.default_rng(3000); half = len(subs) // 2
    splits = []
    for _ in range(n_split):
        p = r2.permutation(len(subs)); splits.append((p[:half], p[half:]))
    split_obs = split_stat(Dobs, splits)

    null = {k: [] for k in obs}; null_per = []; null_split = []
    tp = time.perf_counter()
    for i in range(n_perm):
        sub_rngs = np.random.default_rng([1000, i]).spawn(len(subs))
        yp = [r.permutation(yy) for r, yy in zip(sub_rngs, ys)]
        Dp = all_d(Z, yp, locs); c, per = cons_stats(Dp, subs, grp_of)
        for k in c: null[k].append(c[k])
        null_per.append(per)
        A, B = splits[i % n_split]; med = np.sign(np.median(Dp[A], 0))
        null_split.append(float((np.sign(Dp[B]) == med).mean()))
        if (i + 1) % 20 == 0 or i + 1 == n_perm:
            print(f'[perm {i + 1}/{n_perm}] perm_s={time.perf_counter() - tp:.1f} {el()}', flush=True)
    null_per = np.stack(null_per)

    r1 = np.random.default_rng(2000); agree = np.sign(Dobs) == np.sign(np.median(Dobs, 0))
    subset = {G: [float(agree[r1.choice(len(subs), sizes[G], replace=False)].mean()) for _ in range(n_perm)]
              for G in GROUPS}
    res = {'meta': {'n_perm': n_perm, 'n_split': n_split, 'smoke': args.smoke, 'group_sizes': sizes,
                    'zz': 'fold-0 train-row impute+scale (as anchor diagnostic)',
                    'note': 'groups high/mid/low were defined post-hoc from pipeline performance (circular); '
                            'random_subset and split_half address this'},
           'checks': {'reproduces_anchor_json': 'ok', 'determinism': 'ok', 'fold_eq_npz': 'ok'},
           'label_perm': {k: interval(obs[k], null[k]) for k in obs},
           'random_subset': {G: interval(obs[G], subset[G]) for G in GROUPS},
           'split_half': interval(split_obs, null_split), 'per_subject': {}}
    for j, s in enumerate(subs):
        q = interval(float(per_obs[j]), null_per[:, j]); q['group'] = grp_of[s]; res['per_subject'][s] = q
    for G in GROUPS:
        ss = [s for s in subs if grp_of[s] == G]
        res['label_perm'][G]['n_subjects_p_low_lt_.05'] = int(sum(res['per_subject'][s]['p_low'] < .05 for s in ss))
        res['label_perm'][G]['n_subjects_p_high_lt_.05'] = int(sum(res['per_subject'][s]['p_high'] < .05 for s in ss))
    for k, q in res['label_perm'].items():
        print(f'[label_perm {k}] obs={q["observed"]:.4f} null={q["null_mean"]:.4f} 95%=[{q["null_95"][0]:.4f},'
              f'{q["null_95"][1]:.4f}] p_low={q["p_low"]:.4f} p_high={q["p_high"]:.4f}'
              + (f' subj p_low<.05={q["n_subjects_p_low_lt_.05"]} p_high<.05={q["n_subjects_p_high_lt_.05"]}'
                 if k in GROUPS else ''), flush=True)
    for G, q in res['random_subset'].items():
        print(f'[random_subset {G} n={sizes[G]}] obs={q["observed"]:.4f} subset mean={q["null_mean"]:.4f} '
              f'95%=[{q["null_95"][0]:.4f},{q["null_95"][1]:.4f}] p_low={q["p_low"]:.4f}', flush=True)
    q = res['split_half']
    print(f'[split_half] obs={q["observed"]:.4f} null={q["null_mean"]:.4f} 95%=[{q["null_95"][0]:.4f},'
          f'{q["null_95"][1]:.4f}] p_high={q["p_high"]:.4f}', flush=True)
    res['seconds'] = time.perf_counter() - __import__('common').T0
    save(HERE, stem, res); print(f'[done] {el()} -> {stem}.json', flush=True)
    sys.stdout = tee.stream; tee.f.close()


if __name__ == '__main__':
    main()
