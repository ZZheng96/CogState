"""Fixed nonlinear kernel-distribution conditioned online state mapping."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.signal import welch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
PRIOR = 32.
BANDS = ((1, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 45), (52, 100))


def features(raw):
    assert raw.shape[1] == 1500
    f, p = welch(raw.astype(float), fs=250, nperseg=250, noverlap=125, axis=1)
    bp = np.stack([p[:, (f >= a) & (f < b)].sum(1) + 1e-20 for a, b in BANDS], 1)
    q = p[:, (f >= 1) & (f < 100)] + 1e-20
    total = q.sum(1)
    q /= total[:, None]
    return np.column_stack([np.log(bp), bp / total[:, None],
                            -(q * np.log(q)).sum(1), np.exp(np.log(q).mean(1)) / q.mean(1)])


def transport(x, groups, target):
    """Kernel mean embedding of prior personal observations; no count feature."""
    out = np.empty((len(x), 3 * x.shape[1]))
    for subject in np.unique(groups):
        n = PRIOR
        total = PRIOR * target.copy()
        for i in np.flatnonzero(groups == subject):
            history = total / n
            out[i] = np.r_[x[i], x[i] - history, x[i] * history]
            total += x[i]
            n += 1
    return out


def metrics(y, g, p, mask):
    m = mask & np.isfinite(p)
    per = [balanced_accuracy_score(y[q], p[q] >= .5)
           for s in np.unique(g[m]) if np.unique(y[q := m & (g == s)]).size == 2]
    return {'n': int(m.sum()), 'AUC': float(roc_auc_score(y[m], p[m])) if np.unique(y[m]).size == 2 else None,
            'subject_macro_bACC': float(np.mean(per)) if per else None,
            'micro_bACC': float(balanced_accuracy_score(y[m], p[m] >= .5))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    start = time.perf_counter()
    d = np.load(OUT / 'highdim_deep_features.npz')
    y, g, raw = d['y'].astype(int), d['groups'], d['raw']
    x = features(raw)
    print(f'[start] rows={len(y)} subjects={len(np.unique(g))} features={x.shape[1]} smoke={args.smoke}', flush=True)
    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(x, y, g))
    scores = {k: np.full(len(y), np.nan) for k in ('kernel_history', 'no_history')}
    fold_id = np.full(len(y), -1)
    checks = []
    with threadpool_limits(limits=4):
        for fold, (tr, te) in enumerate(splits):
            assert not set(g[tr]) & set(g[te])
            if args.smoke:
                tr = tr[np.isin(g[tr], np.unique(g[tr])[:5])]
                te = te[g[te] == np.unique(g[te])[0]]
            scaler = StandardScaler().fit(x[tr])
            ztr, zte = scaler.transform(x[tr]), scaler.transform(x[te])
            kernel = RBFSampler(gamma=1 / x.shape[1], n_components=64, random_state=0).fit(ztr)
            ztr, zte = kernel.transform(ztr), kernel.transform(zte)
            target = ztr.mean(0)
            fold_id[te] = fold
            for name in scores:
                if name == 'no_history':
                    a, b = ztr, zte
                else:
                    a = transport(ztr, g[tr], target)
                    b = transport(zte, g[te], target)
                    for s in np.unique(g[te]):
                        ii = np.flatnonzero(g[te] == s)
                        cut = len(ii) // 2
                        altered = zte[ii].copy()
                        altered[cut:] += 100
                        check = transport(altered, g[te][ii], target)
                        assert np.array_equal(check[:cut], b[ii][:cut])
                        assert np.allclose(transport(zte[ii][:cut], g[te][ii][:cut], target), b[ii][:cut], rtol=0, atol=0)
                    checks.append({'fold': fold, 'variant': name, 'future_and_prefix_invariance': True})
                model = HistGradientBoostingClassifier(max_iter=150, max_depth=3, learning_rate=.06,
                                                       l2_regularization=1., random_state=0)
                scores[name][te] = model.fit(a, y[tr]).predict_proba(b)[:, 1]
                print(f'[progress] fold={fold+1}/5 method={name} elapsed={time.perf_counter()-start:.1f}s', flush=True)
            if args.smoke:
                break
    mask = fold_id >= 0
    switching = np.zeros(len(y), bool)
    history_mask = np.zeros(len(y), bool)
    for s in np.unique(g):
        ii = np.flatnonzero(g == s)
        # Only reporting uses labels/position: first five observations after an actual switch.
        cuts = np.flatnonzero(y[ii][1:] != y[ii][:-1]) + 1
        for cut in cuts:
            switching[ii[cut:cut+5]] = True
        history_mask[ii[120:]] = True
    result = {'meta': {'smoke': args.smoke, 'prior': PRIOR, 'kernel_components': 64, 'features': 16,
                       'threshold': .5, 'history': 'all previous unlabeled windows, predict before update',
                       'alternation_assumption': False, 'hyperparameter_search': False,
                       'switch_definition': 'first five windows after actual label change; reporting only'},
              'checks': checks, 'methods': {}}
    for name, p in scores.items():
        result['methods'][name] = {k: metrics(y, g, p, mask & m) for k, m in
                                  [('all', np.ones(len(y), bool)), ('history_comparable', history_mask),
                                   ('switching', switching), ('steady', ~switching)]}
        print('[result]', name, json.dumps(result['methods'][name]), flush=True)
    baseline = np.load(OUT / 'merge_best_window6_oof.npz')
    assert np.array_equal(y, baseline['y']) and np.array_equal(g, baseline['groups'])
    from scipy.special import expit
    result['strict_reference'] = {k: metrics(y, g, expit(baseline['combo__C4_bb_resnet_eq']), mask & m)
                                  for k, m in [('all', np.ones(len(y), bool)), ('history_comparable', history_mask),
                                               ('switching', switching), ('steady', ~switching)]}
    result['seconds'] = time.perf_counter() - start
    stem = 'causal_kernel_state_mapping' + ('_smoke' if args.smoke else '')
    np.savez_compressed(OUT / (stem + '_oof.npz'), y=y, groups=g, fold_id=fold_id,
                        eval_mask=mask, switching=switching, history_mask=history_mask, **scores)
    (OUT / (stem + '_results.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'[done] elapsed={result["seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    main()
