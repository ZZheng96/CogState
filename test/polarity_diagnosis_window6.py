"""Subject-disjoint polarity diagnostics; oracle metrics are never model selection."""
import hashlib
import json
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'test'
SEED = 20261029


def load_data():
    with np.load(OUT / 'highdim_deep_features.npz', allow_pickle=False) as hd:
        y, groups = hd['y'].astype(int), hd['groups'].astype(str)
        assert y.ndim == groups.ndim == 1 and len(y) == len(groups)
        assert np.array_equal(np.unique(y), [0, 1])
        if 'f' in hd.files and 'e' in hd.files:
            f, e = hd['f'], hd['e']
            source = 'highdim_deep_features.npz f/e'
        else:
            # Existing cache has X/raw/y/groups, not f/e. Match database f/e
            # to the exact same six-second windows; raw is only an identity key.
            raw = hd['raw']
            assert hd['X'].shape == (len(y), 858)
            assert raw.shape == (len(y), 1500)
            lookup = defaultdict(deque)
            for i, row in enumerate(raw):
                lookup[(groups[i], int(y[i]), hashlib.sha256(row.astype(np.float32).tobytes()).digest())].append(i)
            indices, fs, es = [], [], []
            paths = sorted((ROOT / 'database' / 'windows').glob('*/*.npz'))
            assert paths, 'No database f/e source available'
            for path in paths:
                with np.load(path, allow_pickle=False) as data:
                    n = len(data['label'])
                    assert data['f'].ndim == data['e'].ndim == 2
                    assert data['f'].shape[0] == data['e'].shape[0] == n
                    assert data['x'].shape == (n, 1500) and data['subject'].shape == (n,)
                    fs.append(data['f']); es.append(data['e'])
                    for row, label, subject in zip(data['x'], data['label'], data['subject']):
                        key = (str(subject), int(label), hashlib.sha256(row.astype(np.float32).tobytes()).digest())
                        assert lookup[key], f'Unmatched window in {path}'
                        indices.append(lookup[key].popleft())
            indices = np.asarray(indices)
            assert len(indices) == len(y) and len(np.unique(indices)) == len(y)
            assert all(not queue for queue in lookup.values())
            order = np.argsort(indices)
            f, e = np.vstack(fs)[order], np.vstack(es)[order]
            source = ('database/windows/*/*.npz f/e, bijectively aligned to highdim_deep_features.npz '
                      'by subject, label and float32 waveform SHA256; raw/X never model inputs')
    assert f.ndim == e.ndim == 2 and f.shape[0] == e.shape[0] == len(y)
    x = np.hstack([f, e]).astype(np.float64)
    assert np.isfinite(x).all()
    return x, y, groups, {'source': source, 'f_shape': list(f.shape), 'e_shape': list(e.shape), 'input_shape': list(x.shape)}


def model():
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, solver='lbfgs', max_iter=3000, tol=1e-5, random_state=0))


def auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None


def metrics(y, p, groups):
    per_subject = []
    for subject in np.unique(groups):
        keep = groups == subject
        a = auc(y[keep], p[keep])
        per_subject.append({'subject': str(subject), 'n': int(keep.sum()), 'auc': a,
                            'bacc': float(balanced_accuracy_score(y[keep], p[keep] >= .5)),
                            'oracle_only_orientation_invariant_auc': None if a is None else max(a, 1-a)})
    valid = [r for r in per_subject if r['auc'] is not None]
    oriented = [r['oracle_only_orientation_invariant_auc'] for r in valid]
    return {'oof_auc': auc(y, p), 'oof_bacc': float(balanced_accuracy_score(y, p >= .5)),
            'subject_macro_auc': float(np.mean([r['auc'] for r in valid])),
            'subject_macro_bacc': float(np.mean([r['bacc'] for r in per_subject])),
            'subjects_auc_below_0_5_count': sum(r['auc'] < .5 for r in valid),
            'subjects_auc_below_0_5_fraction': float(np.mean([r['auc'] < .5 for r in valid])),
            'auc_eligible_subjects': len(valid), 'single_class_subjects': len(per_subject)-len(valid),
            'oracle_only': {'mean_max_auc_1_minus_auc': float(np.mean(oriented)),
                            'max_max_auc_1_minus_auc': float(np.max(oriented)),
                            'warning': 'Uses held-out labels separately for each subject; diagnostic upper bound, NOT deployable.'},
            'per_subject': per_subject}


def main():
    started = time.perf_counter()
    x, y, groups, source = load_data()
    subjects = np.unique(groups)
    # Register random subject signs before any model fit. Never pass stress_y to training.
    signs = np.random.default_rng(SEED).choice(np.array([-1, 1]), size=len(subjects))
    q_by_subject = dict(zip(subjects, signs))
    q = np.array([q_by_subject[s] for s in groups])
    stress_y = np.where(q == -1, 1-y, y)
    print(f'[data] shape={x.shape} subjects={len(subjects)} CPU Logistic; GPU not used', flush=True)
    print(f'[preregister] seed={SEED} flipped_subjects={int((signs == -1).sum())}; fixed C=1, threshold=.5', flush=True)
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(x, y, groups))
    oof = np.full(len(y), np.nan)
    polarity_oof = np.full(len(y), np.nan)
    fold_id = np.zeros(len(y), dtype=int)
    rows = []
    with threadpool_limits(limits=2):
        for number, (train, test) in enumerate(folds, 1):
            tick = time.perf_counter()
            assert not set(groups[train]) & set(groups[test])
            print(f'[fold {number}/5] baseline train={len(train)} test={len(test)}', flush=True)
            baseline = model().fit(x[train], y[train])
            oof[test] = baseline.predict_proba(x[test])[:, 1]
            fold_id[test] = number
            # Estimate each TRAIN subject's direction from subject-held-out inner
            # scores. Each inner scaler sees inner training subjects only.
            inner_p = np.full(len(train), np.nan)
            inner = StratifiedGroupKFold(4, shuffle=True, random_state=0)
            for j, (it, iv) in enumerate(inner.split(x[train], y[train], groups[train]), 1):
                assert not set(groups[train[it]]) & set(groups[train[iv]])
                print(f'[fold {number}/5] polarity inner {j}/4', flush=True)
                fitted = model().fit(x[train[it]], y[train[it]])
                inner_p[iv] = fitted.predict_proba(x[train[iv]])[:, 1]
            assert np.isfinite(inner_p).all()
            aligned_y = y[train].copy()
            training_directions = {}
            for subject in np.unique(groups[train]):
                keep = groups[train] == subject
                a = auc(y[train][keep], inner_p[keep])
                direction = -1 if a is not None and a < .5 else 1
                training_directions[str(subject)] = {'inner_oof_auc': a, 'direction': direction}
                if direction == -1:
                    aligned_y[keep] = 1-aligned_y[keep]
            polarity = model().fit(x[train], aligned_y)
            train_p = polarity.predict_proba(x[train])[:, 1]
            global_direction = -1 if auc(y[train], train_p) < .5 else 1
            p = polarity.predict_proba(x[test])[:, 1]
            polarity_oof[test] = p if global_direction == 1 else 1-p
            row = {'fold': number, 'train_subjects': sorted(set(groups[train])),
                   'test_subjects': sorted(set(groups[test])), 'training_directions': training_directions,
                   'fixed_global_test_direction': global_direction,
                   'baseline_iterations': int(baseline[-1].n_iter_[0]),
                   'polarity_iterations': int(polarity[-1].n_iter_[0]),
                   'baseline_test': metrics(y[test], oof[test], groups[test]),
                   'polarity_test': metrics(y[test], polarity_oof[test], groups[test]),
                   'seconds': time.perf_counter()-tick}
            rows.append(row)
            print(f'[fold {number}/5] done {row["seconds"]:.1f}s AUC={row["baseline_test"]["oof_auc"]:.4f} '
                  f'polarity_AUC={row["polarity_test"]["oof_auc"]:.4f}', flush=True)
    assert np.isfinite(oof).all() and np.isfinite(polarity_oof).all() and (fold_id > 0).all()
    assert all(len(np.unique(fold_id[groups == s])) == 1 for s in subjects)
    result = {'protocol': {'outer_cv': 'StratifiedGroupKFold(5, shuffle=True, random_state=0)',
                          'model': 'StandardScaler + LogisticRegression(C=1, lbfgs, max_iter=3000, tol=1e-5)',
                          'scaling': 'Fit only on training fold; inner diagnostic scalers fit only inner training fold.',
                          'threshold': .5, 'selection': 'No hyperparameter selection or early stopping (Logistic).',
                          'input': source, 'device': 'CPU', 'gpu_used': False,
                          'stress_seed': SEED, 'stress_rng': 'numpy.default_rng; sorted unique subjects; choice([-1,1])',
                          'stress_usage': 'Flip evaluation labels only; never used for training, tuning, orientation or selection.',
                          'polarity_control': 'DIAGNOSTIC, not deployment claim: train-subject signs estimated from 4-fold subject-disjoint inner OOF AUC; refit with aligned TRAIN labels. One global output direction fixed from original training-label AUC; never uses test labels/statistics for inference. Training sign estimation still uses labels and may overfit.',
                          'oracle_usage': 'Evaluation only, per-subject max(AUC,1-AUC); never alters model predictions.'},
              'stress_subject_q': {str(s): int(q_by_subject[s]) for s in subjects},
              'baseline': {'normal': metrics(y, oof, groups), 'random_flip_stress': metrics(stress_y, oof, groups)},
              'polarity_train_only': {'normal': metrics(y, polarity_oof, groups),
                                      'random_flip_stress': metrics(stress_y, polarity_oof, groups)},
              'folds': rows, 'runtime_seconds': time.perf_counter()-started}
    for name in ('baseline', 'polarity_train_only'):
        a = result[name]['normal']['oracle_only']['mean_max_auc_1_minus_auc']
        b = result[name]['random_flip_stress']['oracle_only']['mean_max_auc_1_minus_auc']
        assert np.isclose(a, b), 'Oracle AUC should be invariant to per-subject label flips'
    np.savez_compressed(OUT / 'polarity_diagnosis_window6_oof.npz', y=y, groups=groups, fold_id=fold_id,
                        baseline=oof, polarity_train_only=polarity_oof, stress_y=stress_y, stress_q=q)
    with (OUT / 'polarity_diagnosis_window6_results.json').open('w', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    for name in ('baseline', 'polarity_train_only'):
        for condition in ('normal', 'random_flip_stress'):
            m = result[name][condition]
            print(f'[result] {name}/{condition} AUC={m["oof_auc"]:.6f} bACC={m["oof_bacc"]:.6f} '
                  f'macro_AUC={m["subject_macro_auc"]:.6f} macro_bACC={m["subject_macro_bacc"]:.6f} '
                  f'below_chance={m["subjects_auc_below_0_5_fraction"]:.6f} '
                  f'oracle_only={m["oracle_only"]["mean_max_auc_1_minus_auc"]:.6f}', flush=True)
    print(f'[done] seconds={result["runtime_seconds"]:.1f}; GPU used=False', flush=True)


if __name__ == '__main__':
    main()
