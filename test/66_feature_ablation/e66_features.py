"""Fixed six-fold feature ablation. Never run before event-audit clearance and E65 completion."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TESTS = HERE.parent
spec = importlib.util.spec_from_file_location('e63_features_reference', TESTS / '63_layer_ablation/e63_layers.py')
E63 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E63)
L, E42 = E63.L, E63.E61.E42
TARGETS = ('004', '009', '010', '018', '034', '050')
QUALITY = ('amplitude_max', 'peak_to_peak', 'rms', 'flat_fraction', 'diff_energy',
           'spike_fraction', 'low_freq_drift', 'high_freq_ratio', 'nan_fraction')
STAGES = ('raw', 'dt1200', 'dt1200_hmm300')
PARTS = ('full', 'early', 'late')
metric, evaluate, stages = E63.metric, E63.evaluate, E63.stages


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def save_json(path, obj):
    tmp = path.with_suffix('.json.tmp')
    E63.save_json(tmp, obj)
    tmp.replace(path)


def selections(names, emg):
    assert len(names) == len(set(names)) and set(QUALITY) <= set(names)
    excluded = set(emg) | set(QUALITY)
    indices = {
        'no_emg_quality': [i for i, name in enumerate(names) if name not in excluded],
        'relative_only': [i for i, name in enumerate(names)
                          if name.startswith(('dwt_relative_', 'stft_relative_'))],
    }
    assert tuple(E42.SCALES) == (60, 120, 300, 600, 99999)
    result = {}
    for candidate, keep in indices.items():
        assert keep
        result[candidate] = {
            'base_indices': keep, 'base_names': [names[i] for i in keep],
            'excluded_indices': [i for i in range(len(names)) if i not in keep],
            'excluded_names': [n for i, n in enumerate(names) if i not in keep],
            'expanded_columns': [{'scale': w, 'original_index': j * len(names) + i,
                                  'base_index': i, 'name': names[i]}
                                 for j, w in enumerate(E42.SCALES) for i in keep],
        }
    return result


def prepare():
    manifest_path = ROOT / 'database/manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    names = manifest['feature_names'] + manifest['emg_feature_names']
    choices = selections(names, manifest['emg_feature_names'])
    subs = sorted(p.stem[4:] for p in L.SLID.glob('sub-*.npz'))
    assert len(subs) == 57 and set(TARGETS) <= set(subs)
    hashes, metadata = {}, {}
    for q in subs:
        path = L.SLID / f'sub-{q}.npz'
        hashes[q] = sha256(path)
        with np.load(path, allow_pickle=False) as z:
            assert z['F'].ndim == 2 and z['F'].shape[1] == len(names), q
            order = np.argsort(z['t'])
            t, y, b = z['t'][order], z['y'][order].astype(int), z['blk'][order].astype(int)
            assert len(t) == len(y) == len(b) == z['F'].shape[0]
            assert np.all(np.diff(t) > 0) and set(np.unique(y)) <= {0, 1}
            metadata[q] = dict(t=t, y=y, block=b, source_row=order)
    baseline, baseline_hashes = {}, {}
    for q in TARGETS:
        tr = [p for p in subs if p != q]
        components = E63.HERE / f'fold_{q}_components.npz'
        scores = E63.HERE / f'fold_{q}_scores.npz'
        for path in (components, scores):
            baseline_hashes[str(path.relative_to(ROOT))] = sha256(path)
        with np.load(components, allow_pickle=False) as z:
            assert str(z['target']) == q and list(z['train_subjects']) == tr
            assert json.loads(str(z['data_hashes'])) == hashes
            for key, expected in [('t', metadata[q]['t']), ('y', metadata[q]['y']),
                                  ('block', metadata[q]['block'])]:
                np.testing.assert_array_equal(z[f'{key}_{q}'], expected)
            raw = z[f'lr_scaled_{q}'] + z[f'mlp_scaled_{q}']
            np.testing.assert_array_equal(raw, z[f'raw_{q}'])
        baseline[q] = stages(raw)
        with np.load(scores, allow_pickle=False) as z:
            for key in ('t', 'y', 'block'):
                np.testing.assert_array_equal(z[key], metadata[q][key])
            for stage in STAGES:
                np.testing.assert_array_equal(z[f'w0.50_{stage}'], baseline[q][stage])
    source_paths = [Path(__file__), Path(E63.__file__), Path(E63.E61.__file__),
                    Path(E42.__file__), Path(E42.E41.__file__), Path(L.__file__),
                    TESTS / '17_compliant/multiscale.py']
    protocol = {
        'version': 'e66-fixed-features-v1', 'subjects': subs, 'targets': list(TARGETS),
        'train_subjects': {q: [p for p in subs if p != q] for q in TARGETS},
        'training_subject_count': 56, 'feature_names': names,
        'emg_feature_names': manifest['emg_feature_names'], 'quality_exact_names': list(QUALITY),
        'feature_scales': list(E42.SCALES), 'candidates': choices,
        'selection': 'same original column order at each scale; columnwise L.feats unchanged',
        'LR': {'C': .001, 'iters': 200},
        'MLP': {'seeds': [0, 1, 2], 'hid': 128, 'ep': 30, 'wd': .001, 'drop': .3,
                'input_dropout': .2, 'lr': .001, 'batch_size': 512},
        'scale': 'pooled 56 training-subject output torch.std correction=1',
        'fusion': 'LR/std_lr + mean(MLP seeds 0,1,2)/std_mlp; fixed equal weight',
        'threshold': 0, 'positive_label': 1, 'postprocess': 'E63.stages unchanged, continuous sequence',
        'regions': {'full': 'all', 'early': 'block<=2', 'late': 'block>=4'},
        'single_class': 'E63.metric: null AUC/bacc, F1 zero_division=0',
        'baseline': 'cached E63 w0.50; no retraining',
        'scope': 'six diagnostic folds only; both predefined candidates reported; no winner/generalization claim',
        'prohibitions': 'no weight selection, hyperparameter tuning, sign flipping or target-label training',
        'label_use': 'target labels/blocks only for offline evaluation',
        'data_sha256': hashes, 'manifest_sha256': sha256(manifest_path),
        'baseline_sha256': baseline_hashes,
        'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in source_paths},
        'versions': {'python': platform.python_version(),
                     **{k: importlib.metadata.version(k) for k in ('numpy', 'torch', 'scikit-learn')},
                     'cuda': torch.version.cuda},
        'launch_attestation': 'operator confirmed E65 finished and event audit found no major label issue',
    }
    digest = fingerprint(protocol)
    path = HERE / 'protocol.json'
    if path.exists():
        old = json.loads(path.read_text(encoding='utf-8'))
        assert old['fingerprint'] == fingerprint(old['protocol']) == digest, 'Protocol changed; refuse resume'
    else:
        assert not list(HERE.glob('fold_*.npz')), 'Orphan fold cache without protocol'
        save_json(path, {'fingerprint': digest, 'protocol': protocol})
    return subs, choices, metadata, baseline, protocol, digest


def train_fold(F, D, subs, q):
    tr = [p for p in subs if p != q]
    assert q not in tr and len(tr) == 56
    X = torch.cat([F[p] for p in tr])
    y = np.concatenate([D[p][1] for p in tr])
    lr = L.fit_lr(X, y, .001)
    mlps = [E63.fit_mlp(X, y, seed=s) for s in range(3)]
    with torch.no_grad():
        a = {p: lr(F[p]) for p in subs}
        m = {p: torch.stack([f(F[p]) for f in mlps]).mean(0) for p in subs}
        sl = float(torch.cat([a[p] for p in tr]).std())
        sm = float(torch.cat([m[p] for p in tr]).std())
        assert np.isfinite([sl, sm]).all() and sl > 0 and sm > 0
        arrays = {'lr': a[q].cpu().numpy(), 'mlp': m[q].cpu().numpy(),
                  'std_lr': sl, 'std_mlp': sm,
                  'lr_scaled': (a[q] / sl).cpu().numpy(), 'mlp_scaled': (m[q] / sm).cpu().numpy()}
    del X, lr, mlps, a, m
    return arrays


def aggregate(folds):
    result = {}
    for name in ('baseline', 'no_emg_quality', 'relative_only'):
        result[name] = {}
        for stage in STAGES:
            result[name][stage] = {}
            for part in PARTS:
                group = {}
                for k in E63.METRICS:
                    values = [r['metrics'][name][stage][part][k] for r in folds]
                    valid = [v for v in values if v is not None]
                    group[k] = {'macro': float(np.mean(valid)) if valid else None, 'n_subjects': len(valid)}
                result[name][stage][part] = group
    deltas = {name: {stage: {part: {
        k: (result[name][stage][part][k]['macro'] - result['baseline'][stage][part][k]['macro']
            if result[name][stage][part][k]['macro'] is not None
            and result['baseline'][stage][part][k]['macro'] is not None else None)
        for k in E63.METRICS} for part in PARTS} for stage in STAGES}
        for name in ('no_emg_quality', 'relative_only')}
    return result, deltas


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--event-audit-cleared', action='store_true')
    ap.add_argument('--e65-finished', action='store_true')
    args = ap.parse_args()
    if not (args.event_audit_cleared and args.e65_finished):
        ap.error('Training blocked: requires event-audit clearance and E65 completion attestations')
    subs, choices, metadata, baseline, protocol, digest = prepare()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU training fallback')
    loaded_subs, D = L.load()
    assert loaded_subs == subs
    for q in subs:
        assert D[q][0].shape[1] == len(protocol['feature_names'])
        np.testing.assert_array_equal(D[q][1], metadata[q]['y'])
        np.testing.assert_array_equal(D[q][2], metadata[q]['block'])
    folds = {q: {'target': q, 'train_subjects': [p for p in subs if p != q], 'fingerprint': digest,
                 'metrics': {'baseline': {s: evaluate(metadata[q]['y'], metadata[q]['block'], v)
                                          for s, v in baseline[q].items()}}, 'paired_deltas': {}}
             for q in TARGETS}
    for name, choice in choices.items():
        F = None
        for q in TARGETS:
            path = HERE / f'fold_{q}_{name}.npz'
            tr = folds[q]['train_subjects']
            if path.exists():
                with np.load(path, allow_pickle=False) as z:
                    assert str(z['fingerprint']) == digest and str(z['candidate']) == name
                    assert str(z['target']) == q and list(z['train_subjects']) == tr
                    arrays = {k: z[k] for k in z.files}
                for key in ('t', 'y', 'block', 'source_row'):
                    np.testing.assert_array_equal(arrays[key], metadata[q][key])
            else:
                if F is None:
                    selected = {p: (D[p][0][:, choice['base_indices']], D[p][1], D[p][2]) for p in subs}
                    F = E42.build_features(subs, selected)
                    del selected
                    assert all(v.is_cuda and v.shape[1] == len(choice['expanded_columns']) for v in F.values())
                arrays = train_fold(F, D, subs, q)
                gc.collect()
                torch.cuda.empty_cache()
                arrays.update(stages(arrays['lr_scaled'] + arrays['mlp_scaled']))
                arrays.update(metadata[q], fingerprint=np.array(digest), target=np.array(q),
                              candidate=np.array(name), train_subjects=np.array(tr))
                tmp = path.with_suffix('.tmp.npz')
                np.savez_compressed(tmp, **arrays)
                tmp.replace(path)
            scores = stages(arrays['lr_scaled'] + arrays['mlp_scaled'])
            for stage, score in scores.items():
                assert score.shape == metadata[q]['y'].shape and np.isfinite(score).all()
                np.testing.assert_array_equal(score, arrays[stage])
            metrics = {s: evaluate(metadata[q]['y'], metadata[q]['block'], v) for s, v in scores.items()}
            folds[q]['metrics'][name] = metrics
            delta = {s: {part: {k: (metrics[s][part][k] - folds[q]['metrics']['baseline'][s][part][k]
                                    if metrics[s][part][k] is not None
                                    and folds[q]['metrics']['baseline'][s][part][k] is not None else None)
                                for k in E63.METRICS} for part in PARTS} for s in STAGES}
            folds[q]['paired_deltas'][name] = delta
            save_json(path.with_suffix('.json'), {'fingerprint': digest, 'target': q, 'candidate': name,
                      'train_subjects': tr, 'metrics': metrics, 'baseline': folds[q]['metrics']['baseline'],
                      'delta_vs_e63_equal_weight': delta, 'scores_file': str(path)})
            print(name, q, metrics['dt1200_hmm300']['full'], flush=True)
        del F
        gc.collect()
        torch.cuda.empty_cache()
    del D
    gc.collect()
    torch.cuda.empty_cache()
    rows = list(folds.values())
    summary, deltas = aggregate(rows)
    save_json(HERE / 'e66_results.json', {'fingerprint': digest, 'protocol': protocol,
              'aggregate': summary, 'macro_delta_vs_e63_equal_weight': deltas, 'folds': rows})
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
