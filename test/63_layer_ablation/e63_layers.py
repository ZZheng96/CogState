"""Six fixed diagnostic LOSO folds; layer outputs and fixed fusion ablations."""
from __future__ import annotations
import hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent
sys.path.insert(0, str(TESTS / '17_compliant'))
import loso as L
from multiscale import fit_mlp
spec = importlib.util.spec_from_file_location('e61_layers', TESTS / '61_trainfold_initialization/e59_trainfold.py')
E61 = importlib.util.module_from_spec(spec); spec.loader.exec_module(E61)
TARGETS = ('004', '009', '010', '018', '034', '050')
WEIGHTS = (0., .25, .5, .75, 1.)
METRICS = ('auc', 'f1', 'bacc')


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, allow_nan=False), encoding='utf-8')


def metric(y, s):
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    p = s > 0
    return {'n': len(y), 'n_negative': n0, 'n_positive': n1, 'n_classes': int(n0 > 0) + int(n1 > 0),
            'auc': float(roc_auc_score(y, s)) if n0 and n1 else None,
            'f1': float(f1_score(y, p, zero_division=0)) if len(y) else None,
            'bacc': float(.5 * (p[y == 1].mean() + (~p[y == 0]).mean())) if n0 and n1 else None}


def evaluate(y, b, s):
    masks = {'full': np.ones(len(y), bool), 'early': b <= 2, 'late': b >= 4}
    masks.update({f'block_{int(k)}': b == k for k in np.unique(b)})
    return {name: metric(y[m], s[m]) for name, m in masks.items()}


def stages(raw):
    dt = E61.ema_center(raw, 1200, None)
    return {'raw': raw, 'dt1200': dt, 'dt1200_hmm300': E61.hmm(dt, 300)}


def cache_inventory():
    rows = []
    for directory in (TESTS / '58_stream_block_reset', TESTS / '61_trainfold_initialization'):
        for p in sorted(directory.rglob('*.npz')):
            with np.load(p, allow_pickle=False) as z:
                keys = z.files
                rows.append({'path': str(p), 'keys': keys,
                             'has_all_subject_layer_outputs': all('lr_scaled_' + q in keys and 'mlp_scaled_' + q in keys for q in TARGETS)})
    return rows


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; do not silently run CPU training')
    inventory = cache_inventory()
    # Existing E58/E61 files contain fused scores, not recoverable separate components.
    if any(r['has_all_subject_layer_outputs'] for r in inventory):
        raise RuntimeError('Layer cache found: establish protocol provenance before retraining')
    subs, D = L.load()
    assert len(subs) == 57 and set(TARGETS) <= set(subs)
    metadata, checks = {}, {}
    for q in subs:
        path = L.SLID / f'sub-{q}.npz'
        with np.load(path) as z:
            order = np.argsort(z['t']); t, y, b = z['t'][order], z['y'][order].astype(int), z['blk'][order].astype(int)
            checks[q] = {'source': str(path), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'input_time_monotonic': bool(np.all(np.diff(z['t']) >= 0)),
                         'sorted_time_strictly_increasing': bool(np.all(np.diff(t) > 0)),
                         'label_order_equal_L_load': bool(np.array_equal(y, D[q][1])),
                         'block_order_equal_L_load': bool(np.array_equal(b, D[q][2])),
                         'n': len(t), 'raw_BDF_alignment_proven': False}
            assert checks[q]['sorted_time_strictly_increasing'] and checks[q]['label_order_equal_L_load'] and checks[q]['block_order_equal_L_load']
            assert set(np.unique(y)) <= {0, 1}
            metadata.update({f't_{q}': t, f'y_{q}': y, f'block_{q}': b, f'source_row_{q}': order})
    protocol = {'targets': TARGETS, 'training_subject_count_per_fold': 56, 'feature_scales': E61.E42.SCALES,
                'LR_C': .001, 'MLP_seeds': [0, 1, 2], 'MLP': 'unchanged multiscale.fit_mlp defaults',
                'scale': 'pooled training output torch.std correction=1, no target statistics',
                'fusion': '2*(w*LR/std_lr+(1-w)*MLP/std_mlp)', 'weights': WEIGHTS,
                'baseline_w': .5, 'threshold': 0, 'positive_label': 1,
                'EMA_initialization': 'first score, same as E61 baseline_first; update then subtract',
                'postprocessing': 'one continuous subject sequence; no block reset; forward HMM dwell=300',
                'time_constants': '1200 and 300 measured in retained frames, same as E61; not gap-aware wall clock',
                'labels': 'training labels for training only; target labels and blocks for offline evaluation only',
                'single_class': 'AUC and bACC null; positive-class F1 defined with zero_division=0',
                'scope': 'six selected diagnostic subjects, not generalization or candidate selection evidence',
                'alignment_limit': 'Cache order checks do not prove BDF/event correctness. Parent audit: 57 subjects sliding1/newfeat/scores1 t/y/blk equal; E53 score-only length check; BDF prefix/chunk checks for 004/009/010/018/050 do not exclude common upstream event-label errors. No new BDF audit here.'}
    save_json(HERE / 'protocol.json', {'protocol': protocol, 'cache_inventory': inventory, 'data_checks': checks})
    print('CUDA sequential six folds; layer caches unavailable; building unchanged E42 features', flush=True)
    F = E61.E42.build_features(subs, D)
    folds = []
    for q in TARGETS:
        start = time.time(); tr = [p for p in subs if p != q]
        assert q not in tr and len(tr) == 56
        path = HERE / f'fold_{q}_components.npz'
        if path.exists():
            with np.load(path) as z:
                assert str(z['target']) == q and list(z['train_subjects']) == tr
                assert str(z['data_hashes']) == json.dumps({p: checks[p]['source_sha256'] for p in subs}, sort_keys=True)
                arrays = {k: z[k] for k in z.files}
        else:
            X = torch.cat([F[p] for p in tr]); y = np.concatenate([D[p][1] for p in tr])
            lr = L.fit_lr(X, y, .001)
            mlps = [fit_mlp(X, y, seed=s) for s in range(3)]
            with torch.no_grad():
                a = {p: lr(F[p]) for p in subs}
                m = {p: torch.stack([f(F[p]) for f in mlps]).mean(0) for p in subs}
                sl, sm = float(torch.cat([a[p] for p in tr]).std()), float(torch.cat([m[p] for p in tr]).std())
                assert sl > 0 and sm > 0
                arrays = dict(metadata, train_subjects=np.array(tr), target=np.array(q), std_lr=sl, std_mlp=sm,
                              data_hashes=np.array(json.dumps({p: checks[p]['source_sha256'] for p in subs}, sort_keys=True)))
                for p in subs:
                    arrays.update({f'lr_{p}': a[p].cpu().numpy(), f'mlp_{p}': m[p].cpu().numpy(),
                                   f'lr_scaled_{p}': (a[p]/sl).cpu().numpy(), f'mlp_scaled_{p}': (m[p]/sm).cpu().numpy(),
                                   f'raw_{p}': (a[p]/sl + m[p]/sm).cpu().numpy()})
            np.savez_compressed(path, **arrays)
            del X, lr, mlps, a, m
            torch.cuda.empty_cache()
        y, b = D[q][1], D[q][2]
        layers = {'lr_scaled': arrays[f'lr_scaled_{q}'], 'mlp_scaled': arrays[f'mlp_scaled_{q}']}
        candidates, score_arrays = {}, {}
        for w in WEIGHTS:
            name = f'w{w:.2f}'
            raw = 2*(w*layers['lr_scaled']+(1-w)*layers['mlp_scaled'])
            candidates[name] = {}
            for stage, s in stages(raw).items():
                assert len(s) == len(y) and np.isfinite(s).all()
                candidates[name][stage] = evaluate(y, b, s)
                score_arrays[f'{name}_{stage}'] = s
        np.testing.assert_array_equal(score_arrays['w0.50_raw'], arrays[f'raw_{q}'])
        oldpath = E61.FOLD_DIR / f'fold_{subs.index(q)+1:02d}_{q}.npz'
        with np.load(oldpath) as z:
            assert str(z['target']) == q and list(z['train_subjects']) == tr
            old = z[f'raw_{q}']
        compare = {'source': str(oldpath), 'split_matches': True, 'all_subject_raw_max_abs': None, 'stages': {}}
        with np.load(oldpath) as z:
            compare['all_subject_raw_max_abs'] = float(max(np.max(np.abs(arrays[f'raw_{p}'] - z[f'raw_{p}'])) for p in subs))
        oldmetrics = {}
        for stage, s in stages(old).items():
            new = score_arrays[f'w0.50_{stage}']
            compare['stages'][stage] = {'array_equal': bool(np.array_equal(new, s)), 'max_abs': float(np.max(np.abs(new-s))),
                                        'allclose_atol1e-6_rtol1e-6': bool(np.allclose(new, s, atol=1e-6, rtol=1e-6))}
            oldmetrics[stage] = evaluate(y, b, s)
            score_arrays[f'e61_{stage}'] = s
        layer_metrics = {name: evaluate(y, b, s) for name, s in layers.items()}
        reversal = {}
        for part in candidates['w0.50']['raw']:
            component_reverse = [name for name in layers if layer_metrics[name][part]['auc'] is not None and layer_metrics[name][part]['auc'] < .5]
            reverse_stages = [stage for stage in ('raw', 'dt1200', 'dt1200_hmm300') if candidates['w0.50'][stage][part]['auc'] is not None and candidates['w0.50'][stage][part]['auc'] < .5]
            reversal[part] = {'definition': 'AUC<0.5, diagnostic only, never flip', 'reversed_components': component_reverse,
                              'first_baseline_stage': reverse_stages[0] if reverse_stages else None,
                              'first_layer': 'component' if component_reverse else (reverse_stages[0] if reverse_stages else None)}
        np.savez_compressed(HERE / f'fold_{q}_scores.npz', **score_arrays, t=metadata[f't_{q}'], y=y, block=b)
        row = {'target': q, 'train_subjects': tr, 'target_isolated': q not in tr, 'components_file': str(path),
               'elapsed_s': time.time()-start, 'layers': layer_metrics, 'candidates': candidates,
               'e61_cached_baseline': oldmetrics, 'baseline_comparison': compare, 'reversal': reversal}
        save_json(HERE / f'fold_{q}.json', row); folds.append(row)
        print(q, 'baseline', candidates['w0.50']['dt1200_hmm300']['full'], 'comparison', compare['stages']['raw'], 'reversal', reversal['full'], flush=True)
    aggregate, deltas = {}, []
    for name in folds[0]['candidates']:
        aggregate[name] = {}
        for stage in ('raw', 'dt1200', 'dt1200_hmm300'):
            aggregate[name][stage] = {}
            for part in ('full', 'early', 'late'):
                group = {}
                for k in METRICS:
                    values = [(r['target'], r['candidates'][name][stage][part][k]) for r in folds]
                    values = [(q, v) for q, v in values if v is not None]
                    group[k] = {'macro': float(np.mean([v for _, v in values])) if values else None,
                                'n_subjects': len(values), 'worst': min(values, key=lambda x: x[1]) if values else None}
                aggregate[name][stage][part] = group
                for r in folds:
                    candidate, base = r['candidates'][name][stage][part], r['candidates']['w0.50'][stage][part]
                    cached = r['e61_cached_baseline'][stage][part]
                    deltas.append({'target': r['target'], 'candidate': name, 'stage': stage, 'part': part,
                                   'delta_vs_same_fit_w050': {k: candidate[k]-base[k] if candidate[k] is not None and base[k] is not None else None for k in METRICS},
                                   'delta_vs_e61_cache': {k: candidate[k]-cached[k] if candidate[k] is not None and cached[k] is not None else None for k in METRICS}})
    save_json(HERE / 'e63_results.json', {'protocol': protocol, 'data_checks': checks, 'aggregate': aggregate, 'paired_deltas': deltas, 'folds': folds})
    for name in aggregate:
        print(name, json.dumps(aggregate[name]['dt1200_hmm300'], ensure_ascii=False), flush=True)
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
