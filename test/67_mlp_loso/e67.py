"""E67: fixed-protocol MLP-only 57-subject LOSO and E53 comparison."""
from __future__ import annotations
import hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent
ROOT = TESTS.parent
sys.path.insert(0, str(TESTS / '17_compliant'))
import loso as L
from multiscale import fit_mlp

E61 = importlib.util.module_from_spec(spec := importlib.util.spec_from_file_location('e61', TESTS / '61_trainfold_initialization/e59_trainfold.py'))
spec.loader.exec_module(E61)
TARGET_CACHE = TESTS / '63_layer_ablation'
E53 = TESTS / '53_e17_post_loso/e53_scores.npz'


def save_json(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=1, allow_nan=False), encoding='utf-8')


def metric(y, s):
    y, s = np.asarray(y, int), np.asarray(s, float)
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    p = s > 0
    return {'n': len(y), 'n_negative': n0, 'n_positive': n1,
            'auc': float(roc_auc_score(y, s)) if n0 and n1 else None,
            'f1': float(f1_score(y, p, zero_division=0)) if len(y) else None,
            'bacc': float(.5 * (p[y == 1].mean() + (~p[y == 0]).mean())) if n0 and n1 else None}


def evaluate(y, b, s):
    return {k: metric(y[m], s[m]) for k, m in {
        'full': np.ones(len(y), bool), 'early': b <= 2, 'late': b >= 4}.items()}


def stage(raw):
    dt = E61.ema_center(raw, 1200, None)
    return {'raw': raw, 'dt1200': dt, 'dt1200_hmm300': E61.hmm(dt, 300)}


def checks_and_metadata(subs, D):
    checks, meta = {}, {}
    for q in subs:
        p = L.SLID / f'sub-{q}.npz'
        with np.load(p) as z:
            order = np.argsort(z['t']); t, y, b = z['t'][order], z['y'][order].astype(int), z['blk'][order].astype(int)
            checks[q] = {'source': str(p), 'source_sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                         'n': len(t), 'sorted_time_strictly_increasing': bool(np.all(np.diff(t) > 0)),
                         'label_order_equal_L_load': bool(np.array_equal(y, D[q][1])),
                         'block_order_equal_L_load': bool(np.array_equal(b, D[q][2]))}
            assert checks[q]['sorted_time_strictly_increasing'] and checks[q]['label_order_equal_L_load'] and checks[q]['block_order_equal_L_load']
            meta.update({f't_{q}': t, f'y_{q}': y, f'block_{q}': b, f'source_row_{q}': order})
    return checks, meta


def load_cache(q, tr, checks):
    p = TARGET_CACHE / f'fold_{q}_components.npz'
    if not p.exists(): return None
    with np.load(p, allow_pickle=False) as z:
        assert str(z['target']) == q
        assert list(z['train_subjects']) == tr
        expected = json.dumps({x: checks[x]['source_sha256'] for x in checks}, sort_keys=True)
        assert str(z['data_hashes']) == expected
        assert f'mlp_scaled_{q}' in z.files and 'std_mlp' in z.files
        return np.asarray(z[f'mlp_scaled_{q}'], dtype=float), {'source': str(p), 'validated': True}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; refusing CPU execution')
    existing_outputs = [p for p in HERE.iterdir() if p.name not in ('e67.py', '__pycache__')]
    unexpected = [p for p in existing_outputs if not (p.name == 'protocol.json' or p.name.startswith('fold_'))]
    if unexpected:
        raise RuntimeError(f'Unexpected E67 files; refusing overwrite: {unexpected}')
    subs, D = L.load(); assert len(subs) == 57
    checks, metadata = checks_and_metadata(subs, D)
    F = E61.E42.build_features(subs, D)
    e53 = np.load(E53, allow_pickle=False)
    assert list(e53['subjects'].astype(str)) == subs
    protocol = {'experiment': 'E67_mlp_loso', 'subjects': subs, 'n_folds': 57,
                'features': 'E42 fixed multiscale features, scales=' + repr(E61.E42.SCALES),
                'training': 'LOSO training fold only; MLP seeds [0,1,2], unchanged fit_mlp defaults',
                'score': 'raw = 2 * mlp_scaled; std_mlp is pooled training-fold output std',
                'postprocessing': 'E61 ema_center(raw,1200,None) then forward hmm(dwell=300); no tuning',
                'threshold': 0, 'positive_label': 1, 'device': 'cuda',
                'cache_rule': 'E63 six component caches reused only after target/train_subjects/data_hashes validation',
                'baseline': 'E53 e53_scores raw, same ema_center(1200,None)+hmm(300)'}
    if not (HERE / 'protocol.json').exists():
        save_json(HERE / 'protocol.json', {'protocol': protocol, 'data_checks': checks})
    folds = []
    for i, q in enumerate(subs, 1):
        existing_json = HERE / f'fold_{i:02d}_{q}.json'
        existing_npz = HERE / f'fold_{i:02d}_{q}.npz'
        if existing_json.exists() and existing_npz.exists():
            folds.append(json.loads(existing_json.read_text(encoding='utf-8')))
            print(f'[{i}/57] {q} reused existing E67 fold', flush=True)
            continue
        t0 = time.time(); tr = [p for p in subs if p != q]
        cached = load_cache(q, tr, checks)
        if cached:
            mlp_scaled, provenance = cached
        else:
            X = torch.cat([F[p] for p in tr]); ytr = np.concatenate([D[p][1] for p in tr])
            models = [fit_mlp(X, ytr, seed=s) for s in range(3)]
            with torch.no_grad():
                train_m = torch.cat([torch.stack([m(F[p]) for m in models]).mean(0) for p in tr])
                sd = float(train_m.std()); assert sd > 0
                mlp_scaled = (torch.stack([m(F[q]) for m in models]).mean(0) / sd).cpu().numpy()
            provenance = {'source': 'retrained', 'validated': False, 'std_mlp': sd}
            del X, models, train_m
            torch.cuda.empty_cache()
        raw = 2 * mlp_scaled
        stages = stage(raw)
        y, b = D[q][1], D[q][2]
        metrics = {k: evaluate(y, b, v) for k, v in stages.items()}
        base_raw = e53[q]
        base = {k: evaluate(y, b, v) for k, v in stage(base_raw).items()}
        delta = {s: {part: {m: (metrics[s][part][m] - base[s][part][m]) if metrics[s][part][m] is not None and base[s][part][m] is not None else None for m in ('auc','f1','bacc')} for part in ('full','early','late')} for s in stages}
        np.savez_compressed(HERE / f'fold_{i:02d}_{q}.npz', target=np.array(q), train_subjects=np.array(tr),
                            t=metadata[f't_{q}'], y=y, block=b, mlp_scaled=mlp_scaled, **stages,
                            baseline_raw=base_raw, baseline_dt1200=stage(base_raw)['dt1200'], baseline_dt1200_hmm300=stage(base_raw)['dt1200_hmm300'])
        row = {'fold': i, 'target': q, 'train_subjects': tr, 'provenance': provenance,
               'metrics': metrics, 'baseline_e53': base, 'delta_vs_e53': delta, 'elapsed_s': time.time()-t0}
        save_json(HERE / f'fold_{i:02d}_{q}.json', row); folds.append(row)
        print(f'[{i}/57] {q} source={provenance["source"]} hmm={metrics["dt1200_hmm300"]["full"]}', flush=True)
    aggregate, comparisons = {}, []
    for s in ('raw','dt1200','dt1200_hmm300'):
        aggregate[s] = {}
        for part in ('full','early','late'):
            aggregate[s][part] = {m: (float(np.mean(vals)) if vals else None) for m in ('auc','f1','bacc')
                                    for vals in [[r['metrics'][s][part][m] for r in folds if r['metrics'][s][part][m] is not None]]}
            for r in folds:
                d = r['delta_vs_e53'][s][part]
                comparisons.append({'target': r['target'], 'stage': s, 'part': part, 'baseline': r['baseline_e53'][s][part], 'mlp': r['metrics'][s][part], 'delta': d,
                                    'status': ('improved' if d['bacc'] > 0 else ('regressed' if d['bacc'] < 0 else 'equal')) if d['bacc'] is not None else 'undefined'})
    save_json(HERE / 'e67_results.json', {'protocol': protocol, 'aggregate': aggregate, 'folds': folds})
    save_json(HERE / 'e67_vs_e53_per_subject.json', {'protocol': protocol, 'comparisons': comparisons})
    print('AGGREGATE', json.dumps(aggregate, ensure_ascii=False), flush=True)

if __name__ == '__main__': main()
