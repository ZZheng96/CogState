"""E71: six-target MLP training-strength comparison; only ep differs from E67."""
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
spec = importlib.util.spec_from_file_location('e61', TESTS / '61_trainfold_initialization/e59_trainfold.py')
E61 = importlib.util.module_from_spec(spec); spec.loader.exec_module(E61)
TARGETS = ('004', '009', '010', '018', '034', '050')
E67 = TESTS / '67_mlp_loso/e67_results.json'


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


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; refusing CPU execution')
    subs, D = L.load(); assert len(subs) == 57 and set(TARGETS) <= set(subs)
    checks = {}
    for q in subs:
        p = L.SLID / f'sub-{q}.npz'
        with np.load(p) as z:
            order = np.argsort(z['t']); t, y, b = z['t'][order], z['y'][order].astype(int), z['blk'][order].astype(int)
            checks[q] = {'source': str(p), 'source_sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'n': len(t),
                         'sorted_time_strictly_increasing': bool(np.all(np.diff(t) > 0)),
                         'label_order_equal_L_load': bool(np.array_equal(y, D[q][1])),
                         'block_order_equal_L_load': bool(np.array_equal(b, D[q][2]))}
            assert checks[q]['sorted_time_strictly_increasing'] and checks[q]['label_order_equal_L_load'] and checks[q]['block_order_equal_L_load']
    baseline = json.loads(E67.read_text(encoding='utf-8'))
    protocol = {'experiment': 'E71_mlp_training_strength', 'targets': TARGETS, 'n_folds': 6,
                'features': 'E42 fixed multiscale features, unchanged from E67',
                'training': 'LOSO training fold only; MLP seeds [0,1,2]; fit_mlp ep=60 (E67 default ep=30), all other defaults unchanged',
                'score': 'raw = 2 * mlp_scaled; std_mlp is pooled training-fold output std',
                'postprocessing': 'E61 ema_center(raw,1200,None) then forward hmm(dwell=300); no tuning',
                'baseline': 'E67 default fit_mlp ep=30 target-fold metrics; no target labels used for training or tuning',
                'device': 'cuda', 'target_selection': 'pre-specified diagnostic targets 004,009,010,018,034,050'}
    save_json(HERE / 'protocol.json', {'protocol': protocol, 'data_checks': checks})
    F = E61.E42.build_features(subs, D)
    folds = []
    for i, q in enumerate(TARGETS, 1):
        t0 = time.time(); tr = [p for p in subs if p != q]
        X = torch.cat([F[p] for p in tr]); ytr = np.concatenate([D[p][1] for p in tr])
        models = [fit_mlp(X, ytr, ep=60, seed=s) for s in range(3)]
        with torch.no_grad():
            train_m = torch.cat([torch.stack([m(F[p]) for m in models]).mean(0) for p in tr])
            sd = float(train_m.std()); assert sd > 0
            mlp_scaled = (torch.stack([m(F[q]) for m in models]).mean(0) / sd).cpu().numpy()
        raw = 2 * mlp_scaled; stages = stage(raw); y, b = D[q][1], D[q][2]
        metrics = {k: evaluate(y, b, v) for k, v in stages.items()}
        e67row = next(r for r in baseline['folds'] if r['target'] == q)
        base = e67row['metrics']; delta = {s: {part: {m: (metrics[s][part][m] - base[s][part][m]) if metrics[s][part][m] is not None and base[s][part][m] is not None else None for m in ('auc','f1','bacc')} for part in ('full','early','late')} for s in stages}
        np.savez_compressed(HERE / f'fold_{q}.npz', target=np.array(q), train_subjects=np.array(tr), y=y, block=b, mlp_scaled=mlp_scaled, **stages)
        row = {'fold': i, 'target': q, 'train_subjects': tr, 'epochs': 60, 'seeds': [0,1,2], 'std_mlp': sd,
               'metrics': metrics, 'baseline_e67': base, 'delta_vs_e67': delta, 'elapsed_s': time.time()-t0}
        save_json(HERE / f'fold_{q}.json', row); folds.append(row)
        del X, models, train_m; torch.cuda.empty_cache()
        print(f'[{i}/6] {q} {metrics["dt1200_hmm300"]["full"]}', flush=True)
    aggregate = {}
    for s in ('raw', 'dt1200', 'dt1200_hmm300'):
        aggregate[s] = {}
        for part in ('full', 'early', 'late'):
            aggregate[s][part] = {m: (float(np.mean(vals)) if vals else None) for m in ('auc','f1','bacc') for vals in [[r['metrics'][s][part][m] for r in folds if r['metrics'][s][part][m] is not None]]}
    baseagg = baseline.get('aggregate', {})
    delta_aggregate = {s: {part: {m: (aggregate[s][part][m] - float(np.mean(vals)) if aggregate[s][part][m] is not None and vals else None) for m in ('auc','f1','bacc') for vals in [[next(r for r in baseline['folds'] if r['target']==q)['metrics'][s][part][m] for q in TARGETS if next(r for r in baseline['folds'] if r['target']==q)['metrics'][s][part][m] is not None]]} for part in ('full','early','late')} for s in aggregate}
    save_json(HERE / 'e71_results.json', {'protocol': protocol, 'aggregate': aggregate, 'delta_aggregate_vs_e67': delta_aggregate, 'folds': folds})
    print('AGGREGATE', json.dumps(aggregate, ensure_ascii=False), flush=True)

if __name__ == '__main__': main()
