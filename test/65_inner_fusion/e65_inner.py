from __future__ import annotations
import hashlib, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent
sys.path.insert(0, str(TESTS / '17_compliant'))
import loso as L
from multiscale import fit_mlp
import importlib.util
spec = importlib.util.spec_from_file_location('e59', TESTS / '61_trainfold_initialization' / 'e59_trainfold.py')
E59 = importlib.util.module_from_spec(spec); spec.loader.exec_module(E59)
E42 = E59.E42
TARGETS = ('004', '009', '010', '018', '034', '050')
WEIGHTS = (0., .25, .5, .75, 1.)
SCALES = E59.E42.SCALES
SEED = 20261107

def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, allow_nan=False), encoding='utf-8')

def metric(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float); p = s > 0
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    if not len(y):
        return {'n': 0, 'auc': None, 'f1': None, 'bacc': None}
    if not n0 or not n1:
        return {'n': len(y), 'auc': None, 'f1': float(f1_score(y, p, zero_division=0)), 'bacc': None}
    return {'n': len(y), 'auc': float(roc_auc_score(y, s)), 'f1': float(f1_score(y, p, zero_division=0)),
            'bacc': float(.5 * (p[y == 1].mean() + (~p[y == 0]).mean()))}

def evaluate(y, b, scores):
    masks = {'full': np.ones(len(y), bool), 'early': b <= 2, 'late': b >= 4}
    return {stage: {part: metric(y[m], score[m]) for part, m in masks.items()} for stage, score in scores.items()}

def stages(raw):
    ema = E59.ema_center(raw, 1200, None)
    return {'raw': raw, 'ema1200': ema, 'ema1200_hmm300': E59.hmm(ema, 300)}

def macro(rows, key):
    vals = [r[key] for r in rows if r[key] is not None]
    return float(np.mean(vals)) if vals else None

def aggregate(rows):
    out = {}
    for stage in ('raw', 'ema1200', 'ema1200_hmm300'):
        out[stage] = {}
        for part in ('full', 'early', 'late'):
            out[stage][part] = {k: macro([r[stage][part] for r in rows], k) for k in ('auc', 'f1', 'bacc')}
    return out

def main():
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    start = time.time()
    subs, D = L.load(); assert len(subs) == 57 and set(TARGETS) <= set(subs)
    checks = {}
    for q in subs:
        p = L.SLID / f'sub-{q}.npz'; checks[q] = {'source': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    eligible = [q for q in subs if q not in TARGETS]
    inner_val = list(np.random.RandomState(SEED).permutation(eligible)[:6])
    protocol = {'targets': TARGETS, 'outer_training_subjects': 56, 'inner_validation_subjects': inner_val,
      'inner_validation_pool': 'all subjects excluding all six outer diagnostic targets', 'inner_seed': SEED,
      'inner_fit': 'outer 56 minus shared six-person validation list = 50', 'feature_scales': list(SCALES),
      'LR_C': .001, 'MLP_seeds': [0,1,2], 'std': 'only inner fitting subjects',
      'weights': WEIGHTS, 'selection': 'validation macro mean of (AUC+bACC)/2; full subject sequence; EMA1200+HMM300 only; threshold 0; ties nearest .5 then smaller w',
      'outer_final': 'reuse E63 fold components for each target, 56-person training fit; no target labels for selection',
      'stages': ['raw', 'EMA1200', 'EMA1200+HMM300'], 'scope': 'single subject-held-out split, not complete OOF'}
    save(HERE / 'protocol.json', {'protocol': protocol, 'data_checks': checks})
    print('protocol written; building unchanged E42 five-scale features', flush=True)
    F = E59.E42.build_features(subs, D)
    folds = []
    for q in TARGETS:
        inner_fit = [p for p in subs if p not in inner_val and p != q]
        # q is already excluded from eligible, and therefore not in inner_val; outer q is not in fit pool
        assert len(inner_fit) == 50 and q not in inner_fit
        comp_path = HERE / f'inner_{q}_components.npz'; result_path = HERE / f'inner_{q}.json'
        if comp_path.exists() and result_path.exists():
            row = json.loads(result_path.read_text(encoding='utf-8')); folds.append(row); print(q, 'reused', flush=True); continue
        X = torch.cat([F[p] for p in inner_fit]); ytr = np.concatenate([D[p][1] for p in inner_fit])
        lr = L.fit_lr(X, ytr, .001); mlps = [fit_mlp(X, ytr, seed=s) for s in range(3)]
        with torch.no_grad():
            lr_fit = torch.cat([lr(F[p]) for p in inner_fit]); mlp_fit = torch.cat([torch.stack([f(F[p]) for f in mlps]).mean(0) for p in inner_fit])
            sl, sm = float(lr_fit.std()), float(mlp_fit.std()); assert sl > 0 and sm > 0
            arrays = {'train_subjects': np.array(inner_fit), 'validation_subjects': np.array(inner_val), 'target': np.array(q), 'std_lr': sl, 'std_mlp': sm}
            for p in inner_val:
                a, m = lr(F[p]), torch.stack([f(F[p]) for f in mlps]).mean(0)
                arrays[f'lr_scaled_{p}'] = (a/sl).cpu().numpy(); arrays[f'mlp_scaled_{p}'] = (m/sm).cpu().numpy()
        np.savez_compressed(comp_path, **arrays)
        del X, lr, mlps, lr_fit, mlp_fit; torch.cuda.empty_cache()
        candidates = {}; validation = {}
        for w in WEIGHTS:
            name = f'w{w:.2f}'; validation[name] = {}; candidates[name] = {}
            for p in inner_val:
                raw = 2 * (w * arrays[f'lr_scaled_{p}'] + (1-w) * arrays[f'mlp_scaled_{p}'])
                validation[name][p] = stages(raw)
            rows = [evaluate(D[p][1], D[p][2], validation[name][p]) for p in inner_val]
            candidates[name] = aggregate(rows)
        def key(name):
            r = candidates[name]['ema1200_hmm300']['full']; return ((r['auc'] + r['bacc']) / 2, -abs(float(name[1:])-.5), -float(name[1:]))
        chosen = max(candidates, key=key); chosen_w = float(chosen[1:])
        row = {'target': q, 'inner_fit_subjects': inner_fit, 'inner_validation_subjects': inner_val, 'components_file': str(comp_path), 'chosen_w': chosen_w, 'validation': candidates, 'elapsed_s': time.time()-start}
        save(result_path, row); folds.append(row); print(q, 'chosen', chosen, candidates[chosen]['ema1200_hmm300']['full'], flush=True)
    final_rows = []
    for fold in folds:
        q = fold['target']; zpath = TESTS / '63_layer_ablation' / f'fold_{q}_components.npz'
        with np.load(zpath, allow_pickle=False) as z:
            raw_by_w = {f'w{w:.2f}': 2*(w*z[f'lr_scaled_{q}']+(1-w)*z[f'mlp_scaled_{q}']) for w in WEIGHTS}
        y, b = D[q][1], D[q][2]; chosen = f"w{fold['chosen_w']:.2f}"
        entries = {}
        for name, raw in raw_by_w.items(): entries[name] = evaluate(y, b, stages(raw))
        final_rows.append({'target': q, 'chosen_w': fold['chosen_w'], 'chosen': entries[chosen], 'fixed_w050': entries['w0.50'], 'all_weights': entries,
                           'delta_chosen_minus_fixed': {st: {part: {k: (entries[chosen][st][part][k]-entries['w0.50'][st][part][k]) if entries[chosen][st][part][k] is not None and entries['w0.50'][st][part][k] is not None else None for k in ('auc','f1','bacc')} for part in ('full','early','late')} for st in ('raw','ema1200','ema1200_hmm300')}})
    result = {'protocol': protocol, 'inner_folds': folds, 'final_per_subject': final_rows,
      'aggregate_fixed_w050': aggregate([r['fixed_w050'] for r in final_rows]), 'aggregate_inner_selected': aggregate([r['chosen'] for r in final_rows]),
      'elapsed_s': round(time.time()-start, 1)}
    save(HERE / 'e65_results.json', result)
    for label in ('aggregate_fixed_w050','aggregate_inner_selected'):
        print(label, json.dumps(result[label]['ema1200_hmm300'], ensure_ascii=False), flush=True)
    print('COMPLETE', result['elapsed_s'], flush=True)

if __name__ == '__main__': main()
