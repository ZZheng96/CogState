from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent
sys.path.insert(0, str(TESTS / '17_compliant'))
import loso as L
spec_path = TESTS / '61_trainfold_initialization' / 'e59_trainfold.py'
import importlib.util
spec = importlib.util.spec_from_file_location('e61', spec_path)
E61 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E61)

TARGETS = ('004', '009', '010', '018', '034', '050')
WEIGHTS = (0.0, 0.5)
PARTS = {'full': lambda b: np.ones(len(b), dtype=bool), 'early': lambda b: b <= 2, 'late': lambda b: b >= 4}

def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, allow_nan=False), encoding='utf-8')

def metric(y, score):
    y = np.asarray(y, dtype=int); score = np.asarray(score, dtype=float)
    n0 = int((y == 0).sum()); n1 = int((y == 1).sum())
    pred = score > 0
    return {'n': len(y), 'n_negative': n0, 'n_positive': n1,
            'auc': float(roc_auc_score(y, score)) if n0 and n1 else None,
            'f1': float(f1_score(y, pred, zero_division=0)) if len(y) else None,
            'bacc': float(0.5 * (pred[y == 1].mean() + (~pred[y == 0]).mean())) if n0 and n1 else None}

def evaluate(y, block, score):
    return {part: metric(y[m], score[m]) for part, make_mask in PARTS.items() for m in [make_mask(block)]}

def stages(raw):
    detrend = E61.ema_center(raw, 1200, None)
    return {'raw': raw, 'detrend': detrend, 'hmm': E61.hmm(detrend, 300)}

def main():
    subs, data = L.load()
    assert len(subs) == 57 and set(TARGETS) <= set(subs)
    protocol = {
        'experiment': 'E69_component_diagnosis',
        'targets': list(TARGETS), 'scope': 'six E63 validated component-cache folds',
        'cache_source': str(TESTS / '63_layer_ablation'),
        'components': ['LR scaled', 'MLP scaled'],
        'fusion': 'raw = 2*(w*LR_scaled + (1-w)*MLP_scaled)',
        'weights_evaluated': list(WEIGHTS), 'baseline_weight': 0.5,
        'postprocessing': {'detrend': 'ema_center(1200,None)', 'hmm': 'hmm(300)', 'continuous_sequence': True, 'block_reset': False},
        'threshold': 0, 'positive_label': 1,
        'evaluation_parts': ['full', 'early (block<=2)', 'late (block>=4)'],
        'selection': 'none; no target labels used for weight choice',
        'single_class': 'AUC and bACC are null; F1 uses positive-class threshold 0',
        'training': 'no retraining; reuse E63 component caches'
    }
    save(HERE / 'protocol.json', {'protocol': protocol})
    folds = []
    for q in TARGETS:
        cache = TESTS / '63_layer_ablation' / f'fold_{q}_components.npz'
        with np.load(cache, allow_pickle=False) as z:
            assert str(z['target']) == q
            lr = np.asarray(z[f'lr_scaled_{q}'], dtype=float)
            mlp = np.asarray(z[f'mlp_scaled_{q}'], dtype=float)
            train_subjects = list(z['train_subjects'].astype(str))
        y = np.asarray(data[q][1], dtype=int); block = np.asarray(data[q][2], dtype=int)
        assert len(lr) == len(y) == len(mlp)
        components = {'LR': lr, 'MLP': mlp}
        result = {'target': q, 'train_subjects': train_subjects, 'cache': str(cache), 'metrics': {}, 'scores': {}}
        for name, score in components.items():
            result['metrics'][name] = evaluate(y, block, score)
            result['scores'][name] = score
        for w in WEIGHTS:
            name = f'fusion_w{w:.2f}'
            raw = 2 * (w * lr + (1 - w) * mlp)
            ss = stages(raw)
            result['metrics'][name] = {stage: evaluate(y, block, score) for stage, score in ss.items()}
            result['scores'][name] = raw
            result['scores'][f'{name}_detrend'] = ss['detrend']
            result['scores'][f'{name}_hmm'] = ss['hmm']
        arrays = {'y': y, 'block': block, 'lr_scaled': lr, 'mlp_scaled': mlp}
        for name, scores in result['scores'].items(): arrays[name] = scores
        np.savez_compressed(HERE / f'fold_{q}.npz', **arrays)
        json_result = {k: v for k, v in result.items() if k != 'scores'}
        save(HERE / f'fold_{q}.json', json_result)
        folds.append(json_result)
    aggregate = {}
    for name in ('LR', 'MLP', 'fusion_w0.00', 'fusion_w0.50'):
        aggregate[name] = {}
        metric_stages = ('raw', 'detrend', 'hmm') if name.startswith('fusion') else (None,)
        for stage in metric_stages:
            key = stage or 'direct'
            aggregate[name][key] = {}
            for part in PARTS:
                aggregate[name][key][part] = {}
                metric_key = part if stage is None else (stage, part)
                for m in ('auc', 'f1', 'bacc'):
                    values = [r['metrics'][name][metric_key[0]][metric_key[1]][m] if stage is not None else r['metrics'][name][metric_key][m] for r in folds if (m in (r['metrics'][name][metric_key[0]][metric_key[1]] if stage is not None else r['metrics'][name][metric_key]) and (r['metrics'][name][metric_key[0]][metric_key[1]][m] if stage is not None else r['metrics'][name][metric_key][m]) is not None)]
                    aggregate[name][key][part][m] = float(np.mean(values)) if values else None
    save(HERE / 'e69_results.json', {'protocol': protocol, 'aggregate': aggregate, 'folds': folds})
    print(json.dumps(aggregate, ensure_ascii=False, indent=1))

if __name__ == '__main__':
    main()
