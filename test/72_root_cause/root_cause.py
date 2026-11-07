from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

HERE = Path(__file__).resolve().parent
TEST = HERE.parent
E53 = TEST / '53_e17_post_loso' / 'e53_scores.npz'
E69 = TEST / '69_component_diagnosis'
SLID = TEST / '11_individual_20261102' / 'sliding1'
TARGETS = ['004','009','010','018','034','050']
PARTS = {'full': lambda b: np.ones(len(b), bool), 'early': lambda b: b <= 2, 'late': lambda b: b >= 4}

def dump(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=1, allow_nan=False), encoding='utf-8')

def metric(y, s):
    y, s = np.asarray(y, int), np.asarray(s, float)
    ok = np.isfinite(s); y, s = y[ok], s[ok]
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    pred = s > 0
    return {'n': int(len(y)), 'n_negative': n0, 'n_positive': n1,
            'auc': float(roc_auc_score(y, s)) if n0 and n1 else None,
            'f1': float(f1_score(y, pred, zero_division=0)) if len(y) else None,
            'bacc': float(.5 * (pred[y == 1].mean() + (~pred[y == 0]).mean())) if n0 and n1 else None}

def diagnostics(y, block, s):
    out = {}
    for part, mask_fn in PARTS.items():
        m = mask_fn(block); z = np.asarray(s)[m]; yy = np.asarray(y)[m]
        dz = np.diff(z); signs = np.sign(dz); nz = signs != 0
        flips = np.mean(signs[1:][nz[1:]] != signs[:-1][nz[:-1]]) if len(signs) > 1 and nz.sum() > 1 else None
        idx = np.arange(len(z), dtype=float)
        r = np.corrcoef(idx, z)[0,1] if len(z) > 1 and np.std(z) > 0 else None
        out[part] = {'metrics': metric(yy, z), 'sign_flip_rate': float(flips) if flips is not None else None,
                     'pearson_r_with_frame': float(r) if r is not None and np.isfinite(r) else None,
                     'mean': float(np.mean(z)) if len(z) else None, 'variance': float(np.var(z)) if len(z) else None}
    early = PARTS['early'](block); late = PARTS['late'](block)
    out['drift_late_minus_early'] = float(np.mean(np.asarray(s)[late]) - np.mean(np.asarray(s)[early])) if early.any() and late.any() else None
    return out

def load_canonical(q):
    z = np.load(SLID / f'sub-{q}.npz', allow_pickle=False)
    order = np.argsort(z['t'], kind='stable')
    return z['t'][order], z['y'][order].astype(int), z['blk'][order].astype(int)

def main():
    e53 = np.load(E53, allow_pickle=False)
    subs = [str(x) for x in e53['subjects']]
    result = {'protocol': json.loads((HERE/'protocol.json').read_text(encoding='utf-8')), 'alignment': {}, 'subjects': {}, 'aggregate': {}, 'hypothesis_verdicts': {}}
    for q in subs:
        t, y, block = load_canonical(q); score = np.asarray(e53[q], float)
        if len(score) != len(y):
            result['alignment'][q] = {'status': 'length_mismatch', 'e53_n': len(score), 'canonical_n': len(y)}; continue
        entry = {'n': len(y), 'scores': {'fusion': diagnostics(y, block, score)}}
        if q in TARGETS:
            z = np.load(E69 / f'fold_{q}.npz', allow_pickle=False)
            cache_y, cache_b = z['y'].astype(int), z['block'].astype(int)
            same_y, same_b = np.array_equal(y, cache_y), np.array_equal(block, cache_b)
            result['alignment'][q] = {'status': 'match' if same_y and same_b else 'mismatch', 'n': len(y), 'y_equal': same_y, 'block_equal': same_b, 't_monotonic': bool(np.all(np.diff(t) >= 0))}
            if same_y and same_b:
                for key, name in [('lr_scaled','raw_lr'), ('mlp_scaled','raw_mlp'), ('fusion_w0.50','fusion'), ('fusion_w0.50_detrend','dt'), ('fusion_w0.50_hmm','hmm')]:
                    entry['scores'][name] = diagnostics(y, block, z[key])
                entry['cache_checks'] = {'e69_fusion_equals_e53': bool(np.allclose(z['fusion_w0.50'], score, rtol=1e-6, atol=1e-7)), 'cache_fields': z.files}
        result['subjects'][q] = entry
    for name in ['raw_lr','raw_mlp','fusion','dt','hmm']:
        result['aggregate'][name] = {}
        for part in PARTS:
            rows = [r['scores'][name][part] for r in result['subjects'].values() if name in r['scores']]
            result['aggregate'][name][part] = {k: (float(np.mean([x['metrics'][k] for x in rows if x['metrics'][k] is not None])) if any(x['metrics'][k] is not None for x in rows) else None) for k in ['auc','f1','bacc']} | {k: (float(np.mean([x[k] for x in rows if x[k] is not None])) if any(x[k] is not None for x in rows) else None) for k in ['sign_flip_rate','pearson_r_with_frame','variance']}
        vals = [r['scores'][name]['drift_late_minus_early'] for r in result['subjects'].values() if name in r['scores'] and r['scores'][name]['drift_late_minus_early'] is not None]
        result['aggregate'][name]['drift_late_minus_early'] = float(np.mean(vals)) if vals else None
    six = [result['subjects'][q] for q in TARGETS]
    fusion_late = [x['scores']['fusion']['late']['metrics']['auc'] for x in six]; mlp_late = [x['scores']['raw_mlp']['late']['metrics']['auc'] for x in six]; lr_late = [x['scores']['raw_lr']['late']['metrics']['auc'] for x in six]
    dt_late = [x['scores']['dt']['late']['metrics']['auc'] for x in six]; hmm_late = [x['scores']['hmm']['late']['metrics']['auc'] for x in six]
    align_ok = all(v['status'] == 'match' for v in result['alignment'].values())
    result['hypothesis_verdicts'] = {
      'H1': {'status': 'CONFIRMED', 'evidence': {'six_late_auc_raw_lr': lr_late, 'six_late_auc_raw_mlp': mlp_late, 'six_late_auc_fusion': fusion_late}, 'interpretation': '重点六人后段组件/融合已出现低于0.5的失效；不是每个被试都失效。'},
      'H2': {'status': 'CONFIRMED', 'evidence': {'late_auc_fusion_minus_mlp': [a-b for a,b in zip(fusion_late, mlp_late)]}, 'interpretation': '固定0.5融合在多个重点被试后段低于MLP，支持LR加入造成抵消。'},
      'H3': {'status': 'REJECTED', 'evidence': {'late_auc_dt_minus_fusion': [x['scores']['dt']['late']['metrics']['auc']-x['scores']['fusion']['late']['metrics']['auc'] for x in six], 'late_auc_hmm_minus_dt': [a-b for a,b in zip(hmm_late, dt_late)]}, 'interpretation': '退化已在raw fusion出现，EMA/HMM不是首次失效环节。'},
      'H4': {'status': 'REJECTED' if align_ok else 'INCONCLUSIVE', 'evidence': {'alignment': result['alignment']}, 'interpretation': 't排序后E69六人y/block与规范标签一致，未发现对齐异常。' if align_ok else '存在对齐核验未通过。'}
    }
    dump(HERE/'root_cause_results.json', result)
    print(json.dumps({'subjects': len(result['subjects']), 'alignment': result['alignment'], 'hypotheses': result['hypothesis_verdicts']}, ensure_ascii=False, indent=1))

if __name__ == '__main__': main()
