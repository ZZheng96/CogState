"""B 支：把分位数校准模型纳入集成，并做剔除校准窗的复核。

分位数校准（把每个特征映射为本人校准分布中的百分位）在单模型上明显优于
均值/方差 z-score：quant120 HistGB 0.7026 vs z K=120 0.6762。
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

OUT = Path(__file__).resolve().parent
B = np.load(OUT/'b_branch_oof.npz')
Q = np.load(OUT/'b_calib_variants_oof.npz')
y, g = B['y'], B['groups']; n = len(y)

pool = {k: B[k] for k in B.files if k not in ('y', 'groups')}
pool.update({k: Q[k] for k in Q.files if k not in ('y', 'groups')})
R = {k: rankdata(v)/n for k, v in pool.items()}

pos = np.empty(n, dtype=int)
for s in np.unique(g):
    i = np.where(g == s)[0]; pos[i] = np.arange(len(i))


def smooth(s, k):
    out = np.empty_like(s)
    for sub in np.unique(g):
        idx = np.where(g == sub)[0]; v = s[idx]
        c = np.cumsum(np.r_[0., v])
        lo = np.maximum(np.arange(len(v))-k+1, 0)
        out[idx] = (c[np.arange(len(v))+1]-c[lo])/(np.arange(len(v))+1-lo)
    return out


def metric(s, mask=None):
    m = np.ones(n, bool) if mask is None else mask
    return {'AUC': float(roc_auc_score(y[m], s[m])),
            'bACC': float(balanced_accuracy_score(y[m], s[m] >= .5))}


COMBOS = {
    'N1 · quant120+quant30+quant30lr+svc':
        ['quant120_histgb', 'quant30_histgb', 'quant30_lr', 'base_svc'],
    'N2 · N1 + quant60': ['quant120_histgb', 'quant30_histgb', 'quant60_histgb', 'quant30_lr', 'base_svc'],
    'N3 · quant120+quant60+robust30+quant30lr+svc':
        ['quant120_histgb', 'quant60_histgb', 'K30_robust_histgb', 'quant30_lr', 'base_svc'],
    'N4 · quant 全家族 + warm_svc':
        ['quant120_histgb', 'quant60_histgb', 'quant30_histgb', 'quant30_lr', 'warm_svc'],
}

res, scores = {}, {}
for name, keys in COMBOS.items():
    keys = [k for k in keys if k in R]
    s = np.mean([R[k] for k in keys], axis=0); scores[name] = s
    row = {'keys': keys, 'raw': metric(s)}
    for k in (4, 5):
        row[f'smooth{k}'] = metric(smooth(s, k))
    row['excl120_smooth4'] = metric(smooth(s, 4), pos >= 120)
    row['excl120_raw'] = metric(s, pos >= 120)
    res[name] = row
    print(f'[{name}]')
    print(f'    不平滑 {row["raw"]["AUC"]:.4f}/{row["raw"]["bACC"]:.4f} | '
          f'24s {row["smooth4"]["AUC"]:.4f}/{row["smooth4"]["bACC"]:.4f} | '
          f'30s {row["smooth5"]["AUC"]:.4f}/{row["smooth5"]["bACC"]:.4f} | '
          f'剔校准后 24s {row["excl120_smooth4"]["AUC"]:.4f}/{row["excl120_smooth4"]["bACC"]:.4f}', flush=True)

best = max(res, key=lambda k: res[k]['smooth4']['AUC'])
res['_best'] = best
print(f'\n[best] {best}\n  全窗 24s AUC={res[best]["smooth4"]["AUC"]:.4f} bACC={res[best]["smooth4"]["bACC"]:.4f}'
      f'\n  剔校准 24s AUC={res[best]["excl120_smooth4"]["AUC"]:.4f} '
      f'bACC={res[best]["excl120_smooth4"]["bACC"]:.4f}', flush=True)
(OUT/'b_ensemble2_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
