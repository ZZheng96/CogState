"""分清"分位数校准更强"与"分位数校准在校准窗上自我膨胀"。

对每个模型，比较全窗 AUC 与"剔除全部校准窗后"的 AUC。
膨胀量 = 全窗 - 剔校准；只有膨胀小且剔校准后更高，才算真实的校准改进。
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

OUT = Path(__file__).resolve().parent
B = np.load(OUT/'b_branch_oof.npz')
Q = np.load(OUT/'b_calib_variants_oof.npz')
y, g = B['y'], B['groups']; n = len(y)
pool = {k: B[k] for k in B.files if k not in ('y', 'groups')}
pool.update({k: Q[k] for k in Q.files if k not in ('y', 'groups')})

pos = np.empty(n, dtype=int)
for s in np.unique(g):
    i = np.where(g == s)[0]; pos[i] = np.arange(len(i))
mask = pos >= 120


def smooth(s, k):
    out = np.empty_like(s)
    for sub in np.unique(g):
        idx = np.where(g == sub)[0]; v = s[idx]
        c = np.cumsum(np.r_[0., v]); lo = np.maximum(np.arange(len(v))-k+1, 0)
        out[idx] = (c[np.arange(len(v))+1]-c[lo])/(np.arange(len(v))+1-lo)
    return out


MODELS = ['K120_histgb', 'K30_robust_histgb', 'quant120_histgb', 'quant60_histgb',
          'quant30_histgb', 'quant30_lr', 'base_svc']
print(f'{"模型":22s} {"全窗":>7s} {"剔校准":>7s} {"膨胀":>7s}')
rows = {}
for k in MODELS:
    if k not in pool:
        continue
    s = rankdata(pool[k])/n
    a, b = roc_auc_score(y, s), roc_auc_score(y[mask], s[mask])
    rows[k] = {'full': float(a), 'excl120': float(b), 'inflation': float(a-b)}
    print(f'{k:22s} {a:7.4f} {b:7.4f} {a-b:+7.4f}')

print('\n[24s 积分后]')
rows_s = {}
for k in MODELS:
    if k not in pool:
        continue
    s = smooth(rankdata(pool[k])/n, 4)
    a, b = roc_auc_score(y, s), roc_auc_score(y[mask], s[mask])
    rows_s[k] = {'full': float(a), 'excl120': float(b), 'inflation': float(a-b)}
    print(f'{k:22s} {a:7.4f} {b:7.4f} {a-b:+7.4f}')

(OUT/'b_calib_honesty.json').write_text(json.dumps({'raw': rows, 'smooth4': rows_s},
                                                   ensure_ascii=False, indent=2), encoding='utf-8')
