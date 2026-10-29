"""A/B 成员组合（修正版）。

注意命名冲突：`temporal_integration_oof.npz` 里的 base_svc / base_histgb 是
A 支（未校准）版本，与 b_branch_oof.npz 的同名 key 含义不同，必须先重命名再合并。
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

OUT = Path(__file__).resolve().parent
pool = {}
for f, prefix in (('b_branch_oof.npz', ''), ('b_calib_variants_oof.npz', ''),
                  ('b_calib2_oof.npz', ''), ('temporal_integration_oof.npz', 'A_')):
    d = np.load(OUT/f)
    y, g = d['y'], d['groups']
    for k in d.files:
        if k in ('y', 'groups'):
            continue
        pool[f'{prefix}{k}'] = d[k]
n = len(y)
print(f'[A 支成员] {sorted(k for k in pool if k.startswith("A_"))}', flush=True)

pos = np.empty(n, dtype=int); streams = {}
for s in np.unique(g):
    i = np.where(g == s)[0]; pos[i] = np.arange(len(i)); streams[s] = i
mask = pos >= 120
R = {k: rankdata(v)/n for k, v in pool.items()}


def moving(s, k):
    out = np.empty_like(s)
    for idx in streams.values():
        v = s[idx]; c = np.cumsum(np.r_[0., v])
        lo = np.maximum(np.arange(len(v))-k+1, 0)
        out[idx] = (c[np.arange(len(v))+1]-c[lo])/(np.arange(len(v))+1-lo)
    return out


B = ['quant120_histgb', 'quant120w_histgb', 'quant120_svc', 'base_svc']
A = ['A_base_histgb', 'A_warm_lr', 'A_warm_histgb', 'A_base_svc']


def ev(keys, k=5):
    s = moving(np.mean([R[x] for x in keys], axis=0), k)
    return {'AUC': float(roc_auc_score(y[mask], s[mask])),
            'bACC': float(balanced_accuracy_score(y[mask], s[mask] >= .5)),
            'full': float(roc_auc_score(y, s))}


CANDS = {
    'B 四路（校准 SVC）': B,
    'B 四路，SVC 换成 A 支': ['quant120_histgb', 'quant120w_histgb', 'quant120_svc', 'A_base_svc'],
    'B 四路 + A_base_svc（五路）': B + ['A_base_svc'],
    'B 四路 + A 全部（八路）': B + A,
    'B 三路 + A_base_svc + A_warm_histgb':
        ['quant120_histgb', 'quant120w_histgb', 'quant120_svc', 'A_base_svc', 'A_warm_histgb'],
}
res = {}
for name, keys in CANDS.items():
    r = ev(keys); res[name] = r
    print(f'[{name:30s}] 剔校准 AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f} | 全窗 {r["full"]:.4f}', flush=True)

best = max(res, key=lambda k: res[k]['AUC'])
res['_best'] = best
print(f'\n[best] {best}  {res[best]["AUC"]:.4f}', flush=True)
(OUT/'ab_combine2_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
