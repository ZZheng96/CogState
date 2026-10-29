"""B 支集成扩展：用已保存的 OOF 测试若干预注册组合，不重新训练。"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

OUT = Path(__file__).resolve().parent
d = np.load(OUT/'b_branch_oof.npz')
y, g = d['y'], d['groups']; n = len(y)
pool = {k: d[k] for k in d.files if k not in ('y', 'groups')}
R = {k: rankdata(v)/n for k, v in pool.items()}


def smooth(s, k):
    out = np.empty_like(s)
    for sub in np.unique(g):
        idx = np.where(g == sub)[0]
        v = s[idx]; c = np.cumsum(np.r_[0., v])
        lo = np.maximum(np.arange(len(v))-k+1, 0)
        out[idx] = (c[np.arange(len(v))+1]-c[lo])/(np.arange(len(v))+1-lo)
    return out


COMBOS = {
    'M1 · 现行四方（base_histgb+warm_lr+warm_histgb+base_svc）':
        ['K120_histgb', 'warm_lr', 'warm_histgb', 'base_svc'],
    'M2 · M1 + 稳健校准模型': ['K120_histgb', 'warm_lr', 'warm_histgb', 'base_svc', 'K30_robust_histgb'],
    'M3 · 稳健校准主导（robust+histgb120+warm_lr+base_svc）':
        ['K30_robust_histgb', 'K120_histgb', 'warm_lr', 'base_svc'],
    'M4 · M1 + 稳健 + warm_svc': ['K120_histgb', 'warm_lr', 'warm_histgb', 'base_svc',
                                   'K30_robust_histgb', 'warm_svc'],
}

res = {}
for name, keys in COMBOS.items():
    keys = [k for k in keys if k in R]
    s = np.mean([R[k] for k in keys], axis=0)
    row = {'raw': {'AUC': float(roc_auc_score(y, s)), 'bACC': float(balanced_accuracy_score(y, s >= .5))}}
    for k in (4, 5):
        sm = smooth(s, k)
        row[f'smooth{k}'] = {'AUC': float(roc_auc_score(y, sm)),
                             'bACC': float(balanced_accuracy_score(y, sm >= .5))}
    res[name] = row
    print(f'[{name}]')
    print(f'    不平滑 AUC={row["raw"]["AUC"]:.4f} bACC={row["raw"]["bACC"]:.4f} | '
          f'24s AUC={row["smooth4"]["AUC"]:.4f} bACC={row["smooth4"]["bACC"]:.4f} | '
          f'30s AUC={row["smooth5"]["AUC"]:.4f} bACC={row["smooth5"]["bACC"]:.4f}', flush=True)

best = max(res, key=lambda k: res[k]['smooth4']['AUC'])
print(f'\n[best] {best} 24s AUC={res[best]["smooth4"]["AUC"]:.4f}', flush=True)
res['_best'] = best
(OUT/'b_ensemble_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
