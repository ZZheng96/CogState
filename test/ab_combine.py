"""A 支（未校准）模型能否给 B 支（校准）集成带来互补。

A 支模型与 B 支用同一套特征与同一划分，唯一差别是 A 不做逐受试者校准。
两者若错误模式不同，加入后应有增益。
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

OUT = Path(__file__).resolve().parent
pool, y, g = {}, None, None
for f in ('b_branch_oof.npz', 'b_calib_variants_oof.npz', 'b_calib2_oof.npz', 'temporal_integration_oof.npz'):
    d = np.load(OUT/f)
    y, g = d['y'], d['groups']
    pool.update({k: d[k] for k in d.files if k not in ('y', 'groups')})
n = len(y)
print(f'[pool] {sorted(pool)}', flush=True)

pos = np.empty(n, dtype=int); streams = {}
for s in np.unique(g):
    i = np.where(g == s)[0]; pos[i] = np.arange(len(i)); streams[s] = i
mask = pos >= 120


def moving(s, k):
    out = np.empty_like(s)
    for idx in streams.values():
        v = s[idx]; c = np.cumsum(np.r_[0., v])
        lo = np.maximum(np.arange(len(v))-k+1, 0)
        out[idx] = (c[np.arange(len(v))+1]-c[lo])/(np.arange(len(v))+1-lo)
    return out


B = ['quant120_histgb', 'quant120w_histgb', 'quant120_svc', 'base_svc']
A = [k for k in ('base_histgb', 'base_svc', 'warm_lr', 'warm_histgb') if k in pool]
# 注意：base_svc 同时出现在两族里，A 支单独引用会重复计数，这里 A 只取三个非重复成员
A = [k for k in A if k != 'base_svc']

R = {k: rankdata(pool[k])/n for k, v in pool.items()}


def ev(keys, k=5):
    s = moving(np.mean([R[x] for x in keys], axis=0), k)
    return {'AUC': float(roc_auc_score(y[mask], s[mask])),
            'bACC': float(balanced_accuracy_score(y[mask], s[mask] >= .5)),
            'full_AUC': float(roc_auc_score(y, s))}


CANDS = {
    'B 四路（现行）': B,
    'B + A_histgb': B + ['base_histgb'],
    'B + A_warm_lr': B + ['warm_lr'],
    'B + A 三路': B + A,
}
res = {}
for name, keys in CANDS.items():
    r = ev(keys); res[name] = r
    print(f'[{name:16s}] 剔校准 AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f} | 全窗 {r["full_AUC"]:.4f}', flush=True)

base = moving(np.mean([R[x] for x in B], axis=0), 5)
best = max(res, key=lambda k: res[k]['AUC'])
cand = moving(np.mean([R[x] for x in CANDS[best]], axis=0), 5)
subs = np.unique(g); idx_by = [streams[s][streams[s] >= 120] for s in subs]
rng = np.random.default_rng(0); d = []
for _ in range(500):
    pick = rng.integers(0, len(subs), len(subs))
    idx = np.concatenate([idx_by[i] for i in pick])
    if len(set(y[idx])) < 2:
        continue
    d.append(roc_auc_score(y[idx], cand[idx])-roc_auc_score(y[idx], base[idx]))
d = np.array(d)
res['_best'] = best
res['_vs_B'] = {'mean': float(d.mean()), 'p2.5': float(np.percentile(d, 2.5)),
                'p97.5': float(np.percentile(d, 97.5)), 'win_rate': float((d > 0).mean())}
print(f'\n[best] {best}')
print(f'[vs B 四路] ΔAUC={d.mean():+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}] '
      f'胜率={100*(d > 0).mean():.0f}%', flush=True)
(OUT/'ab_combine_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
