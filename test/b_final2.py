"""B 支最终集成 v2：纳入分位数校准的 SVC 与暖启动特征。"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

OUT = Path(__file__).resolve().parent
files = ['b_branch_oof.npz', 'b_calib_variants_oof.npz', 'b_calib2_oof.npz']
pool = {}
for f in files:
    d = np.load(OUT/f)
    y, g = d['y'], d['groups']
    pool.update({k: d[k] for k in d.files if k not in ('y', 'groups')})
n = len(y); R = {k: rankdata(v)/n for k, v in pool.items()}

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


COMBOS = {
    'P1 · q120histgb+q120svc+z120histgb+zsvc':
        ['quant120_histgb', 'quant120_svc', 'K120_histgb', 'base_svc'],
    'P2 · P1 + q120warm': ['quant120_histgb', 'quant120_svc', 'K120_histgb', 'base_svc', 'quant120w_histgb'],
    'P3 · q120histgb+q120warm+q120svc+zsvc':
        ['quant120_histgb', 'quant120w_histgb', 'quant120_svc', 'base_svc'],
    'P0 · 上一轮最优（q120+z120+svc+warm）':
        ['quant120_histgb', 'K120_histgb', 'base_svc', 'warm_histgb'],
}

res, scores = {}, {}
for name, keys in COMBOS.items():
    keys = [k for k in keys if k in R]
    s = np.mean([R[k] for k in keys], axis=0); scores[name] = s
    ss = smooth(s, 4)
    row = {'keys': keys,
           'full_smooth24': float(roc_auc_score(y, ss)),
           'excl_smooth24': float(roc_auc_score(y[mask], ss[mask])),
           'excl_bACC': float(balanced_accuracy_score(y[mask], ss[mask] >= .5)),
           'excl_raw': float(roc_auc_score(y[mask], s[mask]))}
    res[name] = row
    print(f'[{name}]\n    全窗24s {row["full_smooth24"]:.4f} | 剔校准24s {row["excl_smooth24"]:.4f} '
          f'/ bACC {row["excl_bACC"]:.4f} | 剔校准不平滑 {row["excl_raw"]:.4f}', flush=True)

best = max(res, key=lambda k: res[k]['excl_smooth24'])
res['_best'] = best
print(f'\n[best] {best} 剔校准24s AUC={res[best]["excl_smooth24"]:.4f}', flush=True)

splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(y[:, None], y, g))
ss = smooth(scores[best], 4)
per = []
for _, te in splits:
    sel = te[mask[te]]
    if len(set(y[sel])) == 2:
        per.append(float(roc_auc_score(y[sel], ss[sel])))
res['_per_fold'] = per
print(f'[逐折 剔校准] {np.round(per, 3).tolist()} 均值={np.mean(per):.4f}', flush=True)

# 受试者级 bootstrap：与上一轮最优比较
old = smooth(scores['P0 · 上一轮最优（q120+z120+svc+warm）'], 4)
subs = np.unique(g); idx_by = [np.where(g == s)[0] for s in subs]
idx_by = [i[i >= 120] for i in idx_by]
rng = np.random.default_rng(0); d = []
for _ in range(500):
    pick = rng.integers(0, len(subs), len(subs))
    idx = np.concatenate([idx_by[i] for i in pick])
    if len(set(y[idx])) < 2:
        continue
    d.append(roc_auc_score(y[idx], ss[idx])-roc_auc_score(y[idx], old[idx]))
d = np.array(d)
res['_vs_P0'] = {'mean': float(d.mean()), 'p2.5': float(np.percentile(d, 2.5)),
                 'p97.5': float(np.percentile(d, 97.5)), 'win_rate': float((d > 0).mean())}
print(f'[vs 上一轮最优] ΔAUC={d.mean():+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}] '
      f'胜率={100*(d > 0).mean():.0f}%', flush=True)
(OUT/'b_final2_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
