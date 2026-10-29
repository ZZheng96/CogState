"""B 支的诚实性检验：把用于校准的窗从评估中剔除，只看校准之后的窗。

B 支用每个受试者开头 K 个窗估计均值/方差。若这 K 个窗也进入评估，
它们与标准化参数共享数据，分数会偏高。这里只评估 index >= K 的窗。
"""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

OUT = Path(__file__).resolve().parent
d = np.load(OUT/'b_branch_oof.npz')
y, g = d['y'], d['groups']
keys = ['K120_histgb', 'warm_lr', 'warm_histgb', 'base_svc']
from scipy.stats import rankdata
n = len(y)
ens = np.mean([rankdata(d[k])/n for k in keys], axis=0)


def smooth(s, k):
    out = np.empty_like(s)
    for sub in np.unique(g):
        idx = np.where(g == sub)[0]
        v = s[idx]
        c = np.cumsum(np.r_[0., v])
        lo = np.maximum(np.arange(len(v))-k+1, 0)
        out[idx] = (c[np.arange(len(v))+1]-c[lo])/(np.arange(len(v))+1-lo)
    return out


# 每个受试者内部的位置序号
pos = np.empty(n, dtype=int)
for sub in np.unique(g):
    idx = np.where(g == sub)[0]
    pos[idx] = np.arange(len(idx))

res = {}
for K in (30, 60, 120):
    keep = pos >= K
    for k in (1, 4, 5):
        s = smooth(ens, k)
        r = {'AUC': float(roc_auc_score(y[keep], s[keep])),
             'bACC': float(balanced_accuracy_score(y[keep], s[keep] >= .5)),
             'n': int(keep.sum())}
        res[f'calib{K}_smooth{k}'] = r
        print(f'[校准 {K:3d} 窗, 时序积分 {k*6:2d}s] AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f} '
              f'n={r["n"]}（原评估 16067 窗）', flush=True)

print(f'\n[参考] 全窗评估 + 24s 积分 = 0.7291', flush=True)
(OUT/'b_exclusion_check.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
