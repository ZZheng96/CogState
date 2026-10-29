"""不搜索的固定组合检验：只评估事先指定的组合，并用 bootstrap 给出区间。

前几轮的"最佳组合"是在大量候选上搜出来的，存在选择性偏差。
这里锁定组合，只看新增模型是否带来可重复的增量。
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT/'test'
C = np.load(OUT/'database_combined_oof.npz')
M = np.load(OUT/'database_models_oof.npz')
K = np.load(OUT/'kernel_trees_oof.npz')
S = np.load(OUT/'consistency_oof.npz')
y, g = C['y'], C['groups']
n = len(y)

pool = {'histgb': C['combined_histgb'], 'lda': M['lda'], 'consist': S['consist_lr_300'],
        'consist_lda': S['consist_lda_300'], 'svc': K['svc_rbf_c1']}
R = {k: rankdata(v)/n for k, v in pool.items()}


def rep(s, thr=0.5):
    return {'AUC': float(roc_auc_score(y, s)), 'bACC': float(balanced_accuracy_score(y, s >= thr))}


FIXED = {
    'histgb+lda（当前主方案）': ('histgb', 'lda'),
    'histgb+lda+consist': ('histgb', 'lda', 'consist'),
    'histgb+lda+svc': ('histgb', 'lda', 'svc'),
    'histgb+lda+consist+svc': ('histgb', 'lda', 'consist', 'svc'),
}
res, scores = {}, {}
for name, keys in FIXED.items():
    s = np.mean([R[k] for k in keys], axis=0)
    scores[name] = s
    res[name] = rep(s)
    print(f'[fixed] {name:32s} AUC={res[name]["AUC"]:.4f} bACC={res[name]["bACC"]:.4f}', flush=True)

print('\n[与主方案的打分相关（越低越互补）]', flush=True)
base = scores['histgb+lda（当前主方案）']
for k in ('histgb', 'lda', 'consist', 'consist_lda', 'svc'):
    print(f'   {k:12s} spearman={spearmanr(base, R[k]).statistic:.3f}', flush=True)

print('\n[相对主方案的 bootstrap 差值]', flush=True)
rng = np.random.default_rng(0)
for name in ('histgb+lda+consist', 'histgb+lda+svc', 'histgb+lda+consist+svc'):
    d = []
    for _ in range(600):
        idx = rng.integers(0, n, n)
        if len(set(y[idx])) < 2:
            continue
        d.append(roc_auc_score(y[idx], scores[name][idx]) - roc_auc_score(y[idx], base[idx]))
    d = np.array(d)
    res[name]['delta_vs_main'] = {'mean': float(d.mean()), 'p2.5': float(np.percentile(d, 2.5)),
                                  'p97.5': float(np.percentile(d, 97.5)),
                                  'win_rate': float((d > 0).mean())}
    print(f'   {name:26s} ΔAUC={d.mean():+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}] '
          f'胜率={100*(d > 0).mean():.0f}%', flush=True)

(OUT/'fixed_combo_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
print('\n[done]', flush=True)
