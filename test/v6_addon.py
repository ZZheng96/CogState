"""预注册检验：v6 模型能否给当前主方案（histgb+lda+svc）带来增量。

只评估事先指定的三个组合，不做组合搜索。
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

OUT = Path(__file__).resolve().parent
C = np.load(OUT/'database_combined_oof.npz'); M = np.load(OUT/'database_models_oof.npz')
K = np.load(OUT/'kernel_trees_oof.npz'); V6 = np.load(OUT/'features_v6_oof.npz')
y, g = C['y'], C['groups']; n = len(y)

R = {k: rankdata(v)/n for k, v in {
    'histgb': C['combined_histgb'], 'lda': M['lda'], 'svc': K['svc_rbf_c1'],
    'svc_v6': V6['base_v6_svc'], 'histgb_v6': V6['base_v6_histgb'],
}.items()}

MAIN = ('histgb', 'lda', 'svc')
TESTS = {
    'main（histgb+lda+svc）': MAIN,
    'main + svc_v6': MAIN + ('svc_v6',),
    'main + histgb_v6': MAIN + ('histgb_v6',),
}

res, scores = {}, {}
for name, keys in TESTS.items():
    s = np.mean([R[k] for k in keys], axis=0); scores[name] = s
    res[name] = {'AUC': float(roc_auc_score(y, s)),
                 'bACC': float(balanced_accuracy_score(y, s >= .5))}
    print(f'[fixed] {name:24s} AUC={res[name]["AUC"]:.4f} bACC={res[name]["bACC"]:.4f}', flush=True)

print('\n[与主方案的 spearman]', flush=True)
base = scores['main（histgb+lda+svc）']
for k in ('svc_v6', 'histgb_v6'):
    print(f'   {k:12s} {spearmanr(base, R[k]).statistic:.3f}', flush=True)

rng = np.random.default_rng(0)
splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(y[:, None], y, g))
for name in ('main + svc_v6', 'main + histgb_v6'):
    d = []
    for _ in range(600):
        idx = rng.integers(0, n, n)
        if len(set(y[idx])) < 2:
            continue
        d.append(roc_auc_score(y[idx], scores[name][idx]) - roc_auc_score(y[idx], base[idx]))
    d = np.array(d)
    per = [roc_auc_score(y[te], scores[name][te]) - roc_auc_score(y[te], base[te]) for _, te in splits]
    res[name]['delta'] = {'mean': float(d.mean()), 'p2.5': float(np.percentile(d, 2.5)),
                          'p97.5': float(np.percentile(d, 97.5)), 'win_rate': float((d > 0).mean()),
                          'per_fold': [float(x) for x in per]}
    print(f'\n[{name}] ΔAUC={d.mean():+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}] '
          f'胜率={100*(d > 0).mean():.0f}%', flush=True)
    print(f'   分折 {np.round(per, 4).tolist()}', flush=True)

(OUT/'v6_addon_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
print('\n[done]', flush=True)
