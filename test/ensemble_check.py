"""异构集成评估：rank 平均 + 按折嵌套选择权重，避免用全量 OOF 挑模型造成选择偏差。"""
import json, itertools, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT/'test'
sys.path[:0] = [str(OUT)]
from causal_history import report

A = np.load(OUT/'online_deep_oof.npz'); B = np.load(OUT/'highdim_deep_oof.npz')
y, groups = A['y'], A['groups']
pool = {'online_mlp300': A['online_mlp300'], 'online_logistic300': A['online_logistic300'],
        'large_mlp300': B['large_mlp300'], 'logistic600': B['logistic600'],
        'mlp300': B['mlp300'], 'raw_cnn': B['raw_cnn']}
R = {k: rankdata(v)/len(y) for k, v in pool.items()}
names = list(pool)


def one(sub):
    return np.mean([R[k] for k in sub], axis=0)


# 全量 OOF 上的单模型与全部组合（仅用于展示）
full = {k: report(y, v, groups, .5) for k, v in pool.items()}
combos, subs_list = {}, []
for r in range(2, 5):
    for sub in itertools.combinations(names, r):
        combos['+'.join(sub)] = report(y, one(sub), groups, .5)
        subs_list.append(sub)
best = max(combos, key=lambda k: combos[k]['AUC'])
print(f'[full-OOF] best combo {best} AUC={combos[best]["AUC"]:.4f} bACC={combos[best]["bACC"]:.4f}', flush=True)

# 嵌套：每折用其余 4 折选组合，再在该折上评估
folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(R[names[0]], y, groups))
nested, chosen = [], []
for f, (tr, te) in enumerate(folds, 1):
    inner_best, inner_auc = None, -1
    for sub in subs_list:
        a = roc_auc_score(y[tr], one(sub)[tr])
        if a > inner_auc:
            inner_auc, inner_best = a, sub
    s = one(inner_best)
    nested.append({'fold': f, 'combo': '+'.join(inner_best),
                   'AUC': float(roc_auc_score(y[te], s[te])),
                   'bACC': float(balanced_accuracy_score(y[te], s[te] >= .5))})
    chosen.append('+'.join(inner_best))
    print(f'[nested fold {f}/5] {"+".join(inner_best)} AUC={nested[-1]["AUC"]:.4f}', flush=True)

single_best = report(y, pool['online_mlp300'], groups, .5)
single_fold_auc = [np.mean([d['AUC'] for d in nested if True]),
                   np.mean([report(y[te], pool['online_mlp300'][te], groups[te], .5)['AUC'] for tr, te in folds])]
out = {'single': full, 'combos': combos, 'best_full_combo': best,
       'nested_folds': nested, 'nested_mean_AUC': float(np.mean([d['AUC'] for d in nested])),
       'nested_mean_bACC': float(np.mean([d['bACC'] for d in nested])),
       'nested_vs_single_AUC': float(np.mean([d['AUC'] for d in nested]) - single_fold_auc[1]),
       'online_mlp300_fold_mean_AUC': float(single_fold_auc[1])}
(OUT/'ensemble_results.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
top = sorted(combos, key=lambda k: -combos[k]['AUC'])[:10]
ax[0].barh([k.replace('+', '\n+') for k in top][::-1], [combos[k]['AUC'] for k in top][::-1], color='#4878a8')
ax[0].axvline(full['online_mlp300']['AUC'], color='r', ls='--', label='online_mlp300 single')
ax[0].axvline(0.5, color='k', lw=.8)
ax[0].set(xlabel='pooled AUC', title='Rank-average ensembles (full OOF)'); ax[0].legend()
ax[1].bar([d['fold'] for d in nested], [d['AUC'] for d in nested], color='#e08a3c', label='nested ensemble')
ax[1].axhline(np.mean([d['AUC'] for d in nested]), color='#e08a3c', ls='--', lw=1)
ax[1].axhline(single_fold_auc[1], color='r', ls='--', lw=1, label='online_mlp300')
ax[1].set(xlabel='fold', ylabel='AUC', title='Nested ensemble selection', ylim=(.5, .8)); ax[1].legend()
fig.tight_layout(); fig.savefig(OUT/'ensemble_results.png', dpi=150); plt.close(fig)
print(f'[nested] mean AUC={out["nested_mean_AUC"]:.4f} bACC={out["nested_mean_bACC"]:.4f} '
      f'vs single {single_fold_auc[1]:.4f} (delta {out["nested_vs_single_AUC"]:+.4f})', flush=True)
