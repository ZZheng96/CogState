"""可分性评估图：pooled/受试者内 AUC 分布、二者关系、最佳特征的分布重叠。"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
s = np.load(OUT/'separability_scores.npz')
pw, mw, md, ag = s['pooled'], s['mean_within'], s['median_within'], s['agree']
X_cache = OUT/'features_v6.npz'
r = json.loads((OUT/'separability_results.json').read_text(encoding='utf-8'))

fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))

# A. pooled AUC 分布
ax[0].hist(pw, bins=60, color='#4c72b0')
ax[0].axvline(.65, color='#c44e52', ls='--', lw=1.2)
ax[0].axvline(float(pw.max()), color='#55a868', ls='-', lw=1.2)
ax[0].annotate(f'max {pw.max():.3f}', (pw.max(), ax[0].get_ylim()[1]*.9), fontsize=9,
               color='#55a868', ha='right')
ax[0].set(xlabel='Pooled single-feature AUC', ylabel='#features',
          title=f'No single feature exceeds 0.61  (n={len(pw)})')
ax[0].grid(alpha=.25)

# B. 受试者内 AUC vs 跨受试者方向一致性
ax[1].scatter(mw, ag, s=6, alpha=.35, color='#4c72b0', edgecolors='none')
ax[1].axhline(.85, color='#c44e52', ls='--', lw=1)
ax[1].axvline(.65, color='#c44e52', ls='--', lw=1)
ax[1].annotate('useful region\n(empty)', (.66, .86), fontsize=9, color='#c44e52')
ax[1].set(xlabel='Mean within-subject AUC', ylabel='Direction consistency across subjects',
          title='Best within-subject features flip sign across subjects', ylim=(.45, 1.0))
ax[1].grid(alpha=.25)

# C. 子集维度 vs AUC
ks = sorted(int(k) for k in r['subset_auc'])
vals = [r['subset_auc'][str(k)] for k in ks]
ax[2].plot(ks, vals, 'o-', color='#4c72b0', label='top-k by consistency score')
ax[2].axhline(.65, color='#c44e52', ls='--', lw=1.2, label='full feature set (0.65)')
ax[2].set(xlabel='Subset size (features)', ylabel='Grouped-CV AUC', xscale='log',
          title='Small "clean" subsets are worse than using all features', ylim=(.55, .68))
ax[2].legend(fontsize=8.5, loc='lower right')
ax[2].grid(alpha=.25)

fig.tight_layout()
fig.savefig(OUT/'separability.png', dpi=170)
print('saved')

print('silhouette', r['silhouette_top20'])
print('counts', r['counts'])
print('best single', r['best_single'])
