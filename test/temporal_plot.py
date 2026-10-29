"""时序积分可视化：平滑窗长曲线与逐折对照。"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
r = json.loads((OUT/'temporal_refine_results.json').read_text(encoding='utf-8'))

fig, ax = plt.subplots(1, 2, figsize=(13.5, 5))

ks = list(range(1, 13))
mean = [r[f'mean_K{k}']['AUC'] for k in ks]
med = [r[f'median_K{k}']['AUC'] for k in ks]
x = [k*6 for k in ks]
ax[0].plot(x, mean, 'o-', color='#4c72b0', label='causal moving mean')
ax[0].plot(x, med, 's--', color='#dd8452', label='causal moving median')
ax[0].axhline(0.6629, ls=':', color='gray', label='no smoothing')
ax[0].scatter([24], [0.6799], s=110, facecolors='none', edgecolors='#c44e52', linewidths=2, zorder=5)
ax[0].annotate('0.6799 @ 24 s', (24, .6799), textcoords='offset points', xytext=(8, -14), fontsize=9, color='#c44e52')
ax[0].set(xlabel='Causal integration horizon (s)', ylabel='OOF AUC',
          title='Temporal integration of scores (label-free, strictly causal)', ylim=(.64, .69))
ax[0].legend(fontsize=8.5, loc='lower left')
ax[0].grid(alpha=.25)

i = np.arange(5)
raw = r['_per_fold']['raw']
sm = r['_per_fold']['smoothed']
ax[1].bar(i-.19, raw, .38, label='no smoothing', color='#9aa7b8')
ax[1].bar(i+.19, sm, .38, label='causal mean, K=4 (24 s)', color='#4c72b0')
for j in range(5):
    ax[1].text(j-.19, raw[j]+.004, f'{raw[j]:.3f}', ha='center', fontsize=7.5, rotation=90)
    ax[1].text(j+.19, sm[j]+.004, f'{sm[j]:.3f}', ha='center', fontsize=7.5, rotation=90)
ax[1].set(xticks=i, xticklabels=[f'fold {k}' for k in range(1, 6)], ylim=(.55, .78),
          title='Per-fold AUC: all five folds improve')
ax[1].legend(fontsize=8.5, loc='upper left')
ax[1].grid(axis='y', alpha=.25)

fig.tight_layout()
fig.savefig(OUT/'temporal_integration.png', dpi=170)
print('saved')
