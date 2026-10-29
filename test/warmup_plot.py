"""warm 扫描与集成的可视化。"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
e = json.loads((OUT/'warmup_ensemble_results.json').read_text(encoding='utf-8'))

fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))

x = [0, .25, .5, .75, 1, 2, 3]
lr = [.6311, .6457, .6450, .6414, .6439, .6423, .6393]
gb = [.6293, .6522, .6528, .6447, .6499, .6428, .6439]
ax[0].plot(x, lr, 'o-', color='#4c72b0', label='Logistic300')
ax[0].plot(x, gb, 's-', color='#dd8452', label='HistGB (full-dim)')
ax[0].scatter([0, .25, 1, 2], [.6439, .6478, .6442, .6409], marker='^', s=75,
              color='#55a868', label='RBF SVC', zorder=5)
ax[0].axvline(.5, ls=':', color='gray')
ax[0].annotate('peak 0.25–0.5 s', (.55, .6305), fontsize=9, color='gray', rotation=90, va='bottom')
ax[0].set(xlabel='Warm-up length for window z-score (s; 0 = no normalization)',
          ylabel='OOF AUC', ylim=(.625, .658),
          title='Warm-up length sweep (subject-grouped 5-fold)')
ax[0].legend(fontsize=8.5, loc='lower right')
ax[0].grid(alpha=.25)

keys = list(e.keys())
keys = [k for k in keys if isinstance(e[k], dict) and 'AUC' in e[k]]
labels = ['A\nwarm1\n(0.6635)', 'B\nwarm0.5', 'C\n+svc', 'D\n+svc\n+lda',
          'E\n+svc\n+consist', 'F\n0.25\ntrio', 'G\nF\n+base']
auc = [e[k]['AUC'] for k in keys]
bacc = [e[k]['bACC'] for k in keys]
i = np.arange(len(keys))
ax[1].bar(i-.19, auc, .38, label='AUC', color='#4c72b0')
ax[1].bar(i+.19, bacc, .38, label='bACC', color='#dd8452')
for j, (a, b) in enumerate(zip(auc, bacc)):
    ax[1].text(j-.19, a+.0007, f'{a:.4f}', ha='center', fontsize=7.5, rotation=90)
ax[1].axhline(.6590, ls='--', lw=1, color='#4c72b0', alpha=.6)
ax[1].set(xticks=i, xticklabels=labels, ylim=(.60, .678),
          title='Pre-specified ensembles vs the 0.6635 recipe (A)')
ax[1].tick_params(axis='x', labelsize=7.5)
ax[1].legend(fontsize=8.5, loc='lower right')
ax[1].grid(axis='y', alpha=.25)

fig.tight_layout()
fig.savefig(OUT/'warmup_sweep_comparison.png', dpi=170)
print('saved', OUT/'warmup_sweep_comparison.png', keys)
