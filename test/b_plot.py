"""B 支提升的可视化：校准窗长扫描 + 逐级叠加。"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
r = json.loads((OUT/'b_branch_results.json').read_text(encoding='utf-8'))

fig, ax = plt.subplots(1, 2, figsize=(13.5, 5))

# 左：校准窗长扫描
x = [15, 30, 60, 120]
gb = [r[f'K{k}_histgb']['AUC'] for k in x]
lr = [r[f'K{k}_lr']['AUC'] for k in x]
ax[0].plot(x, gb, 'o-', color='#dd8452', label='HistGB (full-dim)')
ax[0].plot(x, lr, 's-', color='#4c72b0', label='Logistic300')
ax[0].axhline(r['K30_robust_histgb']['AUC'], ls='--', color='#55a868', lw=1.2,
              label=f"robust K=30 ({r['K30_robust_histgb']['AUC']:.3f}, inflated +0.07)")
ax[0].axhline(0.6827, ls=':', color='gray', lw=1.2, label='old B branch (0.683)')
ax[0].set(xlabel='Calibration length K (windows, 6 s each)', ylabel='OOF AUC', xscale='log',
          title='B branch: calibration length sweep', ylim=(.60, .695))
ax[0].set_xticks(x); ax[0].set_xticklabels([str(k) for k in x])
ax[0].legend(fontsize=8.5, loc='lower right'); ax[0].grid(alpha=.25)

# 右：逐级叠加（统一用"剔除校准窗"口径，初版以虚线作参考）
stages = ['ensemble\n+24 s integ.', 'quantile\ncalibration', 'B v2\n(3 quantile + 1 svc)']
auc = [0.7243, 0.7322, 0.7401]
bacc = [0.6744, 0.6796, 0.6871]
i = np.arange(len(stages))
ax[1].bar(i, auc, .5, label='AUC (calib. windows excluded)', color='#4c72b0')
for j, (a, b) in enumerate(zip(auc, bacc)):
    ax[1].text(j, a+.003, f'{a:.3f}', ha='center', fontsize=8.5)
    ax[1].text(j, a-.012, f'bACC {b:.3f}', ha='center', fontsize=7.5, color='white')
ax[1].axhline(0.6827, ls='--', color='#c44e52', lw=1.2, label='old B 0.683 (original protocol)')
ax[1].set(xticks=i, xticklabels=stages, ylim=(.60, .78),
          title='B branch: stage-by-stage (honest protocol)')
ax[1].tick_params(axis='x', labelsize=8); ax[1].legend(fontsize=8, loc='upper left')
ax[1].grid(axis='y', alpha=.25)

fig.tight_layout()
fig.savefig(OUT/'b_branch.png', dpi=170)
print('saved')
