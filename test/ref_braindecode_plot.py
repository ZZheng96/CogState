"""ref 资料评估结果可视化：柱状对比 + 与历史最好成绩的参照线。"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'test'
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load():
    r = json.loads((OUT / 'ref_braindecode_results.json').read_text(encoding='utf-8'))
    rows = [
        ('基线 858+20 · Logistic300', r['base_logistic300']),
        ('基线 858+20 · HistGB', r['base_histgb']),
        ('EMA α=1e-3 · Logistic300', r['ema1e-3_logistic300']),
        ('EMA α=1e-3 · HistGB', r['ema1e-3_histgb']),
        ('EMA α=3e-4 · Logistic300', r['ema3e-4_logistic300']),
        ('暖启动 1s · Logistic300', r['warm1s_logistic300']),
    ]
    for tag, v in r['shallow'].items():
        rows.append((f'ShallowFBCSPNet {tag}', v))
    if (OUT / 'ref_warmup_followup_results.json').exists():
        f = json.loads((OUT / 'ref_warmup_followup_results.json').read_text(encoding='utf-8'))
        for k in ('warm1s_histgb', 'warm1s_lda', 'base_lda'):
            if k in f:
                rows.append((k.replace('_', ' · '), f[k]))
        if 'best_combo' in f:
            rows.append((f'集成 {f["best_combo"]}', f['combos'][f['best_combo']]))
    return rows


def main():
    rows = load()
    labels = [r[0] for r in rows]
    auc = np.array([r[1]['AUC'] for r in rows])
    bacc = np.array([r[1]['bACC'] for r in rows])
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(11, .5 * len(rows) + 2))
    ax.barh(y - .2, auc, .38, label='AUC', color='#2f6fb3')
    ax.barh(y + .2, bacc, .38, label='bACC', color='#e08b3c')
    for i, (a, b) in enumerate(zip(auc, bacc)):
        ax.text(a + .003, i - .2, f'{a:.4f}', va='center', fontsize=8)
        ax.text(b + .003, i + .2, f'{b:.4f}', va='center', fontsize=8)
    ax.axvline(.5, color='#999', lw=.8, ls='--')
    ax.axvline(.6510, color='#c0392b', lw=1, ls=':')
    ax.text(.6510, len(rows) - .3, ' 既往最好 0.651', color='#c0392b', fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(.45, max(auc.max(), bacc.max()) + .05)
    ax.set_xlabel('指标值')
    ax.set_title('ref（braindecode + Schirrmeister 2017）方案在本赛题上的实测', fontsize=12)
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=.25)
    fig.tight_layout()
    p = OUT / 'ref_braindecode_comparison.png'
    fig.savefig(p, dpi=160)
    print('saved', p)


if __name__ == '__main__':
    main()
