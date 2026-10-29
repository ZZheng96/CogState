"""warm 归一化的集成检验：全部在 database 口径、同一划分下比较。

预注册 5 个组合，不做候选搜索：
  A  warm1 复刻 0.6635 配方（base_histgb + warm1_lr + warm1_histgb）
  B  换成 0.5 s 预热（同一配方）
  C  B + 基线特征上的 RBF SVC（前一阶段的最好单模型）
  D  B + warm0.5 的 SVC/LDA（若存在）
  E  C + 一致性选特征模型
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

OUT = Path(__file__).resolve().parent
W = np.load(OUT/'warmup_sweep_oof.npz')
K = np.load(OUT/'kernel_trees_oof.npz')
S = np.load(OUT/'consistency_oof.npz')
y, g = W['y'], W['groups']; n = len(y)

pool = {k: W[k] for k in W.files if k not in ('y', 'groups')}
pool['svc_base'] = K['svc_rbf_c1']
pool['consist'] = S['consist_lr_300']
for extra in ('warmup_more_oof.npz',):
    p = OUT/extra
    if p.exists():
        d = np.load(p)
        pool.update({k: d[k] for k in d.files if k not in ('y', 'groups')})

R = {k: rankdata(v)/n for k, v in pool.items()}
print(f'[pool] {len(pool)} 个候选: {sorted(pool)}', flush=True)

COMBOS = {
    'A · warm1 复刻 0.6635': ('base_histgb', 'warm1_lr', 'warm1_histgb'),
    'B · warm0.5 同配方': ('base_histgb', 'warm0.5_lr', 'warm0.5_histgb'),
    'C · B + svc_base': ('base_histgb', 'warm0.5_lr', 'warm0.5_histgb', 'svc_base'),
    'D · B + svc_base + lda': ('base_histgb', 'warm0.5_lr', 'warm0.5_histgb', 'svc_base', 'warm1_lda'),
    'E · C + consist': ('base_histgb', 'warm0.5_lr', 'warm0.5_histgb', 'svc_base', 'consist'),
    'F · warm0.25 三模型': ('warm0.25_histgb', 'warm0.25_lr', 'warm0.25_svc'),
    'G · F + base_histgb': ('warm0.25_histgb', 'warm0.25_lr', 'warm0.25_svc', 'base_histgb'),
}
COMBOS = {k: tuple(x for x in v if x in pool) for k, v in COMBOS.items()}

splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(y[:, None], y, g))
res, scores = {}, {}
for name, keys in COMBOS.items():
    s = np.mean([R[k] for k in keys], axis=0); scores[name] = s
    res[name] = {'keys': list(keys), 'AUC': float(roc_auc_score(y, s)),
                 'bACC': float(balanced_accuracy_score(y, s >= .5))}
    print(f'[combo] {name:24s} AUC={res[name]["AUC"]:.4f} bACC={res[name]["bACC"]:.4f}', flush=True)

ref = 'A · warm1 复刻 0.6635'
rng = np.random.default_rng(0)
print(f'\n[相对 A 的差值]', flush=True)
for name in COMBOS:
    if name == ref:
        continue
    d = []
    for _ in range(600):
        idx = rng.integers(0, n, n)
        if len(set(y[idx])) < 2:
            continue
        d.append(roc_auc_score(y[idx], scores[name][idx]) - roc_auc_score(y[idx], scores[ref][idx]))
    d = np.array(d)
    per = [roc_auc_score(y[te], scores[name][te]) - roc_auc_score(y[te], scores[ref][te]) for _, te in splits]
    res[name]['delta_vs_A'] = {'mean': float(d.mean()), 'p2.5': float(np.percentile(d, 2.5)),
                               'p97.5': float(np.percentile(d, 97.5)), 'win_rate': float((d > 0).mean()),
                               'per_fold': [float(x) for x in per]}
    print(f'  {name:24s} Δ={d.mean():+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}] '
          f'胜率={100*(d > 0).mean():.0f}% 分折={np.round(per, 4).tolist()}', flush=True)

best = max(res, key=lambda k: res[k]['AUC'])
print(f'\n[best] {best} AUC={res[best]["AUC"]:.4f} bACC={res[best]["bACC"]:.4f}', flush=True)
print(f'[与 A 的相关] {spearmanr(scores[best], scores[ref]).statistic:.3f}', flush=True)
res['_best'] = best
(OUT/'warmup_ensemble_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
print('[done]', flush=True)
