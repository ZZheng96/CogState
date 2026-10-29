"""按受试者重采样的 bootstrap：窗口级重采样会低估受试者内相关，结论偏乐观。

对 warm 集成与复刻配方（A）比较，重采样单位改为受试者。
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

OUT = Path(__file__).resolve().parent
W = np.load(OUT/'warmup_sweep_oof.npz'); K = np.load(OUT/'kernel_trees_oof.npz')
S = np.load(OUT/'consistency_oof.npz')
y, g = W['y'], W['groups']; n = len(y)
pool = {k: W[k] for k in W.files if k not in ('y', 'groups')}
pool['svc_base'] = K['svc_rbf_c1']; pool['consist'] = S['consist_lr_300']
R = {k: rankdata(v)/n for k, v in pool.items()}

COMBOS = {
    'A': ('base_histgb', 'warm1_lr', 'warm1_histgb'),
    'C': ('base_histgb', 'warm0.5_lr', 'warm0.5_histgb', 'svc_base'),
    'E': ('base_histgb', 'warm0.5_lr', 'warm0.5_histgb', 'svc_base', 'consist'),
}
sc = {k: np.mean([R[x] for x in v], axis=0) for k, v in COMBOS.items()}
for k, s in sc.items():
    print(f'[{k}] AUC={roc_auc_score(y, s):.4f} bACC={balanced_accuracy_score(y, s >= .5):.4f}', flush=True)

subs = np.unique(g)
idx_by_sub = [np.where(g == s)[0] for s in subs]
rng = np.random.default_rng(0)
out = {}
for k in ('C', 'E'):
    d, db = [], []
    for _ in range(500):
        pick = rng.integers(0, len(subs), len(subs))
        idx = np.concatenate([idx_by_sub[i] for i in pick])
        if len(set(y[idx])) < 2:
            continue
        d.append(roc_auc_score(y[idx], sc[k][idx]) - roc_auc_score(y[idx], sc['A'][idx]))
        db.append(balanced_accuracy_score(y[idx], sc[k][idx] >= .5) - balanced_accuracy_score(y[idx], sc['A'][idx] >= .5))
    d, db = np.array(d), np.array(db)
    out[k] = {'dAUC_mean': float(d.mean()), 'dAUC_p2.5': float(np.percentile(d, 2.5)),
              'dAUC_p97.5': float(np.percentile(d, 97.5)), 'win_rate': float((d > 0).mean()),
              'dbACC_mean': float(db.mean()), 'dbACC_p2.5': float(np.percentile(db, 2.5)),
              'dbACC_p97.5': float(np.percentile(db, 97.5))}
    print(f'[{k} vs A 受试者级] ΔAUC={d.mean():+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}] '
          f'胜率={100*(d > 0).mean():.0f}% | ΔbACC={db.mean():+.4f} '
          f'[{np.percentile(db, 2.5):+.4f}, {np.percentile(db, 97.5):+.4f}]', flush=True)

(OUT/'warmup_subject_bootstrap.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
print('[done]', flush=True)
