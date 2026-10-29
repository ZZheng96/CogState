"""B 支最终集成：只使用校准窗膨胀小的成员（K=120 一档），并给出诚实口径。

成员（全部 K=120 校准，膨胀约 +0.01）：
  quant120_histgb（分位数校准 + HistGB 全维）
  K120_histgb    （均值/方差校准 + HistGB 全维）
  base_svc       （均值/方差校准 + RBF SVC）
  warm_histgb    （0.5 s 窗内暖启动 + 均值/方差校准 + HistGB）
评估：剔除全部 120 个校准窗，只用之后的窗；24 s 因果时序积分。
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

OUT = Path(__file__).resolve().parent
B = np.load(OUT/'b_branch_oof.npz'); Q = np.load(OUT/'b_calib_variants_oof.npz')
y, g = B['y'], B['groups']; n = len(y)
pool = {k: B[k] for k in B.files if k not in ('y', 'groups')}
pool.update({k: Q[k] for k in Q.files if k not in ('y', 'groups')})

pos = np.empty(n, dtype=int)
for s in np.unique(g):
    i = np.where(g == s)[0]; pos[i] = np.arange(len(i))
mask = pos >= 120


def smooth(s, k):
    out = np.empty_like(s)
    for sub in np.unique(g):
        idx = np.where(g == sub)[0]; v = s[idx]
        c = np.cumsum(np.r_[0., v]); lo = np.maximum(np.arange(len(v))-k+1, 0)
        out[idx] = (c[np.arange(len(v))+1]-c[lo])/(np.arange(len(v))+1-lo)
    return out


KEYS = ['quant120_histgb', 'K120_histgb', 'base_svc', 'warm_histgb']
KEYS = [k for k in KEYS if k in pool]
ens = np.mean([rankdata(pool[k])/n for k in KEYS], axis=0)
ens_s = smooth(ens, 4)

res = {}
res['all_windows'] = {'raw': float(roc_auc_score(y, ens)),
                      'smooth24': float(roc_auc_score(y, ens_s)),
                      'bACC_smooth24': float(balanced_accuracy_score(y, ens_s >= .5))}
res['excl_calib'] = {'raw': float(roc_auc_score(y[mask], ens[mask])),
                     'smooth24': float(roc_auc_score(y[mask], ens_s[mask])),
                     'bACC_smooth24': float(balanced_accuracy_score(y[mask], ens_s[mask] >= .5)),
                     'n': int(mask.sum())}
print(f'[成员] {KEYS}', flush=True)
print(f'[全窗]     不平滑 {res["all_windows"]["raw"]:.4f} | 24s {res["all_windows"]["smooth24"]:.4f} '
      f'/ bACC {res["all_windows"]["bACC_smooth24"]:.4f}', flush=True)
print(f'[剔校准]   不平滑 {res["excl_calib"]["raw"]:.4f} | 24s {res["excl_calib"]["smooth24"]:.4f} '
      f'/ bACC {res["excl_calib"]["bACC_smooth24"]:.4f} (n={mask.sum()})', flush=True)

splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(y[:, None], y, g))
per = []
for _, te in splits:
    sel = te[mask[te]]
    if len(set(y[sel])) == 2:
        per.append(float(roc_auc_score(y[sel], ens_s[sel])))
res['per_fold_excl'] = per
print(f'[逐折 剔校准] {np.round(per, 3).tolist()} 均值={np.mean(per):.4f}', flush=True)

# 受试者级 bootstrap：与旧 B（z 四路集成）比较
old = np.mean([rankdata(pool[k])/n for k in ('K120_histgb', 'warm_lr', 'warm_histgb', 'base_svc')], axis=0)
old_s = smooth(old, 4)
subs = np.unique(g)
idx_by = [np.where(g == s)[0] for s in subs]
idx_by = [i[i >= 120] for i in idx_by]
rng = np.random.default_rng(0); d = []
for _ in range(500):
    pick = rng.integers(0, len(subs), len(subs))
    idx = np.concatenate([idx_by[i] for i in pick])
    if len(set(y[idx])) < 2:
        continue
    d.append(roc_auc_score(y[idx], ens_s[idx])-roc_auc_score(y[idx], old_s[idx]))
d = np.array(d)
res['vs_old_ensemble'] = {'mean': float(d.mean()), 'p2.5': float(np.percentile(d, 2.5)),
                          'p97.5': float(np.percentile(d, 97.5)), 'win_rate': float((d > 0).mean())}
print(f'[对比旧 B 集成] ΔAUC={d.mean():+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}] '
      f'胜率={100*(d > 0).mean():.0f}%', flush=True)

np.savez_compressed(OUT/'b_final_oof.npz', y=y, groups=g, ens=ens, ens_smooth=ens_s, pos=pos)
(OUT/'b_final_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
print('[done]', flush=True)
