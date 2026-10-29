"""时序积分 v2：变点重置与指数加权。

原方案是固定窗长因果滑动平均，最优 24 s；再长会把块边界两侧的状态混在一起，
所以 AIC 曲线在 24 s 之后单调下降。

本脚本测试两种改进：
  1. EWMA（指数加权）：近期权重更高，天然抑制远端污染；
  2. 变点重置：用"快均值 vs 慢基线"检测状态切换，一旦触发就清空累加器，
     从而可以在状态内部使用很长的积分窗，而不跨边界混合。

全部只用已到达的分数，不用标签、不用窗口序号。
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

OUT = Path(__file__).resolve().parent
pool = {}
for f in ('b_branch_oof.npz', 'b_calib_variants_oof.npz', 'b_calib2_oof.npz'):
    d = np.load(OUT/f)
    y, g = d['y'], d['groups']
    pool.update({k: d[k] for k in d.files if k not in ('y', 'groups')})
n = len(y)
P3 = ['quant120_histgb', 'quant120w_histgb', 'quant120_svc', 'base_svc']
ens = np.mean([rankdata(pool[k])/n for k in P3], axis=0)

pos = np.empty(n, dtype=int)
streams = {}
for s in np.unique(g):
    i = np.where(g == s)[0]; pos[i] = np.arange(len(i)); streams[s] = i
mask = pos >= 120


def moving(s, k):
    out = np.empty_like(s)
    for idx in streams.values():
        v = s[idx]; c = np.cumsum(np.r_[0., v])
        lo = np.maximum(np.arange(len(v))-k+1, 0)
        out[idx] = (c[np.arange(len(v))+1]-c[lo])/(np.arange(len(v))+1-lo)
    return out


def ewma(s, half_life):
    out = np.empty_like(s); a = 1-np.exp(-np.log(2)/half_life)
    for idx in streams.values():
        v = s[idx]; acc = v[0]
        for j, x in enumerate(v):
            acc = a*x+(1-a)*acc
            out[idx[j]] = acc
    return out


def cusum(s, horizon, thr, ref_hl=180.):
    """快均值与慢基线偏离超过 thr 时判定状态切换，清空累加器。"""
    out = np.empty_like(s); a = 1-np.exp(-np.log(2)/ref_hl)
    for idx in streams.values():
        v = s[idx]
        buf = [v[0]]; ref = v[0]
        for j, x in enumerate(v):
            if len(buf) >= 3 and abs(np.mean(buf[-3:])-ref) > thr:
                buf = []
                ref = x
            buf.append(x)
            if len(buf) > horizon:
                buf.pop(0)
            out[idx[j]] = np.mean(buf)
            ref = a*x+(1-a)*ref
    return out


def rep(s):
    return {'AUC': float(roc_auc_score(y[mask], s[mask])),
            'bACC': float(balanced_accuracy_score(y[mask], s[mask] >= .5)),
            'full_AUC': float(roc_auc_score(y, s))}


res = {}
print('[基准]')
for k in (1, 4, 5, 8):
    r = rep(moving(ens, k)); res[f'moving{k}'] = r
    print(f'  滑均 {k*6:3d}s  AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f}', flush=True)
for hl in (6, 12, 24, 48):
    r = rep(ewma(ens, hl)); res[f'ewma{hl}'] = r
    print(f'  EWMA 半衰期 {hl*6:3d}s  AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f}', flush=True)

print('\n[变点重置 网格]')
best, best_key = -1, None
for horizon in (8, 16, 32, 64):
    for thr in (.10, .15, .20, .30):
        r = rep(cusum(ens, horizon, thr))
        key = f'cusum_h{horizon}_t{int(thr*100)}'
        res[key] = r
        flag = ''
        if r['AUC'] > best:
            best, best_key, flag = r['AUC'], key, '  <='
        print(f'  horizon={horizon*6:3d}s thr={thr:.2f}  AUC={r["AUC"]:.4f} '
              f'bACC={r["bACC"]:.4f}{flag}', flush=True)

print(f'\n[best] {best_key} AUC={res[best_key]["AUC"]:.4f}', flush=True)
res['_best'] = best_key

# 与固定滑均 k=4 做受试者级 bootstrap
h, t = best_key.split('_')[1], best_key.split('_')[2]
cand = cusum(ens, int(h[1:]), int(t[1:])/100)
base = moving(ens, 4)
subs = np.unique(g)
idx_by = [streams[s][streams[s] >= 120] for s in subs]
rng = np.random.default_rng(0); d = []
for _ in range(500):
    pick = rng.integers(0, len(subs), len(subs))
    idx = np.concatenate([idx_by[i] for i in pick])
    if len(set(y[idx])) < 2:
        continue
    d.append(roc_auc_score(y[idx], cand[idx])-roc_auc_score(y[idx], base[idx]))
d = np.array(d)
res['_vs_moving4'] = {'mean': float(d.mean()), 'p2.5': float(np.percentile(d, 2.5)),
                      'p97.5': float(np.percentile(d, 97.5)), 'win_rate': float((d > 0).mean())}
print(f'[vs 24s 滑均] ΔAUC={d.mean():+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}] '
      f'胜率={100*(d > 0).mean():.0f}%', flush=True)

splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(y[:, None], y, g))
per = [float(roc_auc_score(y[te[mask[te]]], cand[te[mask[te]]])) for _, te in splits
       if len(set(y[te[mask[te]]])) == 2]
res['_per_fold'] = per
print(f'[逐折] {np.round(per, 3).tolist()} 均值={np.mean(per):.4f}', flush=True)
(OUT/'temporal_v2_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
