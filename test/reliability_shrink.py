"""可部署性检验：用无标签代理估计"该受试者的分数有多可信"，对不可靠受试者做向 0.5 的收缩。

pooled AUC 把所有受试者的窗口放在一起排序，因此一个判反的受试者会占据排序两端、拉低整体。
若某个受试者的分数分布本身没有双峰，就说明模型在他身上没抓到判别结构，收缩可以减轻损害。
这里先看代理指标与真实每受试者 AUC 的相关性，再看收缩能否改善 pooled 指标。
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT/'test'
sys.path[:0] = [str(ROOT/'src'), str(OUT)]
from causal_history import report

A = np.load(OUT/'online_deep_oof.npz'); B = np.load(OUT/'highdim_deep_oof.npz')
F = np.load(OUT/'highdim_deep_features.npz')
y, groups, X = A['y'], A['groups'], F['X']
s = np.mean([rankdata(A['online_mlp300']), rankdata(A['online_logistic300']),
             rankdata(B['large_mlp300'])], axis=0)/len(y)

import json as _j
names = _j.loads((OUT/'highdim_deep_results.json').read_text(encoding='utf-8'))['feature_names']
alpha = X[:, names.index('stft_log_power_8_13_mean')]

rows = {}
for g in np.unique(groups):
    ids = groups == g
    v = s[ids]
    m, sd = v.mean(), v.std()+1e-12
    z = (v-m)/sd
    rows[g] = {
        'auc': float(roc_auc_score(y[ids], v)),
        'bimodality': float((np.mean(z**3)**2+1)/(np.mean(z**4)+1e-12)),
        'separation': float(np.mean(np.sort(z)[-int(.15*len(z)):]) - np.mean(np.sort(z)[:int(.15*len(z))])),
        'alpha_rel': float(alpha[ids].std()/(abs(alpha[ids].mean())+1e-12)),
    }

print(f'{"subject":>8} {"AUC":>7} {"bimod":>7} {"sep":>7} {"alphaCV":>8}', flush=True)
proxies = ('bimodality', 'separation', 'alpha_rel')
for p in proxies:
    v = np.array([rows[g][p] for g in rows]); a = np.array([rows[g]['auc'] for g in rows])
    rho, pv = spearmanr(v, a)
    print(f'[proxy {p}] spearman={rho:.3f} p={pv:.4f}', flush=True)

res = {'baseline': report(y, s, groups, .5), 'subject_rows': {str(k): v for k, v in rows.items()},
       'proxy_spearman': {p: float(spearmanr([rows[g][p] for g in rows], [rows[g]['auc'] for g in rows])[0]) for p in proxies}}

# 用代理指标做收缩：w 在 [wmin,1] 之间按代理分位线性映射（完全无标签）
for p in proxies:
    for wmin in (.4, .6, .8):
        val = np.array([rows[g][p] for g in rows])
        rk = rankdata(val)/len(val)
        w = {g: wmin + (1-wmin)*rk[i] for i, g in enumerate(rows)}
        t = s.copy()
        for g in rows:
            ids = groups == g
            t[ids] = .5 + w[g]*(s[ids]-.5)
        res[f'shrink_{p}_wmin{wmin}'] = report(y, t, groups, .5)

# 上界：按真实每受试者 AUC 收缩（用真标签，仅参考）
for wmin in (.4, .6, .8):
    rk = rankdata([rows[g]['auc'] for g in rows])/len(rows)
    w = {g: wmin + (1-wmin)*rk[i] for i, g in enumerate(rows)}
    t = s.copy()
    for g in rows:
        ids = groups == g
        t[ids] = .5 + w[g]*(s[ids]-.5)
    res[f'oracle_shrink_wmin{wmin}'] = report(y, t, groups, .5)

# 因果版本：只用每个受试者最初的 60 个窗口估计双峰性，之后的窗口才施加收缩
meta_end = None
for k in (60, 120):
    val = {}
    for g in rows:
        z = s[groups == g][:k]
        z = (z-z.mean())/(z.std()+1e-12)
        val[g] = (np.mean(z**3)**2+1)/(np.mean(z**4)+1e-12)
    rk = {g: i for i, g in enumerate(sorted(val, key=val.get))}
    w = {g: .4 + .6*rk[g]/(len(rk)-1) for g in rk}
    t = s.copy()
    for g in rows:
        ids = np.flatnonzero(groups == g)[k:]
        t[ids] = .5 + w[g]*(s[ids]-.5)
    res[f'causal_shrink_bimodality_first{k}'] = report(y, t, groups, .5)

for k, v in res.items():
    if isinstance(v, dict) and 'AUC' in v:
        print(f'[{k}] AUC={v["AUC"]:.4f} bACC={v["bACC"]:.4f} subjAUC={v["subject_AUC"]:.4f}', flush=True)
(OUT/'reliability_shrink_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
