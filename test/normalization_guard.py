"""定位在线标准化被放大的特征：受试者内标准差远小于全局标准差时，标准化后只剩数值噪声。

同时检验"给分母加下限"能否消除伪影，以及在干净条件下树模型真实的维度收益。
"""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT/'test'
sys.path[:0] = [str(ROOT/'src'), str(OUT)]
from causal_history import rebuild_meta, report, running_stats
from highdim_deep_ablation import FeatureTransform

F = np.load(OUT/'highdim_deep_features.npz')
base = json.loads((OUT/'highdim_deep_results.json').read_text(encoding='utf-8'))['feature_names']
E = np.load(OUT/'emg_features_v2.npz')['E']
names = base + list(np.load(OUT/'emg_features_v2.npz')['names'])
X0 = np.hstack([F['X'], E]); y, groups = F['y'], F['groups']
meta = rebuild_meta(); end = np.array([m[2] for m in meta])
subjects = np.unique(groups)
ids_by_subject = [np.where(groups == s)[0][np.argsort(end[groups == s])] for s in subjects]

# 1) 每个特征的“受试者内标准差 / 全局标准差”
glob = X0.std(0) + 1e-30
within = np.array([np.median([X0[ids][:, j].std() for ids in ids_by_subject]) for j in range(X0.shape[1])])
ratio = within/glob
print('[ratio] 分位:', np.round(np.percentile(ratio, [1, 5, 25, 50, 75, 95]), 5), flush=True)
bad = np.flatnonzero(ratio < 1e-3)
print(f'[ratio] 受试者内/全局 < 1e-3 的特征数 = {len(bad)}', flush=True)
for j in bad[:15]:
    print(f'   {names[j]:34s} glob={glob[j]:.3e} within={within[j]:.3e} ratio={ratio[j]:.2e}', flush=True)

# 2) 标准化后与标签的相关（诊断用，看是否集中在这些特征上）
tr = np.concatenate([np.where(groups != s)[0] for s in subjects[:11]])
mu0, var0 = X0[tr].mean(0), X0[tr].var(0)+1e-20
Xn = np.empty_like(X0, dtype=float)
for ids in ids_by_subject:
    m, s = running_stats(X0[ids].astype(float), 5, mu0, var0); Xn[ids] = (X0[ids]-m)/s
corr = np.array([abs(np.corrcoef(Xn[:, j], y)[0, 1]) for j in range(X0.shape[1])])
top = np.argsort(-corr)[:12]
print('[corr] 归一化后 |corr(y)| 最高的 12 个：', flush=True)
for j in top:
    print(f'   {names[j]:34s} |corr|={corr[j]:.3f} ratio={ratio[j]:.2e} zmax={np.abs(Xn[:, j]).max():.1e}', flush=True)

# 3) 给分母加下限后重估（下限 = 该特征训练集标准差的 5%）
splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X0, y, groups))


def norm_floor(tr_ids, floor_frac):
    mu0, var0 = X0[tr_ids].mean(0), X0[tr_ids].var(0)+1e-20
    sd0 = X0[tr_ids].std(0) + 1e-30
    out = np.empty_like(X0, dtype=float)
    for ids in ids_by_subject:
        m, s = running_stats(X0[ids].astype(float), 5, mu0, var0)
        s = np.maximum(s, floor_frac*sd0)
        out[ids] = (X0[ids]-m)/s
    return out


for frac in (0.0, .05):
    for k in (300, 600, 0):
        oof = np.full(len(y), np.nan)
        for tr_ids, te_ids in splits:
            Xn = norm_floor(tr_ids, frac)
            if k:
                tf = FeatureTransform(k, 'kbest').fit(Xn[tr_ids], y[tr_ids])
                A, T = tf.transform(Xn[tr_ids]), tf.transform(Xn[te_ids])
            else:
                A, T = Xn[tr_ids], Xn[te_ids]
            gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_depth=3,
                                                l2_regularization=1., random_state=0).fit(A, y[tr_ids])
            oof[te_ids] = gb.predict_proba(T)[:, 1]
        r = report(y, oof, groups, .5)
        print(f'[floor={frac} k={k or "all"}] AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f} subjAUC={r["subject_AUC"]:.4f}', flush=True)
