"""继续定位：HistGB 的高分来自原始特征、还是来自在线标准化？"""
import sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT/'test'
sys.path[:0] = [str(ROOT/'src'), str(OUT)]
from causal_history import rebuild_meta, running_stats
from mixed_family import Mixed

F = np.load(OUT/'highdim_deep_features.npz')
E = np.load(OUT/'emg_features_v2.npz')['E']
X, y, groups = np.hstack([F['X'], E]), F['y'], F['groups']
meta = rebuild_meta()
end = np.array([m[2] for m in meta])
subjects = np.unique(groups)
order = [np.where(groups == s)[0][np.argsort(end[groups == s])] for s in subjects]
splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, groups))


def cv(make, data, tag):
    oof = np.full(len(y), np.nan)
    for tr, te in splits:
        m = make().fit(data[tr], y[tr])
        oof[te] = m.predict_proba(data[te])[:, 1]
    print(f'[{tag}] AUC={roc_auc_score(y, oof):.4f}', flush=True)
    return oof


def gb():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_depth=3,
                                          l2_regularization=1., random_state=0)


def normalized():
    Xn = np.empty_like(X, dtype=float)
    for tr, _ in splits:  # 占位，实际在下面按折算
        pass
    return Xn


# 1) 原始特征（不做事先归一化），树模型
cv(gb, X, 'HistGB 原始特征')
# 2) 原始特征（全局标准化后），线性模型
cv(lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=3000, class_weight='balanced')), X, 'Logistic 原始特征')
# 3) 只取肌电 20 维
cv(gb, X[:, 858:], 'HistGB 仅肌电')
# 4) 在线归一化后（与正式流程一致）
Xn = np.empty_like(X, dtype=float)
for tr, te in splits:
    mu0, var0 = X[tr].mean(0), X[tr].var(0)+1e-20
    for ids in order:
        m, s = running_stats(X[ids].astype(float), 5, mu0, var0); Xn[ids] = (X[ids]-m)/s
    break  # 归一化与折无关（每受试者用自己的历史），算一次即可
cv(gb, Xn, 'HistGB 在线归一化特征')

# 5) 重要度：在某一折上训练，用留出受试者算 permutation importance
tr, te = splits[0]
m = gb().fit(Xn[tr], y[tr])
imp = permutation_importance(m, Xn[te], y[te], n_repeats=3, random_state=0, scoring='roc_auc', n_jobs=-1)
order_i = np.argsort(-imp.importances_mean)[:12]
import json as _j
names = _j.loads((OUT/'highdim_deep_results.json').read_text(encoding='utf-8'))['feature_names'] + \
        list(np.load(OUT/'emg_features_v2.npz')['names'])
print('[importance] top12:', flush=True)
for i in order_i:
    print(f'   {str(names[i]) if i < len(names) else i:34s} {imp.importances_mean[i]:.4f} ± {imp.importances_std[i]:.4f}', flush=True)

# 6) 只用 top-1 特征
i = order_i[0]
cv(lambda: gb(), X[:, [i]], f'HistGB 仅单特征 {i}')
