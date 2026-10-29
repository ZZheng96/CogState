"""定位 HistGB 0.92 的来源：分组是否被破坏、是否某一特征族在泄漏。"""
import sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
import json

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT/'test'
sys.path[:0] = [str(ROOT/'src'), str(OUT)]
from causal_history import rebuild_meta, report, running_stats
from mixed_family import Mixed

F = np.load(OUT/'highdim_deep_features.npz')
E = np.load(OUT/'emg_features_v2.npz')['E']
X, y, groups = np.hstack([F['X'], E]), F['y'], F['groups']
meta = rebuild_meta()
end = np.array([m[2] for m in meta])
subjects = np.unique(groups)
order = [np.where(groups == s)[0][np.argsort(end[groups == s])] for s in subjects]

splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, groups))
print('[split] 每折 subject 交集:', [len(set(groups[tr]) & set(groups[te])) for tr, te in splits], flush=True)
print('[split] 每折 test 窗口数:', [len(te) for tr, te in splits], flush=True)


def run(cols, tag, seed=None):
    oof = np.full(len(y), np.nan)
    for fold, (tr, te) in enumerate(splits, 1):
        mu0, var0 = X[tr].mean(0), X[tr].var(0)+1e-20
        Xn = np.empty_like(X, dtype=float)
        for ids in order:
            m, s = running_stats(X[ids].astype(float), 5, mu0, var0); Xn[ids] = (X[ids]-m)/s
        A = Xn[tr][:, cols]; T = Xn[te][:, cols]
        yy = y[tr] if seed is None else np.random.default_rng(seed).permutation(y[tr])
        gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_depth=3,
                                            l2_regularization=1., random_state=0).fit(A, yy)
        oof[te] = gb.predict_proba(T)[:, 1]
    print(f'[{tag}] AUC={roc_auc_score(y, oof):.4f}', flush=True)
    return oof


run(np.arange(X.shape[1]), '全部特征(878)')
run(np.arange(858), '仅 EEG 池(858)')
run(np.arange(858, X.shape[1]), '仅肌电(20)')
run(np.arange(X.shape[1]), '全部特征 + 打乱标签', seed=0)

# 逐折 AUC，看是否某一折异常
oof = run(np.arange(X.shape[1]), '全部特征(重复)')
for fold, (tr, te) in enumerate(splits, 1):
    print(f'  fold {fold} AUC={roc_auc_score(y[te], oof[te]):.4f} subjects={sorted(set(groups[te]))[:3]}...', flush=True)
