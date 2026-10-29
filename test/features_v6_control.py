"""v6 的严格对照：同一 876 维基线、同一划分、同一模型，只差是否加入 v6。

features_v6.py 里的 base 是 F+E（876 维），不含 v3 的 16 维，
因此不能直接与之前的 base892 结果比较。这里补齐基线那一侧。
"""
import json, sys, time
from pathlib import Path
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT/'test'
sys.path[:0] = [str(OUT)]
from highdim_deep_ablation import FeatureTransform
from features_v6 import load, report

base, raw, y, g = load()
splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(base, y, g))
tick = time.perf_counter()
print(f'[control] base={base.shape}', flush=True)

MODELS = {
    'lr': (lambda: LogisticRegression(C=1, max_iter=3000, class_weight='balanced', random_state=0), True, 300),
    'lda': (lambda: LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'), False, 300),
    'histgb': (lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_depth=3,
                                                      l2_regularization=1., random_state=0), True, 0),
    'svc': (lambda: make_pipeline(StandardScaler(), SVC(C=1, gamma='scale', kernel='rbf', cache_size=2048)), False, 300),
}

res, oofs = {}, {}
for tag, (make, proba, kbest) in MODELS.items():
    oof = np.full(len(y), np.nan)
    for fold, (tr, te) in enumerate(splits, 1):
        A, T = base[tr], base[te]
        if kbest:
            tf = FeatureTransform(kbest, 'kbest').fit(A, y[tr])
            A, T = tf.transform(A).astype(np.float64), tf.transform(T).astype(np.float64)
        m = make().fit(A, y[tr])
        oof[te] = m.predict_proba(T)[:, 1] if proba else m.decision_function(T)
        print(f'[base_{tag}] fold={fold}/5 elapsed={time.perf_counter()-tick:.0f}s', flush=True)
    res[f'base_{tag}'] = report(y, oof, g, proba)
    oofs[f'base_{tag}'] = oof
    print(f'[result] base_{tag} AUC={res[f"base_{tag}"]["AUC"]:.4f}', flush=True)

np.savez_compressed(OUT/'features_v6_control_oof.npz', y=y, groups=g, **oofs)
res['_meta'] = {'seconds': time.perf_counter()-tick, 'n_features': int(base.shape[1])}
(OUT/'features_v6_control_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
print('[done]', flush=True)
