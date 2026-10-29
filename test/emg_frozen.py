"""在冻结校准（无位置信息）条件下，复核肌电特征的增益是否仍然成立。"""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT/'test'
sys.path[:0] = [str(ROOT/'src'), str(OUT)]
from causal_history import rebuild_meta, report
from highdim_deep_ablation import FeatureTransform

F = np.load(OUT/'highdim_deep_features.npz')
E = np.load(OUT/'emg_features_v2.npz')['E']
meta = rebuild_meta(); end = np.array([m[2] for m in meta])
y, groups = F['y'], F['groups']
subjects = np.unique(groups)
ids_time = [np.where(groups == s)[0][np.argsort(end[groups == s])] for s in subjects]


def norm(X0, tr, N=30):
    mu0, var0 = X0[tr].mean(0), X0[tr].var(0)+1e-20
    out = np.empty_like(X0, dtype=float)
    for ids in ids_time:
        v = X0[ids[:N]].astype(float)
        m = (5*mu0 + v.sum(0))/(5+len(v))
        ex2 = (5*(var0+mu0**2) + (v**2).sum(0))/(5+len(v))
        out[ids] = (X0[ids]-m)/np.sqrt(np.maximum(ex2-m**2, 1e-12))
    return out


res = {}
for tag, X in (('EEG 858', F['X']), ('EEG 858 + 肌电 20', np.hstack([F['X'], E])),
               ('仅肌电 20', E)):
    X = X[:, X.std(0) > 1e-8]
    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, groups))
    for k, mod in ((300, 'logistic'), (300, 'histgb'), (0, 'histgb')):
        oof = np.full(len(y), np.nan)
        for tr, te in splits:
            Xn = norm(X, tr)
            if k:
                tf = FeatureTransform(k, 'kbest').fit(Xn[tr], y[tr])
                A, T = tf.transform(Xn[tr]), tf.transform(Xn[te])
            else:
                A, T = Xn[tr], Xn[te]
            m = (LogisticRegression(C=1, max_iter=2000, class_weight='balanced', random_state=0)
                 if mod == 'logistic' else
                 HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_depth=3,
                                                l2_regularization=1., random_state=0))
            oof[te] = m.fit(A, y[tr]).predict_proba(T)[:, 1]
        r = report(y, oof, groups, .5)
        res[f'{tag}|k{k or "all"}|{mod}'] = r
        print(f'[{tag:18s} k={k or "all":4} {mod:8s}] AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f} subjAUC={r["subject_AUC"]:.4f}', flush=True)
(OUT/'emg_frozen_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
