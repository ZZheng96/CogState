"""对任意已缓存的窗口长度特征做统一评估（在线标准化 + Logistic / HistGB）。"""
import json, sys, time
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT/'test'
sys.path[:0] = [str(ROOT/'src'), str(OUT)]
from causal_history import report, running_stats
from highdim_deep_ablation import FeatureTransform


def evaluate(path):
    d = np.load(path)
    X, y, groups, end = d['X'], d['y'], d['groups'], d['end']
    keep = np.flatnonzero(X.std(0) > 1e-8)
    X = X[:, keep]
    order = [np.where(groups == s)[0][np.argsort(end[groups == s])] for s in np.unique(groups)]
    oof = {'logistic': np.full(len(y), np.nan), 'histgb': np.full(len(y), np.nan)}
    tick = time.perf_counter()
    for fold, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, groups), 1):
        mu0, var0 = X[tr].mean(0), X[tr].var(0)+1e-20
        Xn = np.empty_like(X, dtype=float)
        for ids in order:
            m, s = running_stats(X[ids].astype(float), 5, mu0, var0); Xn[ids] = (X[ids]-m)/s
        tf = FeatureTransform(300, 'kbest').fit(Xn[tr], y[tr])
        A, T = tf.transform(Xn[tr]), tf.transform(Xn[te])
        lr = LogisticRegression(C=1, max_iter=2000, class_weight='balanced', random_state=0).fit(A, y[tr])
        oof['logistic'][te] = lr.predict_proba(T)[:, 1]
        gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_depth=3,
                                            l2_regularization=1., random_state=0).fit(A, y[tr])
        oof['histgb'][te] = gb.predict_proba(T)[:, 1]
        print(f'[{path.stem} fold {fold}/5] log={roc_auc_score(y[te], oof["logistic"][te]):.4f} '
              f'gb={roc_auc_score(y[te], oof["histgb"][te]):.4f}', flush=True)
    res = {'n_windows': int(len(y)), 'n_features': int(X.shape[1])}
    for k, v in oof.items():
        res[k] = report(y, v, groups, .5)
        r = res[k]
        print(f'[{path.stem} {k}] AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f} subjAUC={r["subject_AUC"]:.4f}', flush=True)
    res['seconds'] = time.perf_counter()-tick
    return res


if __name__ == '__main__':
    out = {}
    for p in sorted(OUT.glob('window*_features.npz')):
        out[p.stem] = evaluate(p)
    (OUT/'window_eval_results.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
