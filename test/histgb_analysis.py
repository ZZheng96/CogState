"""HistGB 重要度与维度需求：树模型是否需要 300 维筛选，以及哪些特征族在起作用。"""
import json, sys, time
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT/'test'
sys.path[:0] = [str(ROOT/'src'), str(OUT)]
from causal_history import rebuild_meta, report, running_stats
from highdim_deep_ablation import FeatureTransform


def main():
    tick = time.perf_counter()
    F = np.load(OUT/'highdim_deep_features.npz')
    base = json.loads((OUT/'highdim_deep_results.json').read_text(encoding='utf-8'))['feature_names']
    E = np.load(OUT/'emg_features_v2.npz')['E']
    names = base + list(np.load(OUT/'emg_features_v2.npz')['names'])
    X0 = np.hstack([F['X'], E])
    keep = np.flatnonzero(X0.std(0) > 1e-8)
    X, y, groups = X0[:, keep], F['y'], F['groups']
    kept_names = [names[i] for i in keep]
    print(f'[data] {X.shape}', flush=True)

    meta = rebuild_meta()
    end = np.array([m[2] for m in meta])
    order = [np.where(groups == s)[0][np.argsort(end[groups == s])] for s in np.unique(groups)]
    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, groups))

    def gb():
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_depth=3,
                                              l2_regularization=1., random_state=0)

    res = {}
    for k in (300, 600, 0):
        oof = np.full(len(y), np.nan)
        for tr, te in splits:
            mu0, var0 = X[tr].mean(0), X[tr].var(0)+1e-20
            Xn = np.empty_like(X, dtype=float)
            for ids in order:
                m, s = running_stats(X[ids].astype(float), 5, mu0, var0); Xn[ids] = (X[ids]-m)/s
            if k:
                tf = FeatureTransform(k, 'kbest').fit(Xn[tr], y[tr])
                A, T = tf.transform(Xn[tr]), tf.transform(Xn[te])
            else:
                A, T = Xn[tr], Xn[te]
            oof[te] = gb().fit(A, y[tr]).predict_proba(T)[:, 1]
        res[f'k={k or "all"}'] = report(y, oof, groups, .5)
        r = res[f'k={k or "all"}']
        print(f'[HistGB k={k or "all"}] AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f} subjAUC={r["subject_AUC"]:.4f}', flush=True)

    # 重要度（第 1 折，300 维）
    tr, te = splits[0]
    mu0, var0 = X[tr].mean(0), X[tr].var(0)+1e-20
    Xn = np.empty_like(X, dtype=float)
    for ids in order:
        m, s = running_stats(X[ids].astype(float), 5, mu0, var0); Xn[ids] = (X[ids]-m)/s
    tf = FeatureTransform(300, 'kbest').fit(Xn[tr], y[tr])
    A, T = tf.transform(Xn[tr]), tf.transform(Xn[te])
    model = gb().fit(A, y[tr])
    imp = permutation_importance(model, T, y[te], n_repeats=3, random_state=0, scoring='roc_auc', n_jobs=-1)
    idx = np.argsort(-imp.importances_mean)[:20]
    sel = [kept_names[i] for i in tf.indices]
    print('[importance top20]', flush=True)
    for i in idx:
        print(f'   {sel[i]:38s} {imp.importances_mean[i]:+.4f} ± {imp.importances_std[i]:.4f}', flush=True)
    res['importance'] = [{'name': sel[i], 'mean': float(imp.importances_mean[i]),
                          'std': float(imp.importances_std[i])} for i in idx]
    res['_meta'] = {'seconds': time.perf_counter()-tick}
    (OUT/'histgb_analysis_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[done] {time.perf_counter()-tick:.0f}s', flush=True)


if __name__ == '__main__':
    main()
