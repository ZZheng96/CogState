"""HistGB 超参数网格 + 按折嵌套选择，给出不带挑选偏差的估计。"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT/'test'
sys.path[:0] = [str(ROOT/'src'), str(OUT)]
from causal_history import rebuild_meta, report, running_stats
from highdim_deep_ablation import FeatureTransform

SETTINGS = {'d3_i300_lr06': dict(max_depth=3, max_iter=300, learning_rate=.06),
            'd5_i300_lr06': dict(max_depth=5, max_iter=300, learning_rate=.06),
            'd3_i600_lr03': dict(max_depth=3, max_iter=600, learning_rate=.03),
            'd5_i600_lr03': dict(max_depth=5, max_iter=600, learning_rate=.03),
            'd7_i400_lr05': dict(max_depth=7, max_iter=400, learning_rate=.05)}


def main():
    tick = time.perf_counter()
    F = np.load(OUT/'highdim_deep_features.npz')
    E = np.load(OUT/'emg_features_v2.npz')['E']
    X0 = np.hstack([F['X'], E])
    keep = np.flatnonzero(X0.std(0) > 1e-8)
    X, y, groups = X0[:, keep], F['y'], F['groups']
    meta = rebuild_meta()
    end = np.array([m[2] for m in meta])
    order = [np.where(groups == s)[0][np.argsort(end[groups == s])] for s in np.unique(groups)]
    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, groups))

    oof = {k: np.full(len(y), np.nan) for k in SETTINGS}
    for fold, (tr, te) in enumerate(splits, 1):
        mu0, var0 = X[tr].mean(0), X[tr].var(0)+1e-20
        Xn = np.empty_like(X, dtype=float)
        for ids in order:
            m, s = running_stats(X[ids].astype(float), 5, mu0, var0); Xn[ids] = (X[ids]-m)/s
        tf = FeatureTransform(300, 'kbest').fit(Xn[tr], y[tr])
        A, T = tf.transform(Xn[tr]), tf.transform(Xn[te])
        for name, kw in SETTINGS.items():
            gb = HistGradientBoostingClassifier(l2_regularization=1., random_state=0, **kw).fit(A, y[tr])
            oof[name][te] = gb.predict_proba(T)[:, 1]
            print(f'[fold {fold} {name}] AUC={roc_auc_score(y[te], oof[name][te]):.4f}', flush=True)

    res = {}
    for name, v in oof.items():
        res[name] = report(y, v, groups, .5)
        r = res[name]
        print(f'[{name}] AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f} subjAUC={r["subject_AUC"]:.4f}', flush=True)
    np.savez_compressed(OUT/'histgb_grid_oof.npz', y=y, groups=groups, **oof)

    nested = []
    for fold, (tr, te) in enumerate(splits, 1):
        name = max(oof, key=lambda k: roc_auc_score(y[tr], oof[k][tr]))
        nested.append({'fold': fold, 'setting': name, 'AUC': float(roc_auc_score(y[te], oof[name][te]))})
        print(f'[nested {fold}/5] {name} AUC={nested[-1]["AUC"]:.4f}', flush=True)
    res['nested'] = nested
    res['nested_mean_AUC'] = float(np.mean([d['AUC'] for d in nested]))
    print(f'[nested mean] AUC={res["nested_mean_AUC"]:.4f}', flush=True)
    res['_meta'] = {'seconds': time.perf_counter()-tick, 'n_features': int(X.shape[1])}
    (OUT/'histgb_grid_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[done] {time.perf_counter()-tick:.0f}s', flush=True)


if __name__ == '__main__':
    main()
