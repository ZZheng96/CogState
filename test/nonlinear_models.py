"""非线性树模型：在含肌电的完整特征集上补一个不同的归纳偏置。

此前只用过 Logistic 与 2 层 MLP，集成已饱和。梯度提升与极端随机树能表达特征间的
非线性组合（如频带比、肌电-EEG 交互），且对量纲不敏感，可为集成提供多样性。
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT/'test'
sys.path[:0] = [str(ROOT/'src'), str(OUT)]
from causal_history import rebuild_meta, report, running_stats
from mixed_family import Mixed


def main():
    tick = time.perf_counter()
    F = np.load(OUT/'highdim_deep_features.npz')
    E = np.load(OUT/'emg_features_v2.npz')['E']
    X, y, groups = np.hstack([F['X'], E]), F['y'], F['groups']
    meta = rebuild_meta()
    end = np.array([m[2] for m in meta])
    subjects = np.unique(groups)
    order = [np.where(groups == s)[0][np.argsort(end[groups == s])] for s in subjects]

    oof = {'histgb': np.full(len(y), np.nan), 'extratrees': np.full(len(y), np.nan)}
    for fold, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, groups), 1):
        t0 = time.perf_counter()
        mu0, var0 = X[tr].mean(0), X[tr].var(0)+1e-20
        Xn = np.empty_like(X, dtype=float)
        for ids in order:
            m, s = running_stats(X[ids].astype(float), 5, mu0, var0); Xn[ids] = (X[ids]-m)/s
        tf = Mixed(300, None).fit(Xn[tr], y[tr])
        A, T = tf.transform(Xn[tr]), tf.transform(Xn[te])
        gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_depth=3,
                                            l2_regularization=1., random_state=0)
        gb.fit(A, y[tr]); oof['histgb'][te] = gb.predict_proba(T)[:, 1]
        print(f'[fold {fold} histgb] AUC={roc_auc_score(y[te], oof["histgb"][te]):.4f} {time.perf_counter()-t0:.0f}s', flush=True)
        et = ExtraTreesClassifier(n_estimators=500, max_features='sqrt', min_samples_leaf=20,
                                  class_weight='balanced', n_jobs=-1, random_state=0).fit(A, y[tr])
        oof['extratrees'][te] = et.predict_proba(T)[:, 1]
        print(f'[fold {fold} extratrees] AUC={roc_auc_score(y[te], oof["extratrees"][te]):.4f} {time.perf_counter()-t0:.0f}s', flush=True)

    res = {}
    for k, v in oof.items():
        res[k] = report(y, v, groups, .5)
        print(f'[{k}] AUC={res[k]["AUC"]:.4f} bACC={res[k]["bACC"]:.4f} subjAUC={res[k]["subject_AUC"]:.4f}', flush=True)
    np.savez_compressed(OUT/'nonlinear_oof.npz', y=y, groups=groups, **oof)

    A0 = np.load(OUT/'online_deep_oof.npz'); B0 = np.load(OUT/'highdim_deep_oof.npz')
    M0 = np.load(OUT/'mixed_family_oof.npz'); E0 = np.load(OUT/'emg_v2_oof.npz')
    cand = {**oof,
            'eeg300_emg_all_mlp': M0['eeg300_emg_all_mlp'], 'eeg300_emg_all_logistic': M0['eeg300_emg_all_logistic'],
            'emg_v2_mlp': E0['emg_v2_mlp'], 'online_mlp300': A0['online_mlp300'], 'large_mlp300': B0['large_mlp300']}
    R = {k: rankdata(v)/len(y) for k, v in cand.items()}
    import itertools
    combos, subs = {}, []
    for r in range(2, 5):
        for sub in itertools.combinations(cand, r):
            combos['+'.join(sub)] = report(y, np.mean([R[k] for k in sub], axis=0), groups, .5)
            subs.append(sub)
    best = max(combos, key=lambda k: combos[k]['AUC'])
    print(f'[best] {best} AUC={combos[best]["AUC"]:.4f} bACC={combos[best]["bACC"]:.4f}', flush=True)

    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(R['large_mlp300'], y, groups))
    nested = []
    for f, (tr, te) in enumerate(folds, 1):
        b, ba = None, -1
        for sub in subs:
            a = roc_auc_score(y[tr], np.mean([R[k] for k in sub], axis=0)[tr])
            if a > ba:
                ba, b = a, sub
        s = np.mean([R[k] for k in b], axis=0)
        nested.append({'fold': f, 'combo': '+'.join(b), 'AUC': float(roc_auc_score(y[te], s[te]))})
        print(f'[nested {f}/5] {"+".join(b)} AUC={nested[-1]["AUC"]:.4f}', flush=True)
    res['combos'] = combos; res['best_combo'] = best; res['nested'] = nested
    res['nested_mean_AUC'] = float(np.mean([d['AUC'] for d in nested]))
    res['_meta'] = {'seconds': time.perf_counter()-tick}
    print(f'[nested mean] AUC={res["nested_mean_AUC"]:.4f}', flush=True)
    (OUT/'nonlinear_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[done] {time.perf_counter()-tick:.0f}s', flush=True)


if __name__ == '__main__':
    main()
