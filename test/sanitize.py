"""剔除退化（近常数）特征后重算全部关键结果。

nonlinear() 里 activity_normalized = var((x-mean)/std) 恒等于 1，是一个常数特征。
逐受试者在线标准化时，常数特征被 1e-6 量级的分母放大成浮点噪声，树模型会把它当成
判别特征（AUC 0.98），线性模型也会受轻微干扰。这里先剔除再重算。
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
import torch

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT/'test'
sys.path[:0] = [str(ROOT/'src'), str(OUT)]
from causal_history import rebuild_meta, report, running_stats
from highdim_deep_ablation import FeatureTransform, fit_network, predict

TOL = 1e-8


def main():
    tick = time.perf_counter()
    F = np.load(OUT/'highdim_deep_features.npz')
    base_names = json.loads((OUT/'highdim_deep_results.json').read_text(encoding='utf-8'))['feature_names']
    E = np.load(OUT/'emg_features_v2.npz')['E']
    names = base_names + list(np.load(OUT/'emg_features_v2.npz')['names'])
    X0, y, groups = np.hstack([F['X'], E]), F['y'], F['groups']

    sd = X0.std(0)
    bad = np.flatnonzero(sd < TOL)
    print(f'[sanitize] 剔除 {len(bad)} 个近常数特征: {[names[i] for i in bad]}', flush=True)
    keep = np.flatnonzero(sd >= TOL)
    X = X0[:, keep]
    n_eeg = int((keep < 858).sum())
    print(f'[sanitize] 保留 {X.shape[1]} 维（EEG {n_eeg} + 肌电 {X.shape[1]-n_eeg}）', flush=True)

    meta = rebuild_meta()
    end = np.array([m[2] for m in meta])
    subjects = np.unique(groups)
    order = [np.where(groups == s)[0][np.argsort(end[groups == s])] for s in subjects]
    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, groups))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def normalized(tr):
        mu0, var0 = X[tr].mean(0), X[tr].var(0)+1e-20
        Xn = np.empty_like(X, dtype=float)
        for ids in order:
            m, s = running_stats(X[ids].astype(float), 5, mu0, var0); Xn[ids] = (X[ids]-m)/s
        return Xn

    res, oofs = {}, {}
    for tag, cols in (('eeg', np.arange(n_eeg)), ('eeg_emg', np.arange(X.shape[1]))):
        o = {'logistic': np.full(len(y), np.nan), 'mlp': np.full(len(y), np.nan), 'histgb': np.full(len(y), np.nan)}
        for fold, (tr, te) in enumerate(splits, 1):
            t0 = time.perf_counter()
            Xn = normalized(tr)
            a, b = next(StratifiedGroupKFold(4, shuffle=True, random_state=fold).split(Xn[tr], y[tr], groups[tr]))
            inner, val = tr[a], tr[b]
            tf = FeatureTransform(300, 'kbest').fit(Xn[inner][:, cols], y[inner])
            A, V, T = tf.transform(Xn[inner][:, cols]), tf.transform(Xn[val][:, cols]), tf.transform(Xn[te][:, cols])
            lr = LogisticRegression(C=1, max_iter=2000, class_weight='balanced', random_state=0).fit(A, y[inner])
            o['logistic'][te] = lr.predict_proba(T)[:, 1]
            net, _ = fit_network('large', A, y[inner], V, y[val], device, fold)
            o['mlp'][te] = predict(net, T, device)
            gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_depth=3,
                                                l2_regularization=1., random_state=0).fit(A, y[inner])
            o['histgb'][te] = gb.predict_proba(T)[:, 1]
            print(f'[{tag} fold {fold}/5] log={roc_auc_score(y[te], o["logistic"][te]):.4f} '
                  f'mlp={roc_auc_score(y[te], o["mlp"][te]):.4f} gb={roc_auc_score(y[te], o["histgb"][te]):.4f} '
                  f'{time.perf_counter()-t0:.0f}s', flush=True)
        for k, v in o.items():
            oofs[f'{tag}_{k}'] = v
            res[f'{tag}_{k}'] = report(y, v, groups, .5)
            r = res[f'{tag}_{k}']
            print(f'[{tag}_{k}] AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f} subjAUC={r["subject_AUC"]:.4f}', flush=True)

    np.savez_compressed(OUT/'sanitized_oof.npz', y=y, groups=groups, **oofs)
    cand = dict(oofs)
    R = {k: rankdata(v)/len(y) for k, v in cand.items()}
    import itertools
    combos, subs = {}, []
    for r in range(2, 5):
        for sub in itertools.combinations(cand, r):
            combos['+'.join(sub)] = report(y, np.mean([R[k] for k in sub], axis=0), groups, .5)
            subs.append(sub)
    best = max(combos, key=lambda k: combos[k]['AUC'])
    print(f'[best] {best} AUC={combos[best]["AUC"]:.4f} bACC={combos[best]["bACC"]:.4f}', flush=True)
    nested = []
    for f, (tr, te) in enumerate(splits, 1):
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
    res['_meta'] = {'seconds': time.perf_counter()-tick, 'dropped': [names[i] for i in bad],
                    'n_features': int(X.shape[1])}
    print(f'[nested mean] AUC={res["nested_mean_AUC"]:.4f}', flush=True)
    (OUT/'sanitized_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[done] {time.perf_counter()-tick:.0f}s', flush=True)


if __name__ == '__main__':
    main()
