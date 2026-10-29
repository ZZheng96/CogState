"""按特征族分别选择，避免小家族被大池子挤掉。

统一 kbest 会从 878 维里挑 300 维，20 个肌电特征若 F 值排名靠后就会整体丢失。
这里对比：统一选择 / 肌电全保留 + EEG 各选 300 / 统一选择 600 维。
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
import torch

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT/'test'
sys.path[:0] = [str(ROOT/'src'), str(OUT)]
from causal_history import rebuild_meta, report, running_stats
from highdim_deep_ablation import FeatureTransform, fit_network, predict

N_EEG = 858


class Mixed:
    """EEG 与肌电两个特征族分别标准化、分别选维。"""

    def __init__(self, k_eeg=300, k_emg=None):
        self.k_eeg, self.k_emg = k_eeg, k_emg

    def fit(self, X, y):
        self.t_eeg = FeatureTransform(self.k_eeg, 'kbest').fit(X[:, :N_EEG], y)
        E = X[:, N_EEG:]
        idx = np.arange(E.shape[1]) if self.k_emg is None else SelectKBest(f_classif, k=min(self.k_emg, E.shape[1])).fit(E, y).get_support(indices=True)
        self.i_emg = idx
        self.s_emg = StandardScaler().fit(E[:, idx])
        return self

    def transform(self, X):
        return np.hstack([self.t_eeg.transform(X[:, :N_EEG]), self.s_emg.transform(X[:, N_EEG:][:, self.i_emg])]).astype(np.float32)


def evaluate(X, y, groups, order, make, tag, device, deep=True):
    oof = {'logistic': np.full(len(y), np.nan), 'mlp': np.full(len(y), np.nan)}
    for fold, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, groups), 1):
        t0 = time.perf_counter()
        mu0, var0 = X[tr].mean(0), X[tr].var(0)+1e-20
        Xn = np.empty_like(X, dtype=float)
        for ids in order:
            m, s = running_stats(X[ids].astype(float), 5, mu0, var0); Xn[ids] = (X[ids]-m)/s
        a, b = next(StratifiedGroupKFold(4, shuffle=True, random_state=fold).split(Xn[tr], y[tr], groups[tr]))
        inner, val = tr[a], tr[b]
        tf = make().fit(Xn[inner], y[inner])
        A, V, T = tf.transform(Xn[inner]), tf.transform(Xn[val]), tf.transform(Xn[te])
        lr = LogisticRegression(C=1, max_iter=2000, class_weight='balanced', random_state=0).fit(A, y[inner])
        oof['logistic'][te] = lr.predict_proba(T)[:, 1]
        if deep:
            net, _ = fit_network('large', A, y[inner], V, y[val], device, fold)
            oof['mlp'][te] = predict(net, T, device)
        print(f'[{tag} fold {fold}/5] log={roc_auc_score(y[te], oof["logistic"][te]):.4f} '
              f'mlp={roc_auc_score(y[te], oof["mlp"][te]):.4f} {time.perf_counter()-t0:.0f}s', flush=True)
    return oof


def main():
    tick = time.perf_counter()
    F = np.load(OUT/'highdim_deep_features.npz')
    E = np.load(OUT/'emg_features_v2.npz')['E']
    X, y, groups = np.hstack([F['X'], E]), F['y'], F['groups']
    meta = rebuild_meta()
    end = np.array([m[2] for m in meta])
    subjects = np.unique(groups)
    order = [np.where(groups == s)[0][np.argsort(end[groups == s])] for s in subjects]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 诊断：统一 kbest300 里有多少个是肌电特征
    tf = FeatureTransform(300, 'kbest').fit(X, y)
    n_emg = int((tf.indices >= N_EEG).sum())
    print(f'[diagnostic] kbest300 中肌电特征数 = {n_emg}/20', flush=True)

    variants = {'kbest300_unified': lambda: FeatureTransform(300, 'kbest'),
                'eeg300_emg_all': lambda: Mixed(300, None),
                'kbest600_unified': lambda: FeatureTransform(600, 'kbest')}
    res, oofs = {}, {}
    for name, make in variants.items():
        o = evaluate(X, y, groups, order, make, name, device)
        oofs[name] = o
        for mod in o:
            res[f'{name}_{mod}'] = report(y, o[mod], groups, .5)
            r = res[f'{name}_{mod}']
            print(f'[{name}_{mod}] AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f} subjAUC={r["subject_AUC"]:.4f}', flush=True)

    np.savez_compressed(OUT/'mixed_family_oof.npz', y=y, groups=groups,
                        **{f'{k}_{m}': v for k, o in oofs.items() for m, v in o.items()})

    A0 = np.load(OUT/'online_deep_oof.npz'); B0 = np.load(OUT/'highdim_deep_oof.npz')
    cand = {f'{k}_{m}': v for k, o in oofs.items() for m, v in o.items()}
    cand.update({'online_mlp300': A0['online_mlp300'], 'online_logistic300': A0['online_logistic300'],
                 'large_mlp300': B0['large_mlp300']})
    R = {k: rankdata(v)/len(y) for k, v in cand.items()}
    import itertools
    combos, subs_list = {}, []
    for r in range(2, 5):
        for sub in itertools.combinations(cand, r):
            combos['+'.join(sub)] = report(y, np.mean([R[k] for k in sub], axis=0), groups, .5)
            subs_list.append(sub)
    best = max(combos, key=lambda k: combos[k]['AUC'])
    print(f'[best] {best} AUC={combos[best]["AUC"]:.4f} bACC={combos[best]["bACC"]:.4f}', flush=True)

    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(R['large_mlp300'], y, groups))
    nested = []
    for f, (tr, te) in enumerate(folds, 1):
        b, ba = None, -1
        for sub in subs_list:
            a = roc_auc_score(y[tr], np.mean([R[k] for k in sub], axis=0)[tr])
            if a > ba:
                ba, b = a, sub
        s = np.mean([R[k] for k in b], axis=0)
        nested.append({'fold': f, 'combo': '+'.join(b), 'AUC': float(roc_auc_score(y[te], s[te]))})
        print(f'[nested {f}/5] {"+".join(b)} AUC={nested[-1]["AUC"]:.4f}', flush=True)
    res['combos'] = combos; res['best_combo'] = best; res['nested'] = nested
    res['nested_mean_AUC'] = float(np.mean([d['AUC'] for d in nested]))
    res['_meta'] = {'seconds': time.perf_counter()-tick, 'emg_selected_in_kbest300': n_emg}
    print(f'[nested mean] AUC={res["nested_mean_AUC"]:.4f}', flush=True)
    (OUT/'mixed_family_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[done] {time.perf_counter()-tick:.0f}s', flush=True)


if __name__ == '__main__':
    main()
