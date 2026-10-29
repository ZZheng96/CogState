"""ref 评估的后续：把"窗内因果暖启动归一化"（warm1s）接入现有模型族与集成。

warm1s = 只取 6 秒窗首 1 秒的均值/方差，对该窗做整窗归一化。
统计量全部来自当前窗、且位于被判样本之前，推理时可用，属于窗内因果归一化。

用法：python test/ref_warmup_followup.py
"""
from __future__ import annotations
import itertools, json, sys, time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'test'
sys.path[:0] = [str(ROOT / 'src'), str(OUT)]
RESULT = OUT / 'ref_warmup_followup_results.json'


def report(y, s, groups):
    r = {'AUC': float(roc_auc_score(y, s)), 'bACC': float(balanced_accuracy_score(y, s >= .5))}
    per = [(roc_auc_score(y[groups == g], s[groups == g]), balanced_accuracy_score(y[groups == g], s[groups == g] >= .5))
           for g in np.unique(groups) if len(set(y[groups == g])) == 2]
    r['subject_AUC'] = float(np.mean([p[0] for p in per]))
    r['subject_bACC'] = float(np.mean([p[1] for p in per]))
    return r


def fit_predict(model, Xtr, ytr, Xte, k=300):
    A, T = Xtr.astype(np.float64), Xte.astype(np.float64)
    A = A - A.mean(1, keepdims=True)
    T = T - T.mean(1, keepdims=True)
    sc = StandardScaler().fit(A)
    A, T = sc.transform(A), sc.transform(T)
    if k and k < A.shape[1]:
        sel = SelectKBest(f_classif, k=k).fit(A, ytr)
        A, T = sel.transform(A), sel.transform(T)
    if model == 'logistic':
        clf = LogisticRegression(C=1, max_iter=2000, class_weight='balanced', random_state=0)
        return clf.fit(A, ytr).predict_proba(T)[:, 1]
    if model == 'lda':
        clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(A, ytr)
        s = clf.decision_function(T)
        return 1 / (1 + np.exp(-s))
    if model == 'histgb':
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_leaf_nodes=15,
                                             min_samples_leaf=40, l2_regularization=1., random_state=0)
        return clf.fit(A, ytr).predict_proba(T)[:, 1]
    raise ValueError(model)


def main():
    tick = time.perf_counter()
    F = np.load(OUT / 'highdim_deep_features.npz')
    y, groups = F['y'], F['groups']
    Xb = F['X'].astype(np.float32)
    Xw = np.load(OUT / 'ref_feat_warm1s.npz')['X'].astype(np.float32)
    E = np.load(OUT / 'emg_features_v2.npz')['E'].astype(np.float32)
    keep = Xb.std(0) > 1e-8
    if Xw.shape[1] == len(keep):
        Xw = Xw[:, keep]
    sets = {'base': np.hstack([Xb[:, keep], E]), 'warm1s': np.hstack([Xw, E])}
    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Xb, y, groups))

    oof, res = {}, {}
    for name, X in sets.items():
        for model, k in (('logistic', 300), ('lda', 300), ('histgb', 0)):
            tag = f'{name}_{model}'
            s = np.full(len(y), np.nan)
            for tr, te in splits:
                s[te] = fit_predict(model, X[tr], y[tr], X[te], k)
            oof[tag] = s
            res[tag] = report(y, s, groups)
            print(f'[{tag}] AUC={res[tag]["AUC"]:.4f} bACC={res[tag]["bACC"]:.4f} '
                  f'子均AUC={res[tag]["subject_AUC"]:.4f} ({time.perf_counter()-tick:.0f}s)', flush=True)

    R = {k: rankdata(v) / len(y) for k, v in oof.items()}
    combos = {}
    for r in range(2, 5):
        for sub in itertools.combinations(R, r):
            s = np.mean([R[k] for k in sub], axis=0)
            combos['+'.join(sub)] = report(y, s, groups)
    best = max(combos, key=lambda k: combos[k]['AUC'])
    print(f'[best] {best} AUC={combos[best]["AUC"]:.4f} bACC={combos[best]["bACC"]:.4f}', flush=True)

    # 嵌套选择：每折只用训练折的 OOF 排名挑组合，评估留出折
    nested = []
    for f, (tr, te) in enumerate(splits, 1):
        cand, best_a, best_s = None, -1, None
        for key in combos:
            s = np.mean([R[k] for k in key.split('+')], axis=0)
            a = roc_auc_score(y[tr], s[tr])
            if a > best_a:
                best_a, cand, best_s = a, key, s
        nested.append({'fold': f, 'combo': cand,
                       'AUC': float(roc_auc_score(y[te], best_s[te])),
                       'bACC': float(balanced_accuracy_score(y[te], best_s[te] >= np.median(best_s[tr])))})
        print(f'[nested {f}/5] {nested[-1]["combo"]} AUC={nested[-1]["AUC"]:.4f}', flush=True)
    res['combos'] = combos
    res['best_combo'] = best
    res['nested'] = nested
    res['nested_mean_AUC'] = float(np.mean([d['AUC'] for d in nested]))
    res['_meta'] = {'seconds': time.perf_counter() - tick}
    RESULT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    np.savez_compressed(OUT / 'ref_warmup_followup_oof.npz', y=y, groups=groups, **oof)
    print(f'[done] {time.perf_counter()-tick:.0f}s -> {RESULT}', flush=True)


if __name__ == '__main__':
    main()
