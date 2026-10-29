"""评估现有特征对 relax/task 的区分能力，并检验是否存在"清晰可分的特征子集"。

三个层次：
1. 单特征：pooled AUC、逐受试者 AUC、方向一致性、Cohen's d、分布重叠系数；
2. 子集：按一致性排序取 top-k，用受试者分组 CV 看小维度子集能到多少；
3. 可分性：多维标准化空间里的轮廓系数与类别直方图重叠。

关键区分：pooled AUC 会把"受试者间差异"算成区分度，因此同时报告受试者内 AUC。
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]; DB = ROOT/'database'; OUT = ROOT/'test'
sys.path[:0] = [str(OUT)]


def load():
    fs, es, ys, gs = [], [], [], []
    for p in sorted((DB/'windows').glob('*/*.npz')):
        d = np.load(p); fs.append(d['f']); es.append(d['e'])
        ys.append(d['label']); gs.append(d['subject'])
    F = np.vstack(fs).astype(np.float64); F = F[:, F.std(0) > 1e-8]
    E = np.vstack(es).astype(np.float64)
    V3 = np.load(OUT/'features_v3.npz')['V'].astype(np.float64)
    V6 = np.load(OUT/'features_v6.npz')['V'].astype(np.float64)
    names = ([f'base_{i}' for i in range(F.shape[1])] + [f'emg_{i}' for i in range(E.shape[1])] +
             [f'v3_{i}' for i in range(V3.shape[1])] + [f'v6_{i}' for i in range(V6.shape[1])])
    X = np.hstack([F, E, V3, np.nan_to_num(V6)])
    return X, np.concatenate(ys).astype(int), np.concatenate(gs), np.array(names)


def vec_auc(pos_mask, X):
    """向量化 Mann-Whitney AUC：X 的每一列作为一个打分。"""
    n1 = pos_mask.sum(); n0 = len(pos_mask)-n1
    r = rankdata(X, axis=0)
    s = r[pos_mask].sum(0)
    return (s-n1*(n1+1)/2)/(n1*n0)


def overlap(a, b, bins=40):
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    if hi <= lo:
        return 1.
    e = np.linspace(lo, hi, bins+1)
    p, _ = np.histogram(a, e, density=True)
    q, _ = np.histogram(b, e, density=True)
    w = e[1]-e[0]
    return float(np.sum(np.minimum(p, q))*w)


def main():
    X, y, g, names = load()
    n, d = X.shape
    print(f'[data] {X.shape} features, {len(np.unique(g))} subjects', flush=True)

    pooled = vec_auc(y == 1, X)
    pooled = np.where(pooled >= .5, pooled, 1-pooled)      # 方向统一为"值越大越 task"

    subs = np.unique(g)
    within = np.full((len(subs), d), np.nan)
    for i, s in enumerate(subs):
        m = g == s
        if len(set(y[m])) < 2:
            continue
        a = vec_auc(y[m] == 1, X[m])
        within[i] = np.where(a >= .5, a, 1-a)
    mean_within = np.nanmean(within, axis=0)
    median_within = np.nanmedian(within, axis=0)

    # 方向一致性：受试者内 task-relax 差值的符号
    signs = np.zeros((len(subs), d))
    for i, s in enumerate(subs):
        m = g == s
        if len(set(y[m])) < 2:
            continue
        signs[i] = np.sign(X[m][y[m] == 1].mean(0)-X[m][y[m] == 0].mean(0))
    agree = np.maximum((signs > 0).mean(0), (signs < 0).mean(0))

    print('\n=== 单特征 top 15（按受试者内 AUC 排序）===')
    print(f'{"特征":12s} {"pooled":>7s} {"受试者内":>8s} {"中位":>6s} {"一致性":>7s}')
    order = np.argsort(-median_within)[:15]
    for j in order:
        print(f'{names[j]:12s} {pooled[j]:7.3f} {mean_within[j]:8.3f} {median_within[j]:6.3f} {agree[j]:7.2f}')

    print('\n=== 单特征 top 10（按 pooled AUC 排序）===')
    print(f'{"特征":12s} {"pooled":>7s} {"受试者内":>8s} {"一致性":>7s}')
    for j in np.argsort(-pooled)[:10]:
        print(f'{names[j]:12s} {pooled[j]:7.3f} {mean_within[j]:8.3f} {agree[j]:7.2f}')

    best = int(np.argmax(median_within))
    ovl = overlap(X[y == 1, best], X[y == 0, best])
    print(f'\n[最佳单特征] {names[best]} pooled={pooled[best]:.3f} 受试者内={mean_within[best]:.3f} '
          f'重叠系数={ovl:.3f}', flush=True)

    # 子集：按一致性加权质量排序，取 top-k 用分组 CV 评估
    score = median_within*agree
    rank_idx = np.argsort(-score)
    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, g))
    sub_res = {}
    for k in (5, 10, 20, 50, 100):
        idx = rank_idx[:k]
        oof = np.full(n, np.nan)
        for tr, te in splits:
            A, T = X[tr][:, idx], X[te][:, idx]
            sc = StandardScaler().fit(A)
            m = LogisticRegression(C=1, max_iter=3000, class_weight='balanced', random_state=0)
            oof[te] = m.fit(sc.transform(A), y[tr]).predict_proba(sc.transform(T))[:, 1]
        sub_res[k] = float(roc_auc_score(y, oof))
        print(f'[子集 k={k:3d}] AUC={sub_res[k]:.4f}', flush=True)

    # 多维空间可分性
    top20 = rank_idx[:20]
    Z = StandardScaler().fit_transform(X[:, top20])
    sil = float(silhouette_score(Z, y, sample_size=4000, random_state=0))
    print(f'\n[轮廓系数 top20 空间] {sil:.4f}（>0.25 才算有清晰簇）', flush=True)

    res = {'n_features': int(d), 'best_single': {'name': str(names[best]), 'pooled': float(pooled[best]),
                                                 'within': float(mean_within[best]), 'overlap': ovl},
           'top_within': [{'name': str(names[j]), 'pooled': float(pooled[j]), 'within': float(mean_within[j]),
                           'median_within': float(median_within[j]), 'agree': float(agree[j])} for j in order],
           'top_pooled': [{'name': str(names[j]), 'pooled': float(pooled[j]), 'within': float(mean_within[j]),
                           'agree': float(agree[j])} for j in np.argsort(-pooled)[:10]],
           'subset_auc': sub_res, 'silhouette_top20': sil,
           'counts': {'pooled>0.65': int((pooled > .65).sum()), 'pooled>0.70': int((pooled > .70).sum()),
                      'within>0.60': int((mean_within > .60).sum()), 'within>0.65': int((mean_within > .65).sum()),
                      'agree>0.85 且 within>0.58': int(((agree > .85) & (mean_within > .58)).sum())}}
    print(f'\n[计数] {res["counts"]}', flush=True)
    np.savez_compressed(OUT/'separability_scores.npz', pooled=pooled, mean_within=mean_within,
                        median_within=median_within, agree=agree, names=names, y=y, groups=g)
    (OUT/'separability_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[done]', flush=True)


if __name__ == '__main__':
    main()
