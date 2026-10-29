"""因果协方差对齐（在线 CORAL / 欧氏对齐）。

对角 z-score 只校正每维的均值/方差；受试者之间还存在特征相关结构的差异。
这里在低维（PCA）子空间里用"过去窗口"的累计协方差做白化-再着色，并收缩到训练集协方差，
比较它与纯对角标准化的差异。所有统计量只用当前及过去窗口。
"""
import json, sys, time
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT/'test'
sys.path[:0] = [str(ROOT/'src'), str(OUT)]
from causal_history import rebuild_meta, report, running_stats
from highdim_deep_ablation import FeatureTransform

VARIANTS = {'online_z': (None, None), 'pca40_lam0.5': (40, .5),
            'pca40_lam0.9': (40, .9), 'pca40_lam0.99': (40, .99)}


def sqrtm(cov, power=.5):
    w, V = np.linalg.eigh(cov)
    return (V*np.maximum(w, 1e-10)**power)@V.T


def causal_align(Z, mean, cov, L0, n0=5, lam=.9, step=10):
    """Z: 单个受试者按时间排序的窗口（n×d），返回同顺序的对齐结果。"""
    n, d = Z.shape
    out = np.empty_like(Z)
    m, M = mean, L0
    for k in range(n):
        if k % step == 0:
            past = Z[:max(k, 1)]
            m = (n0*mean + past.sum(0))/(n0+len(past))
            dev = past - m
            c = (n0*cov + dev.T@dev)/(n0+len(past))
            M = sqrtm((1-lam)*c + lam*cov, -.5)@L0
        out[k] = (Z[k]-m)@M
    return out


def main():
    tick = time.perf_counter()
    F = np.load(OUT/'highdim_deep_features.npz')
    X, y, groups = F['X'], F['y'], F['groups']
    meta = rebuild_meta()
    end = np.array([m[2] for m in meta])
    subjects = np.unique(groups)
    order = {s: np.where(groups == s)[0][np.argsort(end[groups == s])] for s in subjects}
    oof = {k: np.full(len(y), np.nan) for k in VARIANTS}

    for fold, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, groups), 1):
        t0 = time.perf_counter()
        mu0, var0 = X[tr].mean(0), X[tr].var(0)+1e-20
        Xn = np.empty_like(X, dtype=float)
        for ids in order.values():
            m, s = running_stats(X[ids].astype(float), 5, mu0, var0); Xn[ids] = (X[ids]-m)/s
        tf = FeatureTransform(300, 'kbest').fit(Xn[tr], y[tr])
        Z = tf.transform(Xn)
        pos_tr = {g: i for i, g in enumerate(tr)}
        pos_te = {g: i for i, g in enumerate(te)}

        for name, (dim, lam) in VARIANTS.items():
            if dim is None:
                A, T = Z[tr], Z[te]
            else:
                pca = PCA(dim, random_state=0).fit(Z[tr])
                P = pca.transform(Z)
                A, T = np.empty((len(tr), dim)), np.empty((len(te), dim))
                mean, cov = P[tr].mean(0), np.cov(P[tr], rowvar=False)
                L0 = sqrtm(cov, .5)
                for s in subjects:
                    ids = order[s]
                    a, b = ids[np.isin(ids, tr)], ids[np.isin(ids, te)]
                    if len(a):
                        A[[pos_tr[i] for i in a]] = causal_align(P[a], mean, cov, L0, lam=lam)
                    if len(b):
                        T[[pos_te[i] for i in b]] = causal_align(P[b], mean, cov, L0, lam=lam)
            clf = LogisticRegression(C=1, max_iter=2000, class_weight='balanced', random_state=0).fit(A, y[tr])
            oof[name][te] = clf.predict_proba(T)[:, 1]
            print(f'[fold {fold} {name}] AUC={roc_auc_score(y[te], oof[name][te]):.4f} '
                  f'elapsed={time.perf_counter()-t0:.0f}s', flush=True)

    res = {k: report(y, v, groups, .5) for k, v in oof.items()}
    for k, v in res.items():
        print(f'[{k}] AUC={v["AUC"]:.4f} bACC={v["bACC"]:.4f} subjAUC={v["subject_AUC"]:.4f}', flush=True)
    res['meta'] = {'seconds': time.perf_counter()-tick}
    (OUT/'coral_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[done] {time.perf_counter()-tick:.0f}s', flush=True)


if __name__ == '__main__':
    main()
