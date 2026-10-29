"""B 支校准方式的交叉测试：统计量类型 × 校准窗长。

已试：均值/方差（K=15/30/60/120）、稳健中位数/MAD（K=30）。
本轮补：
  - 稳健统计 K=60 / 120；
  - 分位数校准：把每个特征映射为"该受试者本人校准分布中的百分位"，
    再转成正态分位。对非线性尺度与离群值都不敏感。

全部只用每名受试者开头 K 窗，不用标签、不用块序号。
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT/'test'
sys.path[:0] = [str(OUT)]
from highdim_deep_ablation import FeatureTransform


def subject_z(X, g, K):
    Z = np.empty_like(X, dtype=np.float64)
    for s in np.unique(g):
        idx = np.where(g == s)[0]; cal = idx[:K]
        mu, sd = X[cal].mean(0), X[cal].std(0)+1e-9
        Z[idx] = (X[idx]-mu)/sd
    return Z


def subject_robust(X, g, K):
    Z = np.empty_like(X, dtype=np.float64)
    for s in np.unique(g):
        idx = np.where(g == s)[0]; cal = idx[:K]
        mu = np.median(X[cal], axis=0)
        sd = np.median(np.abs(X[cal]-mu), axis=0)*1.4826+1e-9
        Z[idx] = (X[idx]-mu)/sd
    return Z


def subject_quantile(X, g, K):
    """把每个值映射为它在本人校准窗分布中的百分位，再转为正态分位。"""
    Z = np.empty_like(X, dtype=np.float64)
    lo, hi = .5/K, 1-.5/K
    for s in np.unique(g):
        idx = np.where(g == s)[0]
        cal = np.sort(X[idx[:K]], axis=0)
        v = X[idx]
        for j in range(X.shape[1]):
            p = np.searchsorted(cal[:, j], v[:, j], side='left')/K
            Z[idx, j] = norm.ppf(np.clip(p, lo, hi))
    return Z


def report(y, s, g):
    per = [roc_auc_score(y[g == k], s[g == k]) for k in np.unique(g) if len(set(y[g == k])) == 2]
    return {'AUC': float(roc_auc_score(y, s)),
            'bACC': float(balanced_accuracy_score(y, s >= .5)),
            'subject_AUC': float(np.mean(per))}


def main():
    tick = time.perf_counter()
    d = np.load(OUT/'highdim_deep_features.npz')
    X0, y, g = d['X'].astype(np.float64), d['y'].astype(int), d['groups']
    E = np.load(OUT/'emg_features_v2.npz')['E'].astype(np.float64)
    keep = X0.std(0) > 1e-8
    base = np.hstack([X0[:, keep], E])
    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(base, y, g))
    print(f'[data] {base.shape}', flush=True)

    def run(Z, tag, make, kbest, proba=True):
        oof = np.full(len(y), np.nan)
        for tr, te in splits:
            A, T = Z[tr], Z[te]
            if kbest:
                tf = FeatureTransform(kbest, 'kbest').fit(A, y[tr])
                A, T = tf.transform(A).astype(np.float64), tf.transform(T).astype(np.float64)
            m = make().fit(A, y[tr])
            oof[te] = m.predict_proba(T)[:, 1] if proba else m.decision_function(T)
        r = report(y, oof, g)
        print(f'[result] {tag:30s} AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f} '
              f'subjAUC={r["subject_AUC"]:.4f} ({time.perf_counter()-tick:.0f}s)', flush=True)
        return r, oof

    gb = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_depth=3,
                                                l2_regularization=1., random_state=0)
    lr = lambda: LogisticRegression(C=1, max_iter=3000, class_weight='balanced', random_state=0)

    res, oofs = {}, {}
    for name, fn, K in (('robust60', subject_robust, 60), ('robust120', subject_robust, 120),
                        ('quant60', subject_quantile, 60), ('quant120', subject_quantile, 120),
                        ('quant30', subject_quantile, 30)):
        Z = fn(base, g, K)
        res[f'{name}_histgb'], oofs[f'{name}_histgb'] = run(Z, f'{name} · HistGB', gb, 0)
        res[f'{name}_lr'], oofs[f'{name}_lr'] = run(Z, f'{name} · Logistic', lr, 300)

    np.savez_compressed(OUT/'b_calib_variants_oof.npz', y=y, groups=g, **oofs)
    res['_meta'] = {'seconds': time.perf_counter()-tick}
    (OUT/'b_calib_variants_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[done]', flush=True)


if __name__ == '__main__':
    main()
