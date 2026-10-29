"""无标签的受试者内双状态聚类 TTA。

假设每个受试者的窗口来自两个潜在状态。用校准后的 EEG+EMG 特征做两簇 GMM，
簇方向只由过去的全局 P3 分数决定，不使用真标签、块序号或交替协议。
同时报告转导上界与严格因果版。
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from b_calib_variants import subject_quantile
from singlem_eval import smooth, boot

K, REFIT = 120, 30

def fit_score(Xfit, Xpred, sfit, dim, seed):
    # 只在个人历史上标准化和降维，避免测试窗口参与变换统计量
    sc = StandardScaler().fit(Xfit)
    A, B = sc.transform(Xfit), sc.transform(Xpred)
    pca = PCA(n_components=min(dim, A.shape[1], max(2, A.shape[0]-1)), random_state=seed).fit(A)
    A, B = pca.transform(A), pca.transform(B)
    gm = GaussianMixture(n_components=2, covariance_type='diag', reg_covar=1e-3,
                         random_state=seed, n_init=3, max_iter=300).fit(A)
    pa = gm.predict_proba(A); pb = gm.predict_proba(B)
    # 簇方向由过去的全局分数决定；若全局分数没有方向，返回中性
    means = np.array([sfit[gm.predict(A) == c].mean() if np.any(gm.predict(A) == c) else .5 for c in range(2)])
    return pb[:, int(np.argmax(means))], float(abs(means[0]-means[1]))

def personal(Z, sg, causal, dim):
    n = len(sg); out = np.full(n, .5); sep = np.zeros(n)
    if not causal:
        out[:], sep0 = fit_score(Z, Z, sg, dim, 0); sep[:] = sep0
        return out, sep
    for start in range(K, n, REFIT):
        end = min(start+REFIT, n)
        p, sep0 = fit_score(Z[:start], Z[start:end], sg[:start], dim, 0)
        # 转成历史个人后验的因果百分位，保持与全局分数同一尺度
        hist, _ = fit_score(Z[:start], Z[:start], sg[:start], dim, 0)
        order = np.sort(hist)
        out[start:end] = (np.searchsorted(order, p)+.5)/(len(order)+1)
        sep[start:end] = sep0
    return out, sep

def main():
    tick = time.perf_counter()
    d = np.load(OUT/'highdim_deep_features.npz')
    X0, y, g = d['X'].astype(float), d['y'].astype(int), d['groups']
    E = np.load(OUT/'emg_features_v2.npz')['E'].astype(float)
    X0 = X0[:, X0.std(0) > 1e-8]
    Z = subject_quantile(np.hstack([X0, E]), g, K)
    pos = np.empty(len(y), int)
    for s in np.unique(g):
        i = np.where(g == s)[0]; pos[i] = np.arange(len(i))
    mask = pos >= K
    pool = {}
    for f in ('b_branch_oof.npz','b_calib_variants_oof.npz','b_calib2_oof.npz'):
        q = np.load(OUT/f); pool.update({k:q[k] for k in q.files if k not in ('y','groups')})
    P3 = ['quant120_histgb','quant120w_histgb','quant120_svc','base_svc']
    glob = np.mean([rankdata(pool[k])/len(y) for k in P3], 0)
    gs = smooth(glob, g, 5)
    res = {'base': {'AUC': float(roc_auc_score(y[mask], gs[mask])),
                    'bACC': float(balanced_accuracy_score(y[mask], gs[mask] >= .5))}}
    for causal in (True, False):
        for dim in (10, 20, 40):
            P, sep = np.full(len(y), .5), np.zeros(len(y))
            for s in np.unique(g):
                idx = np.where(g == s)[0]
                P[idx], sep[idx] = personal(Z[idx], gs[idx], causal, dim)
            Ps = smooth(P, g, 5)
            best = None
            for a in (.25, .5, .75, 1.0):
                score = gs + a*(Ps-.5)
                r = {'AUC': float(roc_auc_score(y[mask], score[mask])),
                     'bACC': float(balanced_accuracy_score(y[mask], score[mask] >= .5)),
                     'vs_base': boot(score, gs, y, g, mask),
                     'median_sep': float(np.median(sep[mask]))}
                if best is None or r['AUC'] > best['AUC']: best = dict(r, alpha=a)
            tag = f'{"causal" if causal else "transd"}_pca{dim}'
            res[tag] = best
            print(f'[result] {tag:14s} AUC={best["AUC"]:.4f} bACC={best["bACC"]:.4f} '
                  f'alpha={best["alpha"]} Δ={best["vs_base"]["mean"]:+.4f} '
                  f'[{best["vs_base"]["p2.5"]:+.4f},{best["vs_base"]["p97.5"]:+.4f}] '
                  f'sep={best["median_sep"]:.4f} ({time.perf_counter()-tick:.0f}s)', flush=True)
    (OUT/'unsup_state_tta_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')

if __name__ == '__main__': main()
