"""B 支（纯 EEG + 开场预热校准）的进一步提升。

B 支做法：用每名受试者开头 K 个窗的均值/方差，标准化该受试者的全部窗。
只用本人数据、不用标签、不用块序号与交替模式，推理前 3 分钟即可获得。

本脚本：
  1. 复现 K=30 + HistGB 的 0.683 锚点；
  2. 扫描 K；
  3. 叠加 0.5 s 窗内暖启动特征、SVC、rank 集成与 24 s 因果时序积分。
"""
import json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT/'test'
sys.path[:0] = [str(OUT)]
from highdim_deep_ablation import FeatureTransform
from warmup_sweep_db import _one
from temporal_integration import causal_smooth


def subject_z(X, g, K, mode='z'):
    """用每个受试者开头 K 个窗的统计量标准化其全部窗。"""
    Z = np.empty_like(X, dtype=np.float64)
    for s in np.unique(g):
        idx = np.where(g == s)[0]
        cal = idx[:K]
        if mode == 'z':
            mu, sd = X[cal].mean(0), X[cal].std(0)+1e-9
        else:
            mu = np.median(X[cal], axis=0)
            sd = np.median(np.abs(X[cal]-mu), axis=0)*1.4826+1e-9
        Z[idx] = (X[idx]-mu)/sd
    return Z


def report(y, s, g, proba=True):
    r = {'AUC': float(roc_auc_score(y, s))}
    r['bACC'] = float(balanced_accuracy_score(y, s >= .5)) if proba else None
    per = [roc_auc_score(y[g == k], s[g == k]) for k in np.unique(g) if len(set(y[g == k])) == 2]
    r['subject_AUC'] = float(np.mean(per))
    return r


def main():
    tick = time.perf_counter()
    d = np.load(OUT/'highdim_deep_features.npz')
    X0, raw, y, g = d['X'].astype(np.float64), d['raw'], d['y'].astype(int), d['groups']
    E = np.load(OUT/'emg_features_v2.npz')['E'].astype(np.float64)
    keep = X0.std(0) > 1e-8
    X0 = X0[:, keep]
    base = np.hstack([X0, E])

    cache = OUT/'b_warm05.npz'
    if cache.exists():
        W = np.load(cache)['W']
    else:
        with ProcessPoolExecutor(max_workers=10) as pool:
            W = np.asarray(list(pool.map(_one, ((x, 'z', .5) for x in raw), chunksize=64)), dtype=np.float32)
        np.savez_compressed(cache, W=W)
        print(f'[extract] warm0.5 {W.shape} ({time.perf_counter()-tick:.0f}s)', flush=True)
    W = W[:, keep] if W.shape[1] == len(keep) else W
    warm = np.hstack([W, E])
    print(f'[data] base={base.shape} warm={warm.shape}', flush=True)

    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(base, y, g))

    def run(Z, tag, make, proba=True, kbest=300):
        oof = np.full(len(y), np.nan)
        for tr, te in splits:
            A, T = Z[tr], Z[te]
            if kbest:
                tf = FeatureTransform(kbest, 'kbest').fit(A, y[tr])
                A, T = tf.transform(A).astype(np.float64), tf.transform(T).astype(np.float64)
            m = make().fit(A, y[tr])
            oof[te] = m.predict_proba(T)[:, 1] if proba else m.decision_function(T)
        r = report(y, oof, g, proba)
        print(f'[result] {tag:26s} AUC={r["AUC"]:.4f} bACC={r["bACC"]} ({time.perf_counter()-tick:.0f}s)', flush=True)
        return r, oof

    gb = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_depth=3,
                                                l2_regularization=1., random_state=0)
    lr = lambda: LogisticRegression(C=1, max_iter=3000, class_weight='balanced', random_state=0)
    svc = lambda: make_pipeline(StandardScaler(), SVC(C=1, gamma='scale', kernel='rbf', cache_size=2048))
    lda = lambda: LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')

    res, oofs = {}, {}
    # 1) 锚点 + 校准窗长扫描
    for K in (15, 30, 60, 120):
        Z = subject_z(base, g, K)
        res[f'K{K}_histgb'], oofs[f'K{K}_histgb'] = run(Z, f'B base K={K} · HistGB', gb, True, 0)
        res[f'K{K}_lr'], oofs[f'K{K}_lr'] = run(Z, f'B base K={K} · Logistic', lr)
    # 稳健统计
    Zr = subject_z(base, g, 30, mode='robust')
    res['K30_robust_histgb'], oofs['K30_robust_histgb'] = run(Zr, 'B base K=30 robust · HistGB', gb, True, 0)

    # 2) 最佳 K 上叠加 warm0.5 特征
    K = int(max((15, 30, 60, 120), key=lambda k: res[f'K{k}_histgb']['AUC']))
    res['best_K'] = K
    print(f'[best K] {K}', flush=True)
    Zb = subject_z(base, g, K); Zw = subject_z(warm, g, K)
    res['warm_histgb'], oofs['warm_histgb'] = run(Zw, f'B warm0.5 K={K} · HistGB', gb, True, 0)
    res['warm_lr'], oofs['warm_lr'] = run(Zw, f'B warm0.5 K={K} · Logistic', lr)
    res['warm_svc'], oofs['warm_svc'] = run(Zw, f'B warm0.5 K={K} · SVC', svc, False)
    res['base_svc'], oofs['base_svc'] = run(Zb, f'B base K={K} · SVC', svc, False)
    res['warm_lda'], oofs['warm_lda'] = run(Zw, f'B warm0.5 K={K} · LDA', lda, False)

    # 3) 集成 + 时序积分
    keys = ('K%d_histgb' % K, 'warm_lr', 'warm_histgb', 'base_svc')
    R = {k: rankdata(oofs[k])/len(y) for k in keys}
    ens = np.mean([R[k] for k in keys], axis=0)
    res['ensemble'], oofs['ensemble'] = report(y, ens, g, True), ens
    print(f'[result] ensemble(4) AUC={res["ensemble"]["AUC"]:.4f} bACC={res["ensemble"]["bACC"]:.4f}', flush=True)
    for k in (1, 2, 3, 4, 5, 8):
        s = causal_smooth(ens, g, k)
        r = report(y, s, g, True)
        res[f'ens_smooth{k}'] = r
        print(f'[result] ensemble + 时序积分 K={k} ({k*6}s) AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f}', flush=True)

    np.savez_compressed(OUT/'b_branch_oof.npz', y=y, groups=g, **oofs)
    res['_meta'] = {'seconds': time.perf_counter()-tick, 'n_features': int(base.shape[1])}
    (OUT/'b_branch_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[done]', flush=True)


if __name__ == '__main__':
    main()
