"""因果时序积分：对过去若干窗的分数做滑动平均。

严格因果：窗口 i 的分数只用 i-K+1..i 这些已经到达的窗，不用标签、不用窗口序号、
不用块序号。它利用的是"认知状态在数分钟内持续"这一事实，推理时可部署。

需要原始时间顺序，因此用 highdim_deep_features.npz（含 raw，顺序未打乱）。
"""
import json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
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
from warmup_sweep_db import warm_z, _one

NJOBS = 10


def causal_smooth(s, g, k):
    """在每个受试者流内做因果滑动平均（只用过去与当前窗）。"""
    out = np.empty_like(s)
    for sub in np.unique(g):
        idx = np.where(g == sub)[0]
        v = s[idx]
        c = np.cumsum(np.r_[0., v])
        lo = np.maximum(np.arange(len(v))-k+1, 0)
        out[idx] = (c[np.arange(len(v))+1]-c[lo])/(np.arange(len(v))+1-lo)
    return out


def report(y, s, g):
    # 分数是 rank/len，固定用 0.5 作阈值，不在测试集上挑阈值
    return {'AUC': float(roc_auc_score(y, s)),
            'bACC': float(balanced_accuracy_score(y, s >= .5))}


def main():
    tick = time.perf_counter()
    d = np.load(OUT/'highdim_deep_features.npz')
    X, raw, y, g = d['X'].astype(np.float64), d['raw'], d['y'].astype(int), d['groups']
    E = np.load(OUT/'emg_features_v2.npz')['E'].astype(np.float64)
    keep = X.std(0) > 1e-8
    X = X[:, keep]
    print(f'[data] X={X.shape} raw={raw.shape} E={E.shape}', flush=True)

    cache = OUT/'warm05_temporal.npz'
    if cache.exists():
        W = np.load(cache)['W']
    else:
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=NJOBS) as pool:
            W = np.asarray(list(pool.map(_one, ((x, 'z', .5) for x in raw), chunksize=64)), dtype=np.float32)
        np.savez_compressed(cache, W=W)
        print(f'[extract] warm0.5 {W.shape} {time.perf_counter()-t0:.0f}s', flush=True)
    W = W[:, keep] if W.shape[1] == len(keep) else W

    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, g))

    def fit(tag, Xtr_all, make, proba=True, kbest=300):
        oof = np.full(len(y), np.nan)
        for tr, te in splits:
            A, T = Xtr_all[tr].astype(np.float64), Xtr_all[te].astype(np.float64)
            A = A-A.mean(1, keepdims=True); T = T-T.mean(1, keepdims=True)
            sc = StandardScaler().fit(A); A, T = sc.transform(A), sc.transform(T)
            if kbest:
                tf = FeatureTransform(kbest, 'kbest').fit(A, y[tr])
                A, T = tf.transform(A).astype(np.float64), tf.transform(T).astype(np.float64)
            m = make().fit(A, y[tr])
            oof[te] = m.predict_proba(T)[:, 1] if proba else m.decision_function(T)
        print(f'[{tag}] AUC={roc_auc_score(y, oof):.4f} ({time.perf_counter()-tick:.0f}s)', flush=True)
        return oof

    gb = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_depth=3,
                                                l2_regularization=1., random_state=0)
    lr = lambda: LogisticRegression(C=1, max_iter=3000, class_weight='balanced', random_state=0)
    svc = lambda: make_pipeline(StandardScaler(), SVC(C=1, gamma='scale', kernel='rbf', cache_size=2048))

    oofs = {}
    oofs['base_histgb'] = fit('base_histgb', np.hstack([X, E]), gb, True, 0)
    oofs['base_svc'] = fit('base_svc', np.hstack([X, E]), svc, False, 300)
    oofs['warm_lr'] = fit('warm_lr', np.hstack([W, E]), lr, True, 300)
    oofs['warm_histgb'] = fit('warm_histgb', np.hstack([W, E]), gb, True, 0)

    R = {k: rankdata(v)/len(y) for k, v in oofs.items()}
    ens = np.mean(list(R.values()), axis=0)
    print(f'\n[ensemble 未平滑] AUC={roc_auc_score(y, ens):.4f}', flush=True)

    res = {'no_smooth': report(y, ens, g), 'by_k': {}}
    for k in (1, 2, 3, 5, 8, 12, 20, 30):
        s = causal_smooth(ens, g, k)
        r = report(y, s, g)
        r['seconds'] = k*6
        res['by_k'][str(k)] = r
        print(f'[K={k:2d} ({k*6:3d}s)] AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f}', flush=True)

    # 逐折稳定性
    best_k = max(res['by_k'], key=lambda k: res['by_k'][k]['AUC'])
    res['best_k'] = best_k
    sb = causal_smooth(ens, g, int(best_k))
    res['per_fold'] = [float(roc_auc_score(y[te], sb[te])) for _, te in splits]
    res['per_fold_raw'] = [float(roc_auc_score(y[te], ens[te])) for _, te in splits]
    print(f'\n[best K={best_k}] 分折={np.round(res["per_fold"], 3).tolist()}')
    print(f'[对照未平滑] 分折={np.round(res["per_fold_raw"], 3).tolist()}', flush=True)

    np.savez_compressed(OUT/'temporal_integration_oof.npz', y=y, groups=g, ens=ens, **oofs)
    res['_meta'] = {'seconds': time.perf_counter()-tick, 'n_windows': int(len(y))}
    (OUT/'temporal_integration_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[done]', flush=True)


if __name__ == '__main__':
    main()
