"""B 支：分位数校准的扩展测试。

  - K=180：校准集更长，看是否继续变好（代价是预热期更长、可评估窗更少）；
  - 暖启动特征 + 分位数校准；
  - 分位数校准 + RBF SVC。

所有结果都同时报告"全窗"与"剔除全部校准窗"两个口径。
"""
import json, sys, time
from pathlib import Path
import numpy as np
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
from b_calib_variants import subject_quantile, subject_z


def main():
    tick = time.perf_counter()
    d = np.load(OUT/'highdim_deep_features.npz')
    X0, y, g = d['X'].astype(np.float64), d['y'].astype(int), d['groups']
    E = np.load(OUT/'emg_features_v2.npz')['E'].astype(np.float64)
    keep = X0.std(0) > 1e-8
    base = np.hstack([X0[:, keep], E])
    W = np.load(OUT/'b_warm05.npz')['W']
    W = W[:, keep] if W.shape[1] == len(keep) else W
    warm = np.hstack([W, E])
    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(base, y, g))

    pos = np.empty(len(y), dtype=int)
    for s in np.unique(g):
        i = np.where(g == s)[0]; pos[i] = np.arange(len(i))

    def run(Z, tag, make, kbest=300, proba=True, K=120):
        oof = np.full(len(y), np.nan)
        for tr, te in splits:
            A, T = Z[tr], Z[te]
            if kbest:
                tf = FeatureTransform(kbest, 'kbest').fit(A, y[tr])
                A, T = tf.transform(A).astype(np.float64), tf.transform(T).astype(np.float64)
            m = make().fit(A, y[tr])
            oof[te] = m.predict_proba(T)[:, 1] if proba else m.decision_function(T)
        m = pos >= K
        r = {'full_AUC': float(roc_auc_score(y, oof)), 'full_bACC': float(balanced_accuracy_score(y, oof >= .5)),
             'excl_AUC': float(roc_auc_score(y[m], oof[m])),
             'excl_bACC': float(balanced_accuracy_score(y[m], oof[m] >= .5)),
             'inflation': float(roc_auc_score(y, oof)-roc_auc_score(y[m], oof[m])), 'n': int(m.sum())}
        print(f'[result] {tag:32s} 全窗 {r["full_AUC"]:.4f} | 剔校准 {r["excl_AUC"]:.4f} '
              f'膨胀 {r["inflation"]:+.4f} ({time.perf_counter()-tick:.0f}s)', flush=True)
        return r, oof

    gb = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_depth=3,
                                                l2_regularization=1., random_state=0)
    lr = lambda: LogisticRegression(C=1, max_iter=3000, class_weight='balanced', random_state=0)
    svc = lambda: make_pipeline(StandardScaler(), SVC(C=1, gamma='scale', kernel='rbf', cache_size=2048))

    res, oofs = {}, {}
    Z180 = subject_quantile(base, g, 180)
    print(f'[quant180] 就绪 ({time.perf_counter()-tick:.0f}s)', flush=True)
    res['quant180_histgb'], oofs['quant180_histgb'] = run(Z180, 'quant180 · HistGB', gb, 0, True, 180)
    res['quant180_lr'], oofs['quant180_lr'] = run(Z180, 'quant180 · Logistic', lr, 300, True, 180)

    Zw = subject_quantile(warm, g, 120)
    print(f'[quant120 warm] 就绪 ({time.perf_counter()-tick:.0f}s)', flush=True)
    res['quant120w_histgb'], oofs['quant120w_histgb'] = run(Zw, 'quant120 warm · HistGB', gb, 0, True, 120)
    res['quant120w_lr'], oofs['quant120w_lr'] = run(Zw, 'quant120 warm · Logistic', lr, 300, True, 120)

    Z120 = subject_quantile(base, g, 120)
    res['quant120_svc'], oofs['quant120_svc'] = run(Z120, 'quant120 · SVC', svc, 300, False, 120)

    np.savez_compressed(OUT/'b_calib2_oof.npz', y=y, groups=g, **oofs)
    res['_meta'] = {'seconds': time.perf_counter()-tick}
    (OUT/'b_calib2_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[done]', flush=True)


if __name__ == '__main__':
    main()
