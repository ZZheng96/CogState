"""暖启动长度的补充扫描：0.25 s 与 0.75 s。

0.5 s 优于 1/2/3 s，趋势指向更短的预热更有利，这里向两侧各补一个点定位峰值。
"""
import json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT/'test'
sys.path[:0] = [str(OUT)]
from highdim_deep_ablation import FeatureTransform
from warmup_sweep_db import load_raw, report, warm_z, _one

EXTRA = {'warm0.25': .25, 'warm0.75': .75}

MODELS = {
    'lr': (lambda: LogisticRegression(C=1, max_iter=3000, class_weight='balanced', random_state=0), True, 300),
    'histgb': (lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_depth=3,
                                                      l2_regularization=1., random_state=0), True, 0),
    'svc': (lambda: make_pipeline(StandardScaler(), SVC(C=1, gamma='scale', kernel='rbf', cache_size=2048)), False, 300),
    'lda': (lambda: LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'), False, 300),
}


def main():
    tick = time.perf_counter()
    raw, F, E, y, g = load_raw()
    keep = F.std(0) > 1e-8
    F = F[:, keep]
    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(F, y, g))

    def run(X, tag, make, proba, kbest):
        oof = np.full(len(y), np.nan)
        for tr, te in splits:
            A, T = X[tr].astype(np.float64), X[te].astype(np.float64)
            A = A-A.mean(1, keepdims=True); T = T-T.mean(1, keepdims=True)
            sc = StandardScaler().fit(A); A, T = sc.transform(A), sc.transform(T)
            if kbest:
                tf = FeatureTransform(kbest, 'kbest').fit(A, y[tr])
                A, T = tf.transform(A).astype(np.float64), tf.transform(T).astype(np.float64)
            m = make().fit(A, y[tr])
            oof[te] = m.predict_proba(T)[:, 1] if proba else m.decision_function(T)
        r = report(y, oof, g, proba)
        print(f'[result] {tag:22s} AUC={r["AUC"]:.4f} bACC={r["bACC"]} ({time.perf_counter()-tick:.0f}s)', flush=True)
        return r, oof

    res, oofs = {}, {}
    for mode, sec in EXTRA.items():
        cache = OUT/f'warmfeat_{mode}.npz'
        if cache.exists():
            Xw = np.load(cache)['X']
        else:
            t0 = time.perf_counter()
            with ProcessPoolExecutor(max_workers=10) as pool:
                Xw = np.asarray(list(pool.map(_one, ((x, 'z', sec) for x in raw), chunksize=64)), dtype=np.float32)
            np.savez_compressed(cache, X=Xw)
            print(f'[extract] {mode} {Xw.shape} {time.perf_counter()-t0:.0f}s', flush=True)
        Xw = Xw[:, keep] if Xw.shape[1] == len(keep) else Xw
        X = np.hstack([Xw, E])
        tags = ('lr', 'histgb', 'svc', 'lda') if mode == 'warm0.25' else ('lr', 'histgb')
        for tag in tags:
            key = f'{mode}_{tag}'
            res[key], oofs[key] = run(X, f'{mode} · {tag}', *MODELS[tag])

    np.savez_compressed(OUT/'warmup_more_oof.npz', y=y, groups=g, **oofs)
    res['_meta'] = {'seconds': time.perf_counter()-tick}
    (OUT/'warmup_more_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[done]', flush=True)


if __name__ == '__main__':
    main()
