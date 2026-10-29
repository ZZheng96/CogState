"""把 0.6635 方案的 warm1s 迁到 database 口径，并扫描暖启动长度。

warm1s：只用 6 秒窗首 1 秒的均值/方差整窗归一化，再算 858 维特征。
统计量全部位于被判样本之前，推理可部署。

这里参数化暖启动长度，并加一个稳健统计（中位数/MAD）变体：
  - 预热越短，统计量越不稳；
  - 预热越长，越接近整窗 z-score，会把 task/relax 的整体幅值差一并除掉。
因此预期存在最优长度，这是此前未测的变量。
"""
import json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]; DB = ROOT/'database'; OUT = ROOT/'test'
sys.path[:0] = [str(OUT)]
from highdim_deep_ablation import FeatureTransform

FS = 250.
NJOBS = 10


def warm_z(x, seconds):
    k = max(int(seconds*FS), 10)
    mu, sd = x[:k].mean(), max(x[:k].std(), 1e-4)
    return (x-mu)/sd


def warm_robust(x, seconds):
    k = max(int(seconds*FS), 10)
    med = np.median(x[:k])
    mad = max(np.median(np.abs(x[:k]-med))*1.4826, 1e-4)
    return (x-med)/mad


MODES = {
    'warm0.5': ('z', .5),
    'warm1': ('z', 1.),
    'warm2': ('z', 2.),
    'warm3': ('z', 3.),
    'robust1': ('r', 1.),
}


def _one(args):
    from highdim_deep_ablation import highdim_features
    x, kind, sec = args
    y = warm_z(x, sec) if kind == 'z' else warm_robust(x, sec)
    return highdim_features(y).astype(np.float32).tolist()


def load_raw():
    xs, ys, gs, fs, es = [], [], [], [], []
    for p in sorted((DB/'windows').glob('*/*.npz')):
        d = np.load(p)
        xs.append(d['x']); fs.append(d['f']); es.append(d['e'])
        ys.append(d['label']); gs.append(d['subject'])
    return (np.vstack(xs).astype(np.float64), np.vstack(fs).astype(np.float64),
            np.vstack(es).astype(np.float64), np.concatenate(ys).astype(int), np.concatenate(gs))


def report(y, s, g, proba=True):
    r = {'AUC': float(roc_auc_score(y, s))}
    r['bACC'] = float(balanced_accuracy_score(y, s >= .5)) if proba else None
    per = [roc_auc_score(y[g == k], s[g == k]) for k in np.unique(g) if len(set(y[g == k])) == 2]
    r['subject_AUC'] = float(np.mean(per))
    return r


def main():
    tick = time.perf_counter()
    raw, F, E, y, g = load_raw()
    keep = F.std(0) > 1e-8
    F = F[:, keep]
    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(F, y, g))
    print(f'[data] raw={raw.shape} base={F.shape} E={E.shape}', flush=True)

    MODELS = {
        'lr': (lambda: LogisticRegression(C=1, max_iter=3000, class_weight='balanced', random_state=0), True, 300),
        'histgb': (lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_depth=3,
                                                          l2_regularization=1., random_state=0), True, 0),
        'svc': (lambda: make_pipeline(StandardScaler(), SVC(C=1, gamma='scale', kernel='rbf', cache_size=2048)), False, 300),
        'lda': (lambda: LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'), False, 300),
    }

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
        print(f'[result] {tag:22s} AUC={r["AUC"]:.4f} bACC={r["bACC"]} subjAUC={r["subject_AUC"]:.4f} '
              f'({time.perf_counter()-tick:.0f}s)', flush=True)
        return r, oof

    res, oofs = {}, {}
    res['base_lr'], oofs['base_lr'] = run(np.hstack([F, E]), 'base · Logistic300', *MODELS['lr'])
    res['base_histgb'], oofs['base_histgb'] = run(np.hstack([F, E]), 'base · HistGB 全维', *MODELS['histgb'])

    for mode, (kind, sec) in MODES.items():
        cache = OUT/f'warmfeat_{mode}.npz'
        if cache.exists():
            Xw = np.load(cache)['X']
        else:
            t0 = time.perf_counter()
            with ProcessPoolExecutor(max_workers=NJOBS) as pool:
                Xw = np.asarray(list(pool.map(_one, ((x, kind, sec) for x in raw), chunksize=64)), dtype=np.float32)
            np.savez_compressed(cache, X=Xw)
            print(f'[extract] {mode} {Xw.shape} {time.perf_counter()-t0:.0f}s', flush=True)
        Xw = Xw[:, keep] if Xw.shape[1] == len(keep) else Xw
        X = np.hstack([Xw, E])
        res[f'{mode}_lr'], oofs[f'{mode}_lr'] = run(X, f'{mode} · Logistic300', *MODELS['lr'])
        res[f'{mode}_histgb'], oofs[f'{mode}_histgb'] = run(X, f'{mode} · HistGB 全维', *MODELS['histgb'])

    # 对最好的两种长度补 SVC / LDA
    for mode in ('warm1', 'warm2'):
        Xw = np.load(OUT/f'warmfeat_{mode}.npz')['X']
        Xw = Xw[:, keep] if Xw.shape[1] == len(keep) else Xw
        X = np.hstack([Xw, E])
        res[f'{mode}_svc'], oofs[f'{mode}_svc'] = run(X, f'{mode} · RBF SVC', *MODELS['svc'])
        res[f'{mode}_lda'], oofs[f'{mode}_lda'] = run(X, f'{mode} · 收缩 LDA', *MODELS['lda'])

    np.savez_compressed(OUT/'warmup_sweep_oof.npz', y=y, groups=g, **oofs)
    res['_meta'] = {'seconds': time.perf_counter()-tick, 'n_windows': int(len(y))}
    (OUT/'warmup_sweep_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[done] {time.perf_counter()-tick:.0f}s', flush=True)


if __name__ == '__main__':
    main()
