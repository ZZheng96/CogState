"""A 支补充模型：完整 RBF SVC 与 ExtraTrees 随机子空间。
只使用去时序数据库中的当前窗特征，按被试隔离评估。
"""
import json, sys, time
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]; DB = ROOT/'database'; OUT = ROOT/'test'
sys.path[:0] = [str(OUT)]
from highdim_deep_ablation import FeatureTransform


def load():
    fs, es, ys, gs = [], [], [], []
    for p in sorted((DB/'windows').glob('*/*.npz')):
        d = np.load(p); fs.append(d['f']); es.append(d['e']); ys.append(d['label']); gs.append(d['subject'])
    F = np.vstack(fs).astype(np.float64); E = np.vstack(es).astype(np.float64)
    V = np.load(OUT/'features_v3.npz')['V']
    F = F[:, F.std(0) > 1e-8]
    return np.hstack([F, E, V]), np.concatenate(ys).astype(int), np.concatenate(gs)


def report(y, s, g, zero=0.):
    r = {'AUC': float(roc_auc_score(y, s)), 'bACC': float(balanced_accuracy_score(y, s >= zero))}
    per = [roc_auc_score(y[g == k], s[g == k]) for k in np.unique(g) if len(set(y[g == k])) == 2]
    r['subject_AUC'] = float(np.mean(per))
    return r


def main():
    tick = time.perf_counter(); X, y, g = load()
    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, g))
    print(f'[data] {X.shape}', flush=True)
    specs = {
        'svc_rbf_c1': lambda: make_pipeline(StandardScaler(), SVC(C=1, gamma='scale', kernel='rbf', cache_size=2048)),
        'svc_rbf_c3': lambda: make_pipeline(StandardScaler(), SVC(C=3, gamma='scale', kernel='rbf', cache_size=2048)),
        'extra_trees': lambda: ExtraTreesClassifier(n_estimators=400, max_features='sqrt', min_samples_leaf=5,
                                                     class_weight='balanced', n_jobs=-1, random_state=0),
        'extra_trees_half': lambda: ExtraTreesClassifier(n_estimators=400, max_features=.5, min_samples_leaf=8,
                                                          class_weight='balanced', n_jobs=-1, random_state=0),
    }
    res, oofs = {}, {}
    for name, make in specs.items():
        oof = np.full(len(y), np.nan)
        for fold, (tr, te) in enumerate(splits, 1):
            tf = FeatureTransform(300, 'kbest').fit(X[tr], y[tr])
            A, T = tf.transform(X[tr]).astype(np.float64), tf.transform(X[te]).astype(np.float64)
            m = make().fit(A, y[tr])
            if name.startswith('svc'):
                s = m.decision_function(T); zero = 0.
            else:
                s = m.predict_proba(T)[:, 1]; zero = .5
            oof[te] = s
            print(f'[{name}] fold={fold}/5 elapsed={time.perf_counter()-tick:.0f}s', flush=True)
        res[name] = report(y, oof, g, 0. if name.startswith('svc') else .5)
        oofs[name] = oof
        print(f'[{name}] AUC={res[name]["AUC"]:.4f} bACC={res[name]["bACC"]:.4f} subjAUC={res[name]["subject_AUC"]:.4f}', flush=True)
    res['_meta'] = {'seconds': time.perf_counter()-tick, 'n_features': int(X.shape[1])}
    np.savez_compressed(OUT/'kernel_trees_oof.npz', y=y, groups=g, **oofs)
    (OUT/'kernel_trees_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[done] elapsed={time.perf_counter()-tick:.0f}s', flush=True)

if __name__ == '__main__': main()
