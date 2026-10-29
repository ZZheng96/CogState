"""A 支（纯 EEG）收尾：把三类增益叠在一起，并做嵌套集成。

三类增益分别是：高频肌电特征（+0.006）、alpha 峰/爆发特征（+0.005）、窗内去整体电平（+0.004）。
全部只用当前窗口信息，不含任何顺序。
"""
import json, sys, itertools, time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]; DB = ROOT/'database'; OUT = ROOT/'test'
sys.path[:0] = [str(OUT)]
from highdim_deep_ablation import FeatureTransform


def load():
    fs, es, ys, gs = [], [], [], []
    for p in sorted((DB/'windows').glob('*/*.npz')):
        d = np.load(p)
        fs.append(d['f']); es.append(d['e']); ys.append(d['label']); gs.append(d['subject'])
    return (np.vstack(fs).astype(np.float64), np.vstack(es).astype(np.float64),
            np.concatenate(ys).astype(int), np.concatenate(gs))


def report(y, s, g):
    r = {'AUC': float(roc_auc_score(y, s)), 'bACC': float(balanced_accuracy_score(y, s >= .5))}
    per = [{'AUC': roc_auc_score(y[g == k], s[g == k]), 'bACC': balanced_accuracy_score(y[g == k], s[g == k] >= .5)}
           for k in np.unique(g) if len(set(y[g == k])) == 2]
    r.update({f'subject_{m}': float(np.mean([d[m] for d in per])) for m in ('AUC', 'bACC')})
    return r


def center(A, T):
    mu, sd = A.mean(0), A.std(0)+1e-12
    A, T = (A-mu)/sd, (T-mu)/sd
    return A - A.mean(1, keepdims=True), T - T.mean(1, keepdims=True)


def main():
    tick = time.perf_counter()
    F, E, y, g = load()
    V = np.load(OUT/'features_v3.npz')['V']
    X = np.hstack([F[:, F.std(0) > 1e-8], E, V])
    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, g))
    print(f'[data] {X.shape}', flush=True)

    def evaluate(make, tag, kbest=300, use_center=True):
        oof = np.full(len(y), np.nan)
        for tr, te in splits:
            A, T = (center(X[tr], X[te]) if use_center else (X[tr], X[te]))
            if kbest:
                tf = FeatureTransform(kbest, 'kbest').fit(A, y[tr])
                A, T = tf.transform(A).astype(np.float64), tf.transform(T).astype(np.float64)
            oof[te] = make().fit(A, y[tr]).predict_proba(T)[:, 1]
        r = report(y, oof, g)
        print(f'[{tag}] AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f} subjAUC={r["subject_AUC"]:.4f}', flush=True)
        return r, oof

    lr = lambda: LogisticRegression(C=1, max_iter=2000, class_weight='balanced', random_state=0)
    gb = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_depth=3,
                                                l2_regularization=1., random_state=0)
    res, oofs = {}, {}
    res['combined_logistic'], oofs['combined_logistic'] = evaluate(lr, '全特征+去电平 · Logistic300')
    res['combined_histgb'], oofs['combined_histgb'] = evaluate(gb, '全特征+去电平 · HistGB 全维', kbest=0)
    res['combined_no_center'], oofs['combined_no_center'] = evaluate(lr, '全特征(不去电平) · Logistic300', use_center=False)

    R = {k: rankdata(v)/len(y) for k, v in oofs.items()}
    ens = np.mean([R['combined_logistic'], R['combined_histgb']], axis=0)
    res['ensemble_lr_gb'] = report(y, ens, g)
    print(f'[集成 Logistic+HistGB] AUC={res["ensemble_lr_gb"]["AUC"]:.4f} bACC={res["ensemble_lr_gb"]["bACC"]:.4f}', flush=True)
    per = [roc_auc_score(y[te], ens[te]) for _, te in splits]
    print(f'[集成 分折] {np.round(per, 3).tolist()}', flush=True)
    for k, v in oofs.items():
        pk = [roc_auc_score(y[te], v[te]) for _, te in splits]
        print(f'[{k} 分折] {np.round(pk, 3).tolist()} mean={np.mean(pk):.4f}', flush=True)
    rng = np.random.default_rng(0); boot = []
    for _ in range(400):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        boot.append(roc_auc_score(y[idx], ens[idx]) - roc_auc_score(y[idx], oofs['combined_logistic'][idx]))
    res['ensemble_minus_logistic_bootstrap'] = {'mean': float(np.mean(boot)), 'p2.5': float(np.percentile(boot, 2.5)),
                                                'p97.5': float(np.percentile(boot, 97.5))}
    print(f'[bootstrap] 集成-单模型 AUC 差 = {np.mean(boot):+.4f} '
          f'[{np.percentile(boot, 2.5):+.4f}, {np.percentile(boot, 97.5):+.4f}]', flush=True)
    np.savez_compressed(OUT/'database_combined_oof.npz', y=y, groups=g, ensemble=ens, **oofs)
    res['_meta'] = {'seconds': time.perf_counter()-tick, 'n_features': int(X.shape[1])}
    (OUT/'database_combined_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[done] {time.perf_counter()-tick:.0f}s', flush=True)


if __name__ == '__main__':
    main()
