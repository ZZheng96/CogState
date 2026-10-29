"""A 支模型族探索：收缩 LDA、RBF 核近似、特征抖动增强，并与已有集成对比。

全部只用当前窗口特征，不含任何顺序信息。
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline

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


def center(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0)+1e-12
    A, T = (Xtr-mu)/sd, (Xte-mu)/sd
    return A - A.mean(1, keepdims=True), T - T.mean(1, keepdims=True)


def main():
    tick = time.perf_counter()
    F, E, y, g = load()
    V = np.load(OUT/'features_v3.npz')['V']
    X = np.hstack([F[:, F.std(0) > 1e-8], E, V])
    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, g))
    print(f'[data] {X.shape}', flush=True)

    def evaluate(step, tag):
        oof = np.full(len(y), np.nan)
        for tr, te in splits:
            A, T = center(X[tr], X[te])
            tf = FeatureTransform(300, 'kbest').fit(A, y[tr])
            A, T = tf.transform(A).astype(np.float64), tf.transform(T).astype(np.float64)
            oof[te] = step(A, y[tr], T)
        r = report(y, oof, g)
        print(f'[{tag}] AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f} subjAUC={r["subject_AUC"]:.4f}', flush=True)
        return r, oof

    def lda(A, yy, T):
        m = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(A, yy)
        return m.decision_function(T)

    def rbf(A, yy, T):
        m = make_pipeline(RBFSampler(gamma='scale', n_components=200, random_state=0),
                          LogisticRegression(C=1, max_iter=2000, class_weight='balanced', random_state=0))
        return m.fit(A, yy).predict_proba(T)[:, 1]

    def jitter(A, yy, T, copies=3, sigma=.15):
        rng = np.random.default_rng(0)
        AA = np.vstack([A]+[A+rng.normal(0, sigma, A.shape) for _ in range(copies)])
        YY = np.tile(yy, copies+1)
        m = LogisticRegression(C=1, max_iter=2000, class_weight='balanced', random_state=0).fit(AA, YY)
        return m.predict_proba(T)[:, 1]

    res, oofs = {}, {}
    res['lda_shrinkage'], oofs['lda'] = evaluate(lda, '收缩 LDA · 300 维')
    res['rbf_logistic'], oofs['rbf'] = evaluate(rbf, 'RBF 核近似 + Logistic · 300 维')
    res['jitter_logistic'], oofs['jitter'] = evaluate(jitter, '抖动增强 + Logistic · 300 维')

    base = np.load(OUT/'database_combined_oof.npz')
    cand = {'ens_base': base['ensemble'], 'lda': oofs['lda'], 'rbf': oofs['rbf'], 'jitter': oofs['jitter']}
    R = {k: rankdata(v)/len(y) for k, v in cand.items()}
    import itertools
    combos, subs = {}, []
    for r in range(2, 5):
        for sub in itertools.combinations(cand, r):
            combos['+'.join(sub)] = report(y, np.mean([R[k] for k in sub], axis=0), g)
            subs.append(sub)
    best = max(combos, key=lambda k: combos[k]['AUC'])
    print(f'[best combo] {best} AUC={combos[best]["AUC"]:.4f} bACC={combos[best]["bACC"]:.4f}', flush=True)
    nested = []
    for f, (tr, te) in enumerate(splits, 1):
        b, ba = None, -1
        for sub in subs:
            a = roc_auc_score(y[tr], np.mean([R[k] for k in sub], axis=0)[tr])
            if a > ba:
                ba, b = a, sub
        s = np.mean([R[k] for k in b], axis=0)
        nested.append({'fold': f, 'combo': '+'.join(b), 'AUC': float(roc_auc_score(y[te], s[te]))})
        print(f'[nested {f}/5] {"+".join(b)} AUC={nested[-1]["AUC"]:.4f}', flush=True)
    res['combos'] = combos; res['best_combo'] = best; res['nested'] = nested
    res['nested_mean_AUC'] = float(np.mean([d['AUC'] for d in nested]))
    res['_meta'] = {'seconds': time.perf_counter()-tick}
    print(f'[nested mean] AUC={res["nested_mean_AUC"]:.4f}', flush=True)
    (OUT/'database_models_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    np.savez_compressed(OUT/'database_models_oof.npz', y=y, groups=g, **oofs)
    print(f'[done] {time.perf_counter()-tick:.0f}s', flush=True)


if __name__ == '__main__':
    main()
