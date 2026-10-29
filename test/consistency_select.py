"""A 支：按"受试者内效应方向一致性"选特征，并测试主成分去偏。

动机：v5 诊断显示高维谱特征的受试者间差异会抵消状态效应。
SelectKBest 用全体汇总 F 值，容易把"受试者间可分的特征"选进来。
这里只统计训练折内每个受试者的 Cohen's d，再按跨受试者的方向一致性加权，
不使用任何顺序、时间或测试受试者信息。

第二个实验：在训练折拟合 PCA，丢弃前 k 个主方向（常对应受试者身份），
再用剩余方向训练。
"""
import json, sys, time
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]; DB = ROOT/'database'; OUT = ROOT/'test'
sys.path[:0] = [str(OUT)]
from highdim_deep_ablation import FeatureTransform


def load():
    fs, es, ys, gs = [], [], [], []
    for p in sorted((DB/'windows').glob('*/*.npz')):
        d = np.load(p); fs.append(d['f']); es.append(d['e']); ys.append(d['label']); gs.append(d['subject'])
    F = np.vstack(fs).astype(np.float64); E = np.vstack(es).astype(np.float64)
    F = F[:, F.std(0) > 1e-8]
    V3 = np.load(OUT/'features_v3.npz')['V']
    return np.hstack([F, E, V3]), np.concatenate(ys).astype(int), np.concatenate(gs)


def report(y, s, g):
    r = {'AUC': float(roc_auc_score(y, s)), 'bACC': float(balanced_accuracy_score(y, s >= .5))}
    per = [roc_auc_score(y[g == k], s[g == k]) for k in np.unique(g) if len(set(y[g == k])) == 2]
    r['subject_AUC'] = float(np.mean(per))
    return r


def consistency_scores(A, yy, gg):
    """训练折内的逐受试者 Cohen's d：返回 |median d| × 方向一致性。"""
    subs = np.unique(gg)
    D = []
    for s in subs:
        m = gg == s
        pos, neg = A[m & (yy == 1)], A[m & (yy == 0)]
        if len(pos) < 2 or len(neg) < 2:
            continue
        sd = np.sqrt((pos.var(0)+neg.var(0))/2)+1e-9
        D.append((pos.mean(0)-neg.mean(0))/sd)
    D = np.array(D)
    med = np.median(D, axis=0)
    agree = np.abs(np.mean(np.sign(D) == np.sign(med), axis=0))
    return np.abs(med)*agree, D


def main():
    tick = time.perf_counter(); X, y, g = load()
    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, g))
    print(f'[data] {X.shape} subjects={len(np.unique(g))}', flush=True)

    def evaluate(pipeline, tag):
        oof = np.full(len(y), np.nan)
        for fold, (tr, te) in enumerate(splits, 1):
            oof[te] = pipeline(X[tr], y[tr], g[tr], X[te], fold, tag)
            print(f'[{tag}] fold={fold}/5 elapsed={time.perf_counter()-tick:.0f}s', flush=True)
        r = report(y, oof, g)
        print(f'[result] {tag} AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f} subjAUC={r["subject_AUC"]:.4f}', flush=True)
        return r, oof

    def std_fit(A):
        return StandardScaler().fit(A)

    def consist_pipeline(k):
        def run(A, yy, gg, T, fold, tag):
            sc = std_fit(A); A2, T2 = sc.transform(A), sc.transform(T)
            score, _ = consistency_scores(A2, yy, gg)
            idx = np.argsort(-score)[:k]
            m = LogisticRegression(C=1, max_iter=3000, class_weight='balanced', random_state=0)
            return m.fit(A2[:, idx], yy).predict_proba(T2[:, idx])[:, 1]
        return run

    def consist_lda(k):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        def run(A, yy, gg, T, fold, tag):
            sc = std_fit(A); A2, T2 = sc.transform(A), sc.transform(T)
            score, _ = consistency_scores(A2, yy, gg)
            idx = np.argsort(-score)[:k]
            m = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(A2[:, idx], yy)
            return m.decision_function(T2[:, idx])
        return run

    def selectk_pipeline(k):
        def run(A, yy, gg, T, fold, tag):
            tf = FeatureTransform(k, 'kbest').fit(A, yy)
            m = LogisticRegression(C=1, max_iter=3000, class_weight='balanced', random_state=0)
            return m.fit(tf.transform(A).astype(np.float64), yy).predict_proba(tf.transform(T).astype(np.float64))[:, 1]
        return run

    def pca_drop(k, kfeat=300):
        def run(A, yy, gg, T, fold, tag):
            tf = FeatureTransform(kfeat, 'kbest').fit(A, yy)
            A2, T2 = tf.transform(A).astype(np.float64), tf.transform(T).astype(np.float64)
            p = PCA(n_components=min(k, A2.shape[1]-1), random_state=0).fit(A2)
            A2, T2 = A2-p.transform(A2)@p.components_, T2-p.transform(T2)@p.components_
            m = LogisticRegression(C=1, max_iter=3000, class_weight='balanced', random_state=0)
            return m.fit(A2, yy).predict_proba(T2)[:, 1]
        return run

    res, oofs = {}, {}
    for k in (100, 300, 700):
        res[f'consist_lr_{k}'], oofs[f'consist_lr_{k}'] = evaluate(consist_pipeline(k), f'一致性选特征 k={k} · Logistic')
    res['consist_lda_300'], oofs['consist_lda_300'] = evaluate(consist_lda(300), '一致性选特征 k=300 · LDA')

    def consist_histgb(k):
        def run(A, yy, gg, T, fold, tag):
            sc = std_fit(A); A2, T2 = sc.transform(A), sc.transform(T)
            idx = np.argsort(-consistency_scores(A2, yy, gg)[0])[:k]
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_depth=3,
                                               l2_regularization=1., random_state=0).fit(A2[:, idx], yy)
            return m.predict_proba(T2[:, idx])[:, 1]
        return run

    res['consist_lr_300_histgb'], oofs['consist_lr_300_histgb'] = evaluate(
        consist_histgb(300), '一致性选特征 k=300 · HistGB')
    res['selectk300'], oofs['selectk300'] = evaluate(selectk_pipeline(300), '基线 SelectKBest 300 · Logistic')
    for k in (1, 3, 5, 10):
        res[f'pca_drop_{k}'], oofs[f'pca_drop_{k}'] = evaluate(pca_drop(k), f'丢弃前 {k} 个主成分')

    np.savez_compressed(OUT/'consistency_oof.npz', y=y, groups=g, **oofs)
    res['_meta'] = {'seconds': time.perf_counter()-tick, 'n_features': int(X.shape[1])}
    (OUT/'consistency_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[done] elapsed={time.perf_counter()-tick:.0f}s', flush=True)


if __name__ == '__main__':
    main()
