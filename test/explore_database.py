"""在去时序数据库上探索：所有变换都只用当前窗口（或训练折统计量），不含任何顺序信息。"""
import json, sys, time
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import QuantileTransformer

ROOT = Path(__file__).resolve().parents[1]; DB = ROOT/'database'; OUT = ROOT/'test'
sys.path[:0] = [str(OUT)]
from highdim_deep_ablation import FeatureTransform

MAN = json.loads((DB/'manifest.json').read_text(encoding='utf-8'))
NAMES = MAN['feature_names']


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


def run(X, y, g, make, tag, kbest=300, prep=None):
    oof = np.full(len(y), np.nan)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, g):
        A, T = X[tr], X[te]
        if prep:
            A, T = prep(A, T)
        if kbest:
            tf = FeatureTransform(kbest, 'kbest').fit(A, y[tr])
            A, T = tf.transform(A).astype(np.float64), tf.transform(T).astype(np.float64)
        oof[te] = make().fit(A, y[tr]).predict_proba(T)[:, 1]
    r = report(y, oof, g)
    print(f'[{tag}] AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f} subjAUC={r["subject_AUC"]:.4f}', flush=True)
    return r


def prep_global_center(A, T):
    """训练折统计量标准化后，扣掉每个窗口自身跨特征的平均（去掉整体增益）。"""
    mu, sd = A.mean(0), A.std(0)+1e-12
    A2, T2 = (A-mu)/sd, (T-mu)/sd
    return A2 - A2.mean(1, keepdims=True), T2 - T2.mean(1, keepdims=True)


LOGP = [i for i, n in enumerate(NAMES) if 'log' in n.lower()]


def prep_log_center(A, T):
    """只对对数功率类特征扣掉该窗自身的平均对数功率，保留绝对电平作为一个特征。"""
    mu, sd = A.mean(0), A.std(0)+1e-12
    A2, T2 = (A-mu)/sd, (T-mu)/sd
    out = []
    for M in (A2, T2):
        lvl = M[:, LOGP].mean(1, keepdims=True)
        M = M.copy()
        M[:, LOGP] = M[:, LOGP] - lvl
        out.append(np.hstack([M, lvl]))
    return out


def prep_quantile(A, T):
    qt = QuantileTransformer(output_distribution='normal', n_quantiles=1000, random_state=0).fit(A)
    return qt.transform(A), qt.transform(T)


def main():
    tick = time.perf_counter()
    F, E, y, g = load()
    F = F[:, F.std(0) > 1e-8]
    X = np.hstack([F, E])
    print(f'[load] X={X.shape} log-power cols={len(LOGP)}', flush=True)
    lr = lambda: LogisticRegression(C=1, max_iter=2000, class_weight='balanced', random_state=0)
    gb = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_depth=3,
                                                l2_regularization=1., random_state=0)
    res = {}
    res['base_logistic'] = run(X, y, g, lr, '基线 Logistic300')
    res['global_center_logistic'] = run(X, y, g, lr, '窗内全特征去整体电平', prep=prep_global_center)
    res['log_center_logistic'] = run(X, y, g, lr, '窗内对数功率去电平', prep=prep_log_center)
    res['quantile_logistic'] = run(X, y, g, lr, '分位变换', prep=prep_quantile)
    res['log_center_histgb'] = run(X, y, g, gb, '窗内对数功率去电平 · HistGB', kbest=0, prep=prep_log_center)
    res['base_histgb'] = run(X, y, g, gb, '基线 HistGB 全维', kbest=0)
    for C in (.01, .1, 10):
        res[f'logcenter_logistic_C{C}'] = run(
            X, y, g, lambda C=C: LogisticRegression(C=C, max_iter=2000, class_weight='balanced', random_state=0),
            f'窗内对数功率去电平 · C={C}', prep=prep_log_center)
    res['_meta'] = {'seconds': time.perf_counter()-tick}
    (OUT/'database_explore_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[done] {time.perf_counter()-tick:.0f}s', flush=True)


if __name__ == '__main__':
    main()
