"""只用数据库里的 (窗口特征, 标签, 受试者) 训练，完全不使用任何时序信息。

数据库按标签组织、受试者内顺序被固定种子打乱、不存时间戳，因此不存在"第几个窗口"。
这里只做全局模型，不做任何逐受试者在线统计。受试者仅用于分组交叉验证。
"""
import json, sys, time
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]; DB = ROOT/'database'; OUT = ROOT/'test'
sys.path[:0] = [str(OUT)]
from highdim_deep_ablation import FeatureTransform


def load():
    xs, fs, es, ys, gs = [], [], [], [], []
    for p in sorted((DB/'windows').glob('*/*.npz')):
        d = np.load(p)
        fs.append(d['f']); es.append(d['e']); ys.append(d['label']); gs.append(d['subject'])
    return (np.vstack(fs).astype(np.float64), np.vstack(es).astype(np.float64),
            np.concatenate(ys).astype(int), np.concatenate(gs))


def score(y, s):
    return {'AUC': float(roc_auc_score(y, s)), 'bACC': float(balanced_accuracy_score(y, s >= .5))}


def report(y, s, g):
    r = score(y, s)
    per = [score(y[g == k], s[g == k]) for k in np.unique(g) if len(set(y[g == k])) == 2]
    r.update({f'subject_{m}': float(np.mean([d[m] for d in per])) for m in ('AUC', 'bACC')})
    return r


def cv(X, y, g, model, tag, kbest=None):
    oof = np.full(len(y), np.nan)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, g):
        if kbest:
            tf = FeatureTransform(kbest, 'kbest').fit(X[tr], y[tr])
            A, T = tf.transform(X[tr]).astype(np.float64), tf.transform(X[te]).astype(np.float64)
        else:
            A, T = X[tr], X[te]
        oof[te] = model().fit(A, y[tr]).predict_proba(T)[:, 1]
    r = report(y, oof, g)
    print(f'[{tag}] AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f} subjAUC={r["subject_AUC"]:.4f}', flush=True)
    return r, oof


def main():
    tick = time.perf_counter()
    F, E, y, g = load()
    print(f'[load] {F.shape} emg={E.shape} relax={int((y == 0).sum())} task={int((y == 1).sum())} '
          f'subjects={len(np.unique(g))} {time.perf_counter()-tick:.1f}s', flush=True)
    keep = F.std(0) > 1e-8
    F = F[:, keep]
    X = np.hstack([F, E])

    gb = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_depth=3,
                                                l2_regularization=1., random_state=0)
    lr = lambda: LogisticRegression(C=1, max_iter=2000, class_weight='balanced', random_state=0)

    res = {}
    res['logistic300_EEG'], _ = cv(F, y, g, lr, 'Logistic300 · 仅EEG', 300)
    res['histgb300_EEG'], _ = cv(F, y, g, gb, 'HistGB300 · 仅EEG', 300)
    res['histgb_all_EEG'], _ = cv(F, y, g, gb, 'HistGB · 仅EEG 全维')
    res['histgb_all_EEGEMG'], _ = cv(X, y, g, gb, 'HistGB · EEG+肌电 全维')
    res['logistic300_EEGEMG'], _ = cv(X, y, g, lr, 'Logistic300 · EEG+肌电', 300)

    # 每窗自归一化（只用本窗，无需任何历史或顺序）：按该窗宽带幅度缩放全部特征
    scale = np.linalg.norm(F, axis=1, keepdims=True)+1e-12
    res['histgb_all_selfnorm'], _ = cv(np.hstack([F/scale, E]), y, g, gb, 'HistGB · 窗内自归一化')
    res['_meta'] = {'seconds': time.perf_counter()-tick, 'n_windows': int(len(y)),
                    'n_features': int(F.shape[1]), 'note': 'no ordering, no per-subject statistics'}
    (OUT/'database_training_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[done] {time.perf_counter()-tick:.0f}s', flush=True)


if __name__ == '__main__':
    main()
