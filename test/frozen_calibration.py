"""冻结校准：只用每个受试者最初的 N 个窗口估计均值/方差，之后不再更新。

这样既保留逐受试者校准（合规：只用过去数据），又不含"现在是流内第几个窗口"的信息，
可以把"真实解码 + 受试者校准"与"位置泄漏"分开。
"""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT/'test'
sys.path[:0] = [str(ROOT/'src'), str(OUT)]
from causal_history import rebuild_meta, report
from highdim_deep_ablation import FeatureTransform

F = np.load(OUT/'highdim_deep_features.npz')
E = np.load(OUT/'emg_features_v2.npz')['E']
X0 = np.hstack([F['X'], E]); y, groups = F['y'], F['groups']
X0 = X0[:, X0.std(0) > 1e-8]
meta = rebuild_meta(); end = np.array([m[2] for m in meta])
subjects = np.unique(groups)
ids_time = [np.where(groups == s)[0][np.argsort(end[groups == s])] for s in subjects]
splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X0, y, groups))
rng = np.random.default_rng(0)


def norm(tr, N):
    """N>0: 用前 N 窗冻结；N=0: 用整段（oracle）；N<0: 随机 N 窗（对照）。"""
    mu0, var0 = X0[tr].mean(0), X0[tr].var(0)+1e-20
    out = np.empty_like(X0, dtype=float)
    for ids in ids_time:
        n = abs(N) if N else len(ids)
        cal = ids[:n] if N >= 0 else ids[rng.permutation(len(ids))[:n]]
        v = X0[cal].astype(float)
        m = (5*mu0 + v.sum(0))/(5+len(v))
        ex2 = (5*(var0+mu0**2) + (v**2).sum(0))/(5+len(v))
        s = np.sqrt(np.maximum(ex2-m**2, 1e-12))
        out[ids] = (X0[ids]-m)/s
    return out


res = {}
for N in (30, 60, 120, 0, -30):
    tag = f'first{N}' if N > 0 else ('whole' if N == 0 else 'random30')
    for k, mod in ((300, 'logistic'), (300, 'histgb'), (0, 'histgb')):
        oof = np.full(len(y), np.nan)
        for tr, te in splits:
            Xn = norm(tr, N)
            if k:
                tf = FeatureTransform(k, 'kbest').fit(Xn[tr], y[tr])
                A, T = tf.transform(Xn[tr]), tf.transform(Xn[te])
            else:
                A, T = Xn[tr], Xn[te]
            if mod == 'logistic':
                m = LogisticRegression(C=1, max_iter=2000, class_weight='balanced', random_state=0)
            else:
                m = HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_depth=3,
                                                   l2_regularization=1., random_state=0)
            oof[te] = m.fit(A, y[tr]).predict_proba(T)[:, 1]
        r = report(y, oof, groups, .5)
        res[f'{tag}_k{k or "all"}_{mod}'] = r
        print(f'[{tag:9s} k={k or "all":4}] {mod:9s} AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f} subjAUC={r["subject_AUC"]:.4f}', flush=True)
(OUT/'frozen_calibration_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
