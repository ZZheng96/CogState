"""区分"真实解码"与"位置泄漏"。

累积均值/方差是 t 的单调函数，会编码"现在是流内第几个窗口"，而标签由固定协议决定，
因此累积归一化可能把时间信息带进特征。这里对比：
  累积（全部历史） / 固定长度滑窗（最近 W 窗，同样因果、可部署）
并与"受试者内打乱顺序"这一非因果控制对照。
"""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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


def norm(tr, mode, W=30):
    """mode='cum' 累积；数字 = 最近 W 窗；'shuf' 受试者内打乱（非因果控制）。"""
    mu0, var0 = X0[tr].mean(0), X0[tr].var(0)+1e-20
    out = np.empty_like(X0, dtype=float)
    for i, ids in enumerate(ids_time):
        v = X0[ids].astype(float)
        if mode == 'shuf':
            perm = np.random.default_rng(i).permutation(len(ids))
            v = v[perm]
        n = np.arange(1, len(v)+1)[:, None]
        if mode == 'cum':
            m = (5*mu0 + np.cumsum(v, 0))/(5+n)
            ex2 = (5*(var0+mu0**2) + np.cumsum(v**2, 0))/(5+n)
        else:
            cs = np.cumsum(np.vstack([np.zeros_like(v[:1]), v]), 0)
            cs2 = np.cumsum(np.vstack([np.zeros_like(v[:1]), v**2]), 0)
            lo = np.maximum(np.arange(len(v))-W, -1)+1
            cnt = (np.arange(len(v))-lo+1)[:, None]
            m = (5*mu0 + (cs[1:] - cs[lo]))/(5+cnt)
            ex2 = (5*(var0+mu0**2) + (cs2[1:] - cs2[lo]))/(5+cnt)
        out[ids] = (v - m)/np.sqrt(np.maximum(ex2-m**2, 1e-12))
    return out


res = {}
CONFIGS = (('logistic', 300), ('histgb', 300), ('histgb', 0))
for mode in ('cum', 10, 30, 60, 'shuf'):
    tag = f'{"cum" if mode == "cum" else ("shuf" if mode == "shuf" else f"win{mode}")}'
    for mod, k in CONFIGS:
        oof = np.full(len(y), np.nan)
        for tr, te in splits:
            Xn = norm(tr, mode, W=(mode if isinstance(mode, int) else 30))
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
        print(f'[{tag:5s} k={k or "all":4}] {mod:9s} AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f} subjAUC={r["subject_AUC"]:.4f}', flush=True)
(OUT/'position_leak_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
