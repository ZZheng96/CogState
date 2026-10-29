"""定位 k=all 时的高 AUC 来源：依赖受试者内时间顺序？依赖逐受试者标准化？"""
import sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT/'test'
sys.path[:0] = [str(ROOT/'src'), str(OUT)]
from causal_history import rebuild_meta, report, running_stats

F = np.load(OUT/'highdim_deep_features.npz')
E = np.load(OUT/'emg_features_v2.npz')['E']
X0 = np.hstack([F['X'], E]); y, groups = F['y'], F['groups']
X0 = X0[:, X0.std(0) > 1e-8]
meta = rebuild_meta(); end = np.array([m[2] for m in meta])
subjects = np.unique(groups)
ids_time = [np.where(groups == s)[0][np.argsort(end[groups == s])] for s in subjects]
splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X0, y, groups))
ids_shuf = [np.random.default_rng(i).permutation(ids) for i, ids in enumerate(ids_time)]


def gb():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_depth=3,
                                          l2_regularization=1., random_state=0)


def cv(builder, tag):
    oof = np.full(len(y), np.nan)
    for tr, te in splits:
        Xn = builder(tr)
        oof[te] = gb().fit(Xn[tr], y[tr]).predict_proba(Xn[te])[:, 1]
    r = report(y, oof, groups, .5)
    per = [roc_auc_score(y[te], oof[te]) for _, te in splits]
    print(f'[{tag}] AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f} subjAUC={r["subject_AUC"]:.4f} '
          f'folds={np.round(per, 3).tolist()}', flush=True)
    return oof


def running(tr, ids_list):
    mu0, var0 = X0[tr].mean(0), X0[tr].var(0)+1e-20
    out = np.empty_like(X0, dtype=float)
    for ids in ids_list:
        m, s = running_stats(X0[ids].astype(float), 5, mu0, var0); out[ids] = (X0[ids]-m)/s
    return out


cv(lambda tr: running(tr, ids_time), '在线归一化(时间顺序)')
cv(lambda tr: running(tr, ids_shuf), '在线归一化(受试者内打乱顺序)')
cv(lambda tr: (X0 - X0[tr].mean(0))/(X0[tr].std(0)+1e-30), '全局标准化')


def per_subject_full(tr):
    out = np.empty_like(X0, dtype=float)
    for ids in ids_time:
        out[ids] = (X0[ids]-X0[ids].mean(0))/(X0[ids].std(0)+1e-30)
    return out


cv(per_subject_full, '逐受试者整段z-score(oracle)')


def per_subject_scale_only(tr):
    """只用逐受试者幅度缩放，不减均值：检验是"去偏移"还是"改尺度"在起作用。"""
    out = np.empty_like(X0, dtype=float)
    for ids in ids_time:
        out[ids] = X0[ids]/(X0[ids].std(0)+1e-30)
    return out


cv(per_subject_scale_only, '逐受试者仅缩放(整段)')
