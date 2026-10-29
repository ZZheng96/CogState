"""分数侧校准：固定阈值 vs 因果在线阈值（均值/中位数/分位数），针对 bACC。

AUC 对单调变换不变，但 pooled AUC 会受"跨受试者分数偏移"影响，所以逐受试者的因果中心化会同时改变两者。
所有在线统计量只用该受试者已过去的窗口。
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT/'test'
sys.path[:0] = [str(ROOT/'src'), str(OUT)]
from causal_history import rebuild_meta, report

A = np.load(OUT/'online_deep_oof.npz'); B = np.load(OUT/'highdim_deep_oof.npz')
y, groups = A['y'], A['groups']
meta = rebuild_meta()
assert len(meta) == len(y)
end = np.array([m[2] for m in meta])
streams = [np.where(groups == s)[0][np.argsort(end[groups == s], kind='stable')] for s in np.unique(groups)]

base = np.mean([rankdata(A['online_mlp300'])/len(y), rankdata(A['online_logistic300'])/len(y),
                rankdata(B['large_mlp300'])/len(y)], axis=0)


def causal(fn, min_hist):
    out = np.empty_like(base)
    for ids in streams:
        out[ids] = fn(base[ids], min_hist)
    return out


def run_mean(v, m):
    o = v.copy()
    for i in range(m, len(v)):
        o[i] = v[i] - v[:i].mean()
    return o


def run_median(v, m):
    o = v.copy()
    for i in range(m, len(v)):
        o[i] = v[i] - np.median(v[:i])
    return o


def run_quantile(v, m):
    o = v.copy()
    for i in range(m, len(v)):
        o[i] = v[i] - np.quantile(v[:i], .5)
    return o


def run_rank(v, m):
    """把当前分数换成它在历史中的分位（0-1），对偏态与离群更稳健。"""
    o = v.copy()
    for i in range(m, len(v)):
        o[i] = (np.sum(v[:i] < v[i]) + .5*0.0)/len(v[:i])
    return o


results = {}
results['固定阈值0.5'] = report(y, base, groups, .5)
for name, fn, m in (('在线均值中心化', run_mean, 10), ('在线中位数中心化', run_median, 10),
                    ('在线中位数中心化(m=30)', run_median, 30), ('在线历史分位', run_rank, 30)):
    s = causal(fn, m)
    results[name] = report(y, s, groups, .5 if fn is run_rank else 0.)
    print(f'[{name}] AUC={results[name]["AUC"]:.4f} bACC={results[name]["bACC"]:.4f} '
          f'subjAUC={results[name]["subject_AUC"]:.4f} subjBACC={results[name]["subject_bACC"]:.4f}', flush=True)

# 全局固定阈值的上界（用真标签扫描，仅作参考）
best = max(((th, balanced_accuracy_score(y, base >= th)) for th in np.quantile(base, np.linspace(.3, .7, 81))),
           key=lambda t: t[1])
results['全局最优阈值(用真标签,参考)'] = {'threshold': float(best[0]), 'bACC': float(best[1]),
                                          'AUC': float(roc_auc_score(y, base))}
print(f'[全局最优阈值 参考] th={best[0]:.4f} bACC={best[1]:.4f}', flush=True)

# bACC 的阈值部分与环境无关：per-subject 各自最优阈值上界
per = []
for s in np.unique(groups):
    ids = groups == s
    th = max(((t, balanced_accuracy_score(y[ids], base[ids] >= t)) for t in np.quantile(base[ids], np.linspace(.2, .8, 61))),
             key=lambda t: t[1])
    per.append(th[1])
results['每受试者最优阈值(用真标签,参考)'] = {'bACC': float(np.mean(per))}
print(f'[每受试者最优阈值 参考] bACC={np.mean(per):.4f}', flush=True)

(OUT/'score_calibration_results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
