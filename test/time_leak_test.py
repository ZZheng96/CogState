"""假设检验：block 序列固定为 R-T-R-T-R-T，若能从"流内位置"推出块序号，标签就被时间决定了。

测三项：(1) 仅用流内时间做特征；(2) 时间对标签的理论可分性；(3) 标签随位置的真实分布。
"""
import sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT/'test'
sys.path[:0] = [str(ROOT/'src'), str(OUT)]
from causal_history import rebuild_meta, report

F = np.load(OUT/'highdim_deep_features.npz')
y, groups = F['y'], F['groups']
meta = rebuild_meta(); end = np.array([m[2] for m in meta])
subjects = np.unique(groups)
t_in_stream = np.zeros(len(y)); t_abs = np.zeros(len(y))
for s in subjects:
    ids = np.where(groups == s)[0][np.argsort(end[groups == s])]
    t_in_stream[ids] = np.arange(len(ids))
    t_abs[ids] = end[ids] - end[ids].min()

print('[理论] 每受试者流内标签序列（按 60 窗一个块观察）:', flush=True)
for s in subjects[:6]:
    ids = np.where(groups == s)[0][np.argsort(end[groups == s])]
    print(f'   {s}: n={len(ids)} y={y[ids][::20].tolist()}', flush=True)

splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(t_in_stream[:, None], y, groups))
for name, col in (('流内窗口序号', t_in_stream[:, None]),
                  ('距流开始的秒数', t_abs[:, None]),
                  ('序号+秒数', np.c_[t_in_stream, t_abs])):
    oof = np.full(len(y), np.nan)
    for tr, te in splits:
        oof[te] = HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=.1,
                                                random_state=0).fit(col[tr], y[tr]).predict_proba(col[te])[:, 1]
    r = report(y, oof, groups, .5)
    print(f'[仅时间特征: {name}] AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f} subjAUC={r["subject_AUC"]:.4f}', flush=True)

# 位置-标签关系：按流内位置分箱看标签比例
bins = np.linspace(0, np.percentile(t_in_stream, 99), 11)
idx = np.digitize(t_in_stream, bins)-1
print('[position -> P(task)]', flush=True)
for b in range(10):
    m = idx == b
    if m.sum():
        print(f'   bin{b} t∈[{bins[b]:.0f},{bins[b+1]:.0f}] n={m.sum():5d} P(task)={y[m].mean():.3f}', flush=True)
