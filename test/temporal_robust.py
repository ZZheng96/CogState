"""时序积分的鲁棒化：过去 k 窗用中位数/截尾均值/置信度加权替代均值。

现有 B 支的 30 s 积分固定为等权滑动平均，单个被伪迹污染的窗会直接拉偏结果。
这里只改"如何汇总过去 k 个已得分数"，全部因果，不用标签、不用块序号。
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from singlem_eval import smooth, boot

K = 120
P3 = ['quant120_histgb', 'quant120w_histgb', 'quant120_svc', 'base_svc']


def aggregate(v, k, mode):
    out = np.empty(len(v))
    for i in range(len(v)):
        w = v[max(0, i-k+1):i+1]
        if mode == 'mean':
            out[i] = w.mean()
        elif mode == 'median':
            out[i] = np.median(w)
        elif mode == 'trim':          # 去掉最大最小后的均值（k>=3 才有意义）
            out[i] = np.sort(w)[1:-1].mean() if len(w) >= 3 else w.mean()
        elif mode == 'conf':          # 按偏离中点的程度加权，越自信权重越大
            c = np.abs(w-.5) + 1e-6
            out[i] = (w*c).sum()/c.sum()
    return out


def main():
    d = np.load(OUT/'highdim_deep_features.npz'); y, g = d['y'].astype(int), d['groups']
    pos = np.empty(len(y), int)
    for s in np.unique(g):
        i = np.where(g == s)[0]; pos[i] = np.arange(len(i))
    mask = pos >= K
    pool = {}
    for f in ('b_branch_oof.npz', 'b_calib_variants_oof.npz', 'b_calib2_oof.npz'):
        z = np.load(OUT/f); pool.update({k: z[k] for k in z.files if k not in ('y', 'groups')})
    p3raw = np.mean([rankdata(pool[k])/len(y) for k in P3], 0)
    ref = smooth(p3raw, g, 5)
    res = {'reference_mean_k5': {'AUC': float(roc_auc_score(y[mask], ref[mask])),
                                 'bACC': float(balanced_accuracy_score(y[mask], ref[mask] >= np.median(ref[mask])))},
           'variants': {}}
    print(f'[ref] 均值 k=5 AUC={res["reference_mean_k5"]["AUC"]:.4f} '
          f'bACC={res["reference_mean_k5"]["bACC"]:.4f}', flush=True)

    for name, sc in (('raw', p3raw), ('smooth5', ref)):
        for mode in ('mean', 'median', 'trim', 'conf'):
            for k in (3, 5, 7):
                out = np.empty(len(y))
                for s in np.unique(g):
                    idx = np.where(g == s)[0]
                    out[idx] = aggregate(sc[idx], k, mode)
                r = {'AUC': float(roc_auc_score(y[mask], out[mask])),
                     'bACC': float(balanced_accuracy_score(y[mask], out[mask] >= np.median(out[mask]))),
                     'vs_mean5': boot(out, ref, y, g, mask)}
                key = f'{name}_{mode}_k{k}'
                res['variants'][key] = r
                print(f'[result] {key:18s} AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f} '
                      f'Δ={r["vs_mean5"]["mean"]:+.4f} [{r["vs_mean5"]["p2.5"]:+.4f},{r["vs_mean5"]["p97.5"]:+.4f}]',
                      flush=True)

    (OUT/'temporal_robust_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
