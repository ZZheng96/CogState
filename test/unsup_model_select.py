"""无标签的逐受试者模型选择：只用本人前 120 个校准窗决定权重。

动机：固定 rank 平均对所有受试者用同一组模型，但单个特征/模型在不同受试者上
表现差异很大（单个频带特征的受试者内 AUC 从 0.12 到 0.87）。这里不猜标签，
只用校准窗上分数的双峰分离度挑模型，属于校准阶段可用的无标签信息。
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from b_calib_variants import subject_quantile
from singlem_eval import smooth, boot

K = 120
P3 = ['quant120_histgb', 'quant120w_histgb', 'quant120_svc', 'base_svc']


def separation(z):
    """1 维两簇分离度：簇间方差 / 簇内方差。"""
    if len(z) < 20 or np.allclose(z, z[0]):
        return 0.
    km = KMeans(2, n_init=4, random_state=0).fit(z[:, None])
    a, b = km.labels_ == 0, km.labels_ == 1
    if a.sum() < 5 or b.sum() < 5:
        return 0.
    between = abs(z[a].mean() - z[b].mean())
    within = np.sqrt(z[a].var() + z[b].var()) + 1e-9
    return float(between/within)


def main():
    tick = time.perf_counter()
    d = np.load(OUT/'highdim_deep_features.npz'); y, g = d['y'].astype(int), d['groups']
    pos = np.empty(len(y), int)
    for s in np.unique(g):
        i = np.where(g == s)[0]; pos[i] = np.arange(len(i))
    mask = pos >= K
    pool = {}
    for f in ('b_branch_oof.npz', 'b_calib_variants_oof.npz', 'b_calib2_oof.npz'):
        z = np.load(OUT/f)
        pool.update({k: z[k] for k in z.files if k not in ('y', 'groups', 'fold')})
    names = [k for k, v in pool.items() if np.ndim(v) == 1 and len(v) == len(y)]
    print(f'[data] {len(names)} 个候选分数', flush=True)

    Z = {k: subject_quantile(pool[k][:, None], g, K)[:, 0] for k in names}
    p3 = np.mean([rankdata(pool[k])/len(y) for k in P3], 0)
    res = {'n_candidates': len(names),
           'P3': {'AUC': float(roc_auc_score(y[mask], p3[mask])),
                  'bACC': float(balanced_accuracy_score(y[mask], p3[mask] >= np.median(p3[mask])))},
           'P3_smooth': {'AUC': float(roc_auc_score(y[mask], smooth(p3, g, 5)[mask])),
                         'bACC': float(balanced_accuracy_score(y[mask], smooth(p3, g, 5)[mask] >= np.median(smooth(p3, g, 5)[mask])))},
           'variants': {}}
    print(f'[base] P3 AUC={res["P3"]["AUC"]:.4f}  P3+smooth AUC={res["P3_smooth"]["AUC"]:.4f}', flush=True)

    subs = np.unique(g)
    cal = {s: np.where((g == s) & (pos < K))[0] for s in subs}
    sep_all = {k: {s: separation(Z[k][cal[s]]) for s in subs} for k in names}
    # 用真标签在校准窗上算 AUC，只作为该选择机制的上限参考
    auc_all = {k: {s: (roc_auc_score(y[cal[s]], Z[k][cal[s]])
                        if len(set(y[cal[s]])) == 2 else .5) for s in subs} for k in names}
    res['oracle_calib_auc_mean'] = float(np.mean([auc_all[k][s] for k in names for s in subs]))

    for m in (1, 3, 5, 8):
        for mode in ('pick', 'oracle', 'inverse'):
            score = np.empty(len(y))
            for s in subs:
                if mode == 'pick':
                    order = sorted(names, key=lambda k: -sep_all[k][s])
                elif mode == 'oracle':
                    order = sorted(names, key=lambda k: -auc_all[k][s])
                else:  # inverse：只保留 P3 成员，按分离度加权
                    order = P3
                idx = np.where(g == s)[0]
                if mode == 'inverse':
                    w = np.array([max(sep_all[k][s], 1e-6) for k in order])
                    w = w/w.sum()
                    score[idx] = sum(wi*Z[k][idx] for wi, k in zip(w, order))
                else:
                    score[idx] = np.mean([Z[k][idx] for k in order[:m]], 0)
            for sm in (0, 5):
                v = smooth(score, g, sm) if sm else score
                thr = np.median(v[mask])
                r = {'AUC': float(roc_auc_score(y[mask], v[mask])),
                     'bACC': float(balanced_accuracy_score(y[mask], v[mask] >= thr)),
                     'vs_p3': boot(v, smooth(p3, g, 5), y, g, mask)}
                key = f'{"pick" if mode=="pick" else "oracle_cal" if mode=="oracle" else "p3_w"}_m{m}' + (f'_s{sm}' if sm else '')
                res['variants'][key] = r
                print(f'[result] {key:16s} AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f} '
                      f'Δ={r["vs_p3"]["mean"]:+.4f} [{r["vs_p3"]["p2.5"]:+.4f},{r["vs_p3"]["p97.5"]:+.4f}] '
                      f'({time.perf_counter()-tick:.0f}s)', flush=True)
    res['seconds'] = time.perf_counter()-tick
    (OUT/'unsup_model_select_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
