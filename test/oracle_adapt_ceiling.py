"""上限诊断：如果每人前 120 个校准窗带真标签，逐受试者自适应能达到多少。

不是可提交方案（用了测试标签），只用于判断瓶颈在"自适应方法"还是"信号本身"。
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from b_calib_variants import subject_quantile
from singlem_eval import smooth, boot

K = 120
P3 = ['quant120_histgb', 'quant120w_histgb', 'quant120_svc', 'base_svc']


def main():
    tick = time.perf_counter()
    d = np.load(OUT/'highdim_deep_features.npz')
    X0, y, g = d['X'].astype(np.float64), d['y'].astype(int), d['groups']
    E = np.load(OUT/'emg_features_v2.npz')['E'].astype(np.float64)
    X0 = X0[:, X0.std(0) > 1e-8]
    F = np.hstack([X0, E])
    pos = np.empty(len(y), int)
    for s in np.unique(g):
        i = np.where(g == s)[0]; pos[i] = np.arange(len(i))
    mask = pos >= K
    pool = {}
    for f in ('b_branch_oof.npz', 'b_calib_variants_oof.npz', 'b_calib2_oof.npz'):
        z = np.load(OUT/f); pool.update({k: z[k] for k in z.files if k not in ('y', 'groups')})
    p3 = smooth(np.mean([rankdata(pool[k])/len(y) for k in P3], 0), g, 5)
    res = {'base_p3': {'AUC': float(roc_auc_score(y[mask], p3[mask])),
                       'bACC': float(balanced_accuracy_score(y[mask], p3[mask] >= np.median(p3[mask])))},
           'variants': {}}
    print(f'[base] P3 AUC={res["base_p3"]["AUC"]:.4f}', flush=True)

    for ncomp in (5, 10, 20):
        for C in (.01, .1):
            sc = np.full(len(y), np.nan)
            for s in np.unique(g):
                idx = np.where(g == s)[0]
                cal, rest = idx[:K], idx[K:]
                if len(rest) == 0 or len(set(y[cal])) < 2:
                    continue
                # 标准化只用本人全部窗（无标签）；PCA/LR 只用校准窗
                mu, sd = F[idx].mean(0), F[idx].std(0)+1e-12
                A, B = (F[cal]-mu)/sd, (F[rest]-mu)/sd
                p = PCA(min(ncomp, A.shape[0]-1, A.shape[1]), random_state=0).fit(A)
                A, B = p.transform(A), p.transform(B)
                m = LogisticRegression(C=C, max_iter=3000, class_weight='balanced').fit(A, y[cal])
                sc[rest] = m.predict_proba(B)[:, 1]
            for sm in (0, 5):
                if sm:  # smooth 遇到 NaN 会污染整个受试者序列，先用本人中位数填补
                    fill = sc.copy()
                    for s in np.unique(g):
                        idx = np.where(g == s)[0]; fin = np.isfinite(fill[idx])
                        if fin.any():
                            fill[idx] = np.where(np.isfinite(fill[idx]), fill[idx], np.median(fill[idx][fin]))
                    v = smooth(fill, g, sm)
                else:
                    v = sc
                ok = mask & np.isfinite(v)
                r = {'AUC': float(roc_auc_score(y[ok], v[ok])),
                     'bACC': float(balanced_accuracy_score(y[ok], v[ok] >= np.median(v[ok]))),
                     'vs_p3': boot(v, p3, y, g, ok), 'n_eval': int(ok.sum())}
                key = f'oracle120_pca{ncomp}_C{C}' + ('_s5' if sm else '')
                res['variants'][key] = r
                print(f'[result] {key:26s} AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f} '
                      f'Δ={r["vs_p3"]["mean"]:+.4f} [{r["vs_p3"]["p2.5"]:+.4f},{r["vs_p3"]["p97.5"]:+.4f}] '
                      f'({time.perf_counter()-tick:.0f}s)', flush=True)

    # 分数层面：用校准窗真标签给 30 个候选分数做受试者内加权（上限）
    names = [k for k, v in pool.items() if np.ndim(v) == 1 and len(v) == len(y)]
    Z = {k: subject_quantile(pool[k][:, None], g, K)[:, 0] for k in names}
    for topm in (3, 8):
        sc = np.empty(len(y))
        for s in np.unique(g):
            idx = np.where(g == s)[0]; cal = idx[:K]
            aucs = [(roc_auc_score(y[cal], Z[k][cal]), k) for k in names if len(set(y[cal])) == 2]
            best = [k for _, k in sorted(aucs, reverse=True)[:topm]]
            sc[idx] = np.mean([Z[k][idx] for k in best], 0)
        v = smooth(sc, g, 5)
        r = {'AUC': float(roc_auc_score(y[mask], v[mask])),
             'bACC': float(balanced_accuracy_score(y[mask], v[mask] >= np.median(v[mask]))),
             'vs_p3': boot(v, p3, y, g, mask)}
        res['variants'][f'oracle120_scoretop{topm}'] = r
        print(f'[result] oracle120_scoretop{topm} AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f}', flush=True)

    res['seconds'] = time.perf_counter()-tick
    (OUT/'oracle_adapt_ceiling_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
