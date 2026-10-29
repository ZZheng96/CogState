"""分类器侧的测试时自训练（文献：T-TIME, Li et al. 2024；伪标签自训练）。

B 支只对特征做了一阶对齐（分位数校准），分类器本身仍是全局的。
这里给每名测试受试者训练一个“个人分类器”：
  伪标签 = 全局 P3 集成的 30 s 平滑分数在本人已见窗中的上/下 q 分位（不用真标签、不用块序号）。
  特征 = K=120 分位数校准后的 base 特征，按训练折 ANOVA 选 top-k。
  因果版：在第 i 窗只用 j<i 的窗训练（每 R 窗重训一次），对 i..i+R-1 预测。
  转导版（上界诊断）：用本人全部窗训练，非因果，只用于判断方向是否可行。
个人分数按本人已见预测转成因果百分位，与全局分数相加：s = global + α·(pct-0.5)。
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from b_calib_variants import subject_quantile
from singlem_eval import smooth, boot

K, REFIT = 120, 30


def personal(Z, sg, q, causal, C):
    """Z: 本人特征; sg: 本人全局平滑分数。返回个人分数（因果百分位）。"""
    n = len(sg); out = np.full(n, .5)
    if not causal:
        lo, hi = np.quantile(sg, [q, 1-q]); sel = (sg <= lo) | (sg >= hi)
        m = LogisticRegression(C=C, max_iter=2000).fit(Z[sel], (sg[sel] >= hi).astype(int))
        return rankdata(m.decision_function(Z))/n
    hist = []
    for start in range(K, n, REFIT):
        past = sg[:start]; lo, hi = np.quantile(past, [q, 1-q]); sel = np.where((past <= lo) | (past >= hi))[0]
        m = LogisticRegression(C=C, max_iter=2000).fit(Z[sel], (past[sel] >= hi).astype(int))
        hist_all = m.decision_function(Z[:start])
        for i in range(start, min(start+REFIT, n)):
            v = m.decision_function(Z[i:i+1])[0]
            out[i] = (np.searchsorted(np.sort(hist_all), v)+.5)/(len(hist_all)+1)
            hist_all = np.append(hist_all, v)
    return out


def main():
    tick = time.perf_counter()
    d = np.load(OUT/'highdim_deep_features.npz')
    X0, y, g = d['X'].astype(np.float64), d['y'].astype(int), d['groups']
    E = np.load(OUT/'emg_features_v2.npz')['E'].astype(np.float64)
    X0 = X0[:, X0.std(0) > 1e-8]
    Z = subject_quantile(np.hstack([X0, E]), g, K)
    pos = np.empty(len(y), int)
    for s in np.unique(g):
        i = np.where(g == s)[0]; pos[i] = np.arange(len(i))
    mask = pos >= K

    pool = {}
    for f in ('b_branch_oof.npz', 'b_calib_variants_oof.npz', 'b_calib2_oof.npz'):
        z = np.load(OUT/f); pool.update({k: z[k] for k in z.files if k not in ('y', 'groups')})
    P3 = ['quant120_histgb', 'quant120w_histgb', 'quant120_svc', 'base_svc']
    glob = np.mean([rankdata(pool[k])/len(y) for k in P3], 0)
    gs = smooth(glob, g, 5)
    base_r = {'excl_30s': float(roc_auc_score(y[mask], gs[mask])),
              'excl_bACC': float(balanced_accuracy_score(y[mask], gs[mask] >= .5))}
    print(f'[base] P3 30s 剔校准 AUC={base_r["excl_30s"]:.4f} bACC={base_r["excl_bACC"]:.4f}', flush=True)

    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Z, y, g))
    res = {'base': base_r}
    grid = [(c, k, q, C) for c in (False, True) for k in (20, 60) for q in (.3,) for C in (.05,)]
    grid += [(True, 60, .2, .05), (True, 60, .4, .05), (True, 60, .3, .5), (True, 150, .3, .05)]
    grid += [(True, 300, .3, .05), (True, 300, .4, .01), (True, 0, .3, .01), (False, 0, .3, .01)]  # k=0 表示全部特征
    for causal, kb, q, C in grid:
        P = np.full(len(y), .5)
        for tr, te in splits:
            F, _ = f_classif(Z[tr], y[tr]); cols = np.argsort(np.nan_to_num(F))[::-1][:kb or None]
            for s in np.unique(g[te]):
                idx = np.where(g == s)[0]
                P[idx] = personal(Z[np.ix_(idx, cols)], gs[idx], q, causal, C)
        Ps = smooth(P, g, 5)
        per_auc = float(np.mean([roc_auc_score(y[(g == s) & mask], P[(g == s) & mask]) for s in np.unique(g)
                                 if len(set(y[(g == s) & mask])) == 2]))
        tag = f'{"causal" if causal else "transd"}_k{kb}_q{q}_C{C}'
        row = {'personal_subjAUC': per_auc}
        for a in (.25, .5, 1.):
            s = gs+a*(Ps-.5)
            row[f'a{a}'] = {'excl_30s': float(roc_auc_score(y[mask], s[mask])),
                            'excl_bACC': float(balanced_accuracy_score(y[mask], s[mask] >= .5))}
        best_a = max((.25, .5, 1.), key=lambda a: row[f'a{a}']['excl_30s'])
        row['best_a'] = best_a
        row['vs_base'] = boot(gs+best_a*(Ps-.5), gs, y, g, mask)
        res[tag] = row
        print(f'[result] {tag:24s} 个人分类器受试者内AUC={per_auc:.4f} | '
              + ' '.join(f'α{a}:{row[f"a{a}"]["excl_30s"]:.4f}' for a in (.25, .5, 1.))
              + f' | Δ {row["vs_base"]["mean"]:+.4f} [{row["vs_base"]["p2.5"]:+.4f},{row["vs_base"]["p97.5"]:+.4f}]'
              f' 胜率 {100*row["vs_base"]["win"]:.0f}% ({time.perf_counter()-tick:.0f}s)', flush=True)
    # 迭代自训练：用上一轮合成分数（已 30 s 平滑、因果）作为下一轮伪标签源
    for kb, a in ((300, .5),):
        F_cols = []
        for tr, te in splits:
            F, _ = f_classif(Z[tr], y[tr]); F_cols.append((te, np.argsort(np.nan_to_num(F))[::-1][:kb]))
        cur = gs.copy()
        for r in range(1, 4):
            P = np.full(len(y), .5)
            for te, cols in F_cols:
                for s in np.unique(g[te]):
                    idx = np.where(g == s)[0]
                    P[idx] = personal(Z[np.ix_(idx, cols)], cur[idx], .3, True, .05)
            comb = gs+a*(smooth(P, g, 5)-.5)
            b = boot(comb, gs, y, g, mask)
            res[f'iter_k{kb}_r{r}'] = {'excl_30s': float(roc_auc_score(y[mask], comb[mask])),
                                      'excl_bACC': float(balanced_accuracy_score(y[mask], comb[mask] >= .5)), 'vs_base': b}
            print(f'[iter] k{kb} round{r}: AUC={res[f"iter_k{kb}_r{r}"]["excl_30s"]:.4f} '
                  f'bACC={res[f"iter_k{kb}_r{r}"]["excl_bACC"]:.4f} Δ {b["mean"]:+.4f} [{b["p2.5"]:+.4f},{b["p97.5"]:+.4f}]', flush=True)
            cur = comb
    gsub = float(np.mean([roc_auc_score(y[(g == s) & mask], gs[(g == s) & mask]) for s in np.unique(g)
                          if len(set(y[(g == s) & mask])) == 2]))
    res['base']['subjAUC'] = gsub
    print(f'[ref] 全局 P3 30s 受试者内AUC={gsub:.4f}', flush=True)
    (OUT/'self_training_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
