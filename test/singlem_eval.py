"""评估 SingLEM 冻结表征：A 支（全局）与 B 支（K=120 分位数校准），并检验能否补充现有 P3 集成。

口径：StratifiedGroupKFold(5, seed 0)；B 支与合流一律剔除每人前 120 窗后评估。
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from b_calib_variants import subject_quantile

K = 120


def smooth(s, g, k):
    out = np.empty_like(s)
    for sub in np.unique(g):
        idx = np.where(g == sub)[0]; v = s[idx]
        c = np.cumsum(np.r_[0., v]); lo = np.maximum(np.arange(len(v))-k+1, 0)
        out[idx] = (c[np.arange(len(v))+1]-c[lo])/(np.arange(len(v))+1-lo)
    return out


def boot(a, b, y, g, mask, n=500):
    subs = np.unique(g); by = [np.where((g == s) & mask)[0] for s in subs]
    rng = np.random.default_rng(0); d = []
    for _ in range(n):
        idx = np.concatenate([by[i] for i in rng.integers(0, len(subs), len(subs))])
        if len(set(y[idx])) == 2:
            d.append(roc_auc_score(y[idx], a[idx])-roc_auc_score(y[idx], b[idx]))
    d = np.array(d)
    return {'mean': float(d.mean()), 'p2.5': float(np.percentile(d, 2.5)),
            'p97.5': float(np.percentile(d, 97.5)), 'win': float((d > 0).mean())}


def main():
    tick = time.perf_counter()
    d = np.load(OUT/'highdim_deep_features.npz'); y, g = d['y'].astype(int), d['groups']
    e = np.load(OUT/'singlem_embed.npz'); R, H = e['R'].astype(np.float64), e['H'].astype(np.float64)
    pos = np.empty(len(y), int)
    for s in np.unique(g):
        i = np.where(g == s)[0]; pos[i] = np.arange(len(i))
    mask = pos >= K
    feats = {'flat': R.reshape(len(R), -1),
             'stat': np.hstack([R.mean(1), R.std(1), H]),
             'all': np.hstack([R.reshape(len(R), -1), R.std(1), H])}
    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(R, y, g))
    models = {
        'lr': lambda: make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=3000)),
        'svc': lambda: make_pipeline(StandardScaler(), SVC(C=1, gamma='scale', cache_size=2048)),
        'histgb': lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_depth=3,
                                                         l2_regularization=1., random_state=0)}
    res, oofs = {}, {}
    for fname, F in feats.items():
        for branch, Z in (('A', F), ('B', subject_quantile(F, g, K))):
            for mname, mk in models.items():
                oof = np.full(len(y), np.nan)
                for tr, te in splits:
                    m = mk().fit(Z[tr], y[tr])
                    oof[te] = m.decision_function(Z[te]) if mname != 'histgb' else m.predict_proba(Z[te])[:, 1]
                tag = f'{branch}_{fname}_{mname}'
                r = {'full': float(roc_auc_score(y, oof)), 'excl': float(roc_auc_score(y[mask], oof[mask]))}
                res[tag], oofs[tag] = r, oof
                print(f'[result] {tag:18s} 全窗 {r["full"]:.4f} | 剔校准 {r["excl"]:.4f} '
                      f'({time.perf_counter()-tick:.0f}s)', flush=True)
    np.savez_compressed(OUT/'singlem_oof.npz', y=y, groups=g, **oofs)

    # 合流：P3（+A 支 SVC 版）为基线
    pool = {}
    for f in ('b_branch_oof.npz', 'b_calib_variants_oof.npz', 'b_calib2_oof.npz'):
        z = np.load(OUT/f); pool.update({k: z[k] for k in z.files if k not in ('y', 'groups')})
    pool['A_base_svc'] = np.load(OUT/'temporal_integration_oof.npz')['base_svc']
    rk = lambda v: rankdata(v)/len(v)
    P3 = ['quant120_histgb', 'quant120w_histgb', 'quant120_svc', 'base_svc']
    base = np.mean([rk(pool[k]) for k in P3], 0)
    bestA = max((k for k in res if k.startswith('A_')), key=lambda k: res[k]['full'])
    bestB = max((k for k in res if k.startswith('B_')), key=lambda k: res[k]['excl'])
    combos = {'P3': [rk(pool[k]) for k in P3],
              f'P3+{bestB}': [rk(pool[k]) for k in P3]+[rk(oofs[bestB])],
              f'P3+{bestB}+{bestA}': [rk(pool[k]) for k in P3]+[rk(oofs[bestB]), rk(oofs[bestA])],
              f'P3+2x{bestB}': [rk(pool[k]) for k in P3]+[rk(oofs[bestB])]*2}
    sb = smooth(base, g, 5)
    for name, parts in combos.items():
        s = np.mean(parts, 0); ss = smooth(s, g, 5)
        r = {'excl_raw': float(roc_auc_score(y[mask], s[mask])),
             'excl_30s': float(roc_auc_score(y[mask], ss[mask])),
             'excl_bACC': float(balanced_accuracy_score(y[mask], ss[mask] >= .5)),
             'vs_P3': boot(ss, sb, y, g, mask)}
        res[f'combo::{name}'] = r
        print(f'[combo] {name:40s} 剔校准 逐窗 {r["excl_raw"]:.4f} | 30s {r["excl_30s"]:.4f} / bACC '
              f'{r["excl_bACC"]:.4f} | Δ {r["vs_P3"]["mean"]:+.4f} [{r["vs_P3"]["p2.5"]:+.4f},'
              f'{r["vs_P3"]["p97.5"]:+.4f}] 胜率 {100*r["vs_P3"]["win"]:.0f}%', flush=True)
    # A 支合流（无校准）：现有 A 四路 + SingLEM A
    t = np.load(OUT/'temporal_integration_oof.npz')
    A4 = [rk(t[k]) for k in ('base_histgb', 'base_svc', 'warm_lr', 'warm_histgb')]
    for name, parts in (('A4', A4), (f'A4+{bestA}', A4+[rk(oofs[bestA])])):
        s = np.mean(parts, 0); ss = smooth(s, g, 5)
        r = {'full_raw': float(roc_auc_score(y, s)), 'full_30s': float(roc_auc_score(y, ss)),
             'bACC_30s': float(balanced_accuracy_score(y, ss >= .5))}
        res[f'Acombo::{name}'] = r
        print(f'[A combo] {name:28s} 逐窗 {r["full_raw"]:.4f} | 30s {r["full_30s"]:.4f} / bACC {r["bACC_30s"]:.4f}',
              flush=True)
    (OUT/'singlem_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[done] {time.perf_counter()-tick:.0f}s', flush=True)


if __name__ == '__main__':
    main()
