"""伪迹混淆诊断：现有 0.742 有多少来自伪迹统计，而不是状态本身。

三个问题：
1. 单看肌电/眨眼类伪迹特征，能分到什么程度（合并 + 受试者内）；
2. 把伪迹特征从分数里回归掉之后，还剩多少；
3. 只在"干净窗"或"脏窗"上评估，AUC 是升还是降。
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from singlem_eval import smooth, boot

K = 120
P3 = ['quant120_histgb', 'quant120w_histgb', 'quant120_svc', 'base_svc']


def main():
    d = np.load(OUT/'highdim_deep_features.npz'); y, g = d['y'].astype(int), d['groups']
    pos = np.empty(len(y), int)
    for s in np.unique(g):
        i = np.where(g == s)[0]; pos[i] = np.arange(len(i))
    mask = pos >= K
    E = np.load(OUT/'emg_blink_features.npz')['E'].astype(np.float64)
    names = json.load(open(OUT/'emg_blink_results.json', encoding='utf-8'))['_meta']['names']
    pool = {}
    for f in ('b_branch_oof.npz', 'b_calib_variants_oof.npz', 'b_calib2_oof.npz'):
        z = np.load(OUT/f); pool.update({k: z[k] for k in z.files if k not in ('y', 'groups')})
    p3 = smooth(np.mean([rankdata(pool[k])/len(y) for k in P3], 0), g, 5)
    res = {'p3': {'AUC': float(roc_auc_score(y[mask], p3[mask])),
                  'bACC': float(balanced_accuracy_score(y[mask], p3[mask] >= np.median(p3[mask])))},
           'artifact_features': {}}
    print(f'[base] P3 AUC={res["p3"]["AUC"]:.4f}', flush=True)

    subs = [s for s in np.unique(g) if len(set(y[g == s])) == 2]
    for j, n in enumerate(names):
        per = np.array([roc_auc_score(y[g == s], E[g == s, j]) for s in subs])
        res['artifact_features'][n] = {'pooled_auc': float(roc_auc_score(y, E[:, j])),
                                       'subject_mean_auc': float(per.mean()),
                                       'subject_frac_above_0.5': float((per > .5).mean(),
                                                                        )}
        print(f'[art] {n:18s} pooled={res["artifact_features"][n]["pooled_auc"]:.3f} '
              f'subj={per.mean():.3f} 方向一致={np.mean(per>.5):.2f}', flush=True)

    # 2) 用本人校准窗拟合"分数 ~ 单一伪迹指数"，把残差作为去伪迹后的分数
    #    15 维伪迹特征直接回归会过拟合 120 窗（AUC 掉到 0.56），因此先压成 1 维再回归。
    resid = p3.copy()
    for s in subs:
        idx = np.where(g == s)[0]
        Z = np.column_stack([E[idx, j] for j in range(E.shape[1])])
        loc = np.arange(len(idx)); cal_l, ev_l = loc[:K], loc[K:]
        Z = (Z - Z[cal_l].mean(0))/(Z[cal_l].std(0)+1e-12)
        z = PCA(1, random_state=0).fit(Z[cal_l]).transform(Z)[:, 0]
        m = LinearRegression().fit(z[cal_l][:, None], p3[idx][cal_l])
        resid[idx[ev_l]] = p3[idx][ev_l] - m.predict(z[ev_l][:, None])
    v = smooth(resid, g, 5)
    res['p3_residualised'] = {'AUC': float(roc_auc_score(y[mask], v[mask])),
                              'bACC': float(balanced_accuracy_score(y[mask], v[mask] >= np.median(v[mask]))),
                              'vs_p3': boot(v, p3, y, g, mask)}
    print(f'[resid] 去伪迹后 AUC={res["p3_residualised"]["AUC"]:.4f} '
          f'Δ={res["p3_residualised"]["vs_p3"]["mean"]:+.4f}', flush=True)

    # 3) 按伪迹负担分层：用肌电高频列做中位数切分
    for j, n in ((8, names[8]), (12, names[12])):
        v_ = E[:, j]
        subs_list = list(np.unique(g))
        thr = {s: np.median(v_[g == s]) for s in subs_list}
        dirty = np.array([v_[i] > thr[g[i]] for i in range(len(y))])
        row = {}
        for tag, sel in (('clean', mask & ~dirty), ('dirty', mask & dirty)):
            row[tag] = {'n': int(sel.sum()), 'AUC': float(roc_auc_score(y[sel], p3[sel])),
                        'bACC': float(balanced_accuracy_score(y[sel], p3[sel] >= np.median(p3[sel])))}
        res[f'stratify_{n}'] = row
        print(f'[strata] {n}: clean AUC={row["clean"]["AUC"]:.4f} (n={row["clean"]["n"]})  '
              f'dirty AUC={row["dirty"]["AUC"]:.4f} (n={row["dirty"]["n"]})', flush=True)

    (OUT/'artifact_confound_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
