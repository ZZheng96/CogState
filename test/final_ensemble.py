"""把目前所有可用的 OOF 放在一起做嵌套集成，确认 0.649 是否已被突破。"""
import json, sys, itertools
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT/'test'
sys.path[:0] = [str(OUT)]

C = np.load(OUT/'database_combined_oof.npz')
M = np.load(OUT/'database_models_oof.npz')
V4 = np.load(OUT/'features_v4_oof.npz')
V5 = np.load(OUT/'features_v5_oof.npz')
KT = np.load(OUT/'kernel_trees_oof.npz')
y, g = C['y'], C['groups']
pool = {'combined_logistic': C['combined_logistic'], 'combined_histgb': C['combined_histgb'],
        'ens_2way': C['ensemble'], 'lda': M['lda'], 'jitter': M['jitter'], 'rbf': M['rbf'],
        'v4': V4['v4'], 'base892': V4['base892'], 'base_v4': V4['base_v4'],
        'v5_all': V5['v5_all_lr'], 'base_v5_lr': V5['base_v5_lr'], 'base_v5_histgb': V5['base_v5_histgb'],
        'svc_rbf_c1': KT['svc_rbf_c1'], 'svc_rbf_c3': KT['svc_rbf_c3'],
        'extra_trees': KT['extra_trees'], 'extra_trees_half': KT['extra_trees_half']}
R = {k: rankdata(v)/len(y) for k, v in pool.items()}


def rep(s):
    r = {'AUC': float(roc_auc_score(y, s)), 'bACC': float(balanced_accuracy_score(y, s >= .5))}
    per = [{'AUC': roc_auc_score(y[g == k], s[g == k]), 'bACC': balanced_accuracy_score(y[g == k], s[g == k] >= .5)}
           for k in np.unique(g) if len(set(y[g == k])) == 2]
    r.update({f'subject_{m}': float(np.mean([d[m] for d in per])) for m in ('AUC', 'bACC')})
    return r


combos, subs = {}, []
for r in range(2, 5):
    for sub in itertools.combinations(pool, r):
        combos['+'.join(sub)] = rep(np.mean([R[k] for k in sub], axis=0))
        subs.append(sub)
best = max(combos, key=lambda k: combos[k]['AUC'])
print(f'[best full-OOF] {best} AUC={combos[best]["AUC"]:.4f} bACC={combos[best]["bACC"]:.4f}', flush=True)

splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(y[:, None], y, g))
nested = []
for f, (tr, te) in enumerate(splits, 1):
    b, ba = None, -1
    for sub in subs:
        a = roc_auc_score(y[tr], np.mean([R[k] for k in sub], axis=0)[tr])
        if a > ba:
            ba, b = a, sub
    s = np.mean([R[k] for k in b], axis=0)
    nested.append({'fold': f, 'combo': '+'.join(b), 'AUC': float(roc_auc_score(y[te], s[te]))})
    print(f'[nested {f}/5] {"+".join(b)} AUC={nested[-1]["AUC"]:.4f}', flush=True)
print(f'[nested mean] AUC={np.mean([d["AUC"] for d in nested]):.4f}', flush=True)
out = {'combos': combos, 'best': best, 'nested': nested,
       'nested_mean_AUC': float(np.mean([d['AUC'] for d in nested]))}
(OUT/'final_ensemble_results.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
