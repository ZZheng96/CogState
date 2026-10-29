"""恢复历史扩展实验：仅重建 D_exclude 与旧 A_include 的共同 OOF。"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
K0, SEED = 120, 0
COMBOS = {'A': (100, 200, 400), 'D': (50, 100, 200, 400, 800)}
sys.path.insert(0, str(OUT))
from b_calib_variants import subject_quantile

class Tee:
    def __init__(self, stream, path):
        self.stream, self.file = stream, path.open('a', encoding='utf-8')
    def write(self, x):
        self.stream.write(x); self.stream.flush(); self.file.write(x); self.file.flush()
    def flush(self): self.stream.flush(); self.file.flush()
    def close(self): self.file.close()

def causal_mean_std(z, g, k, exclude_current):
    mean = np.empty_like(z); std = np.empty_like(z)
    for s in np.unique(g):
        ii = np.where(g == s)[0]; v = z[ii]
        cs = np.vstack([np.zeros((1, v.shape[1])), np.cumsum(v, axis=0)])
        cs2 = np.vstack([np.zeros((1, v.shape[1])), np.cumsum(v*v, axis=0)])
        t = np.arange(len(ii))
        if exclude_current:
            lo = np.maximum(t-k, 0); hi = t; n = t-lo
            m = np.divide(cs[hi]-cs[lo], n[:, None], out=np.zeros_like(v), where=n[:, None] > 0)
            q = np.divide(cs2[hi]-cs2[lo], n[:, None], out=np.zeros_like(v), where=n[:, None] > 0)
        else:
            lo = np.maximum(t-k+1, 0); hi = t+1; n = hi-lo
            m = (cs[hi]-cs[lo])/n[:, None]; q = (cs2[hi]-cs2[lo])/n[:, None]
        mean[ii], std[ii] = m, np.sqrt(np.maximum(q-m*m, 0))
    return mean, std

def smooth(v, g, k=5):
    out = np.full_like(v, np.nan)
    for s in np.unique(g):
        ii = np.where(g == s)[0]; x = v[ii]; good = np.isfinite(x)
        cs = np.cumsum(np.r_[0., np.where(good, x, 0.)]); t = np.arange(len(ii)); lo = np.maximum(t-k+1, 0)
        ng = np.cumsum(np.r_[0, good.astype(int)]); n = ng[t+1]-ng[lo]
        out[ii] = np.divide(cs[t+1]-cs[lo], n, out=np.full(len(ii), np.nan), where=n > 0)
    return out

def metrics(v, y, mask):
    m = mask & np.isfinite(v)
    return {'n': int(m.sum()), 'AUC': float(roc_auc_score(y[m], v[m])),
            'bACC': float(balanced_accuracy_score(y[m], v[m] >= .5))}

def model():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_depth=3,
                                          l2_regularization=1., random_state=0)

def oof(F, y, g, mask, splits, label, start):
    pred = np.full(len(y), np.nan)
    for fold, (tr, te) in enumerate(splits, 1):
        tr, te = tr[mask[tr]], te[mask[te]]
        pred[te] = model().fit(F[tr], y[tr]).predict_proba(F[te])[:, 1]
        print(f'[progress] {label} fold {fold}/5 train={len(tr)} test={len(te)} elapsed={time.perf_counter()-start:.1f}s', flush=True)
    return pred

def bootstrap(new, old, y, g, mask, n=2000):
    subs = np.unique(g); pools = [np.where((g == s) & mask)[0] for s in subs]
    rng = np.random.default_rng(0); d = np.empty((n, 2))
    for i in range(n):
        ix = np.concatenate([pools[j] for j in rng.integers(0, len(subs), len(subs))])
        d[i] = (roc_auc_score(y[ix], new[ix])-roc_auc_score(y[ix], old[ix]),
                balanced_accuracy_score(y[ix], new[ix] >= .5)-balanced_accuracy_score(y[ix], old[ix] >= .5))
        if (i + 1) % 500 == 0: print(f'[progress] bootstrap {i+1}/{n}', flush=True)
    return {name: {'mean': float(np.mean(d[:, j])), 'p2.5': float(np.percentile(d[:, j], 2.5)),
                   'p97.5': float(np.percentile(d[:, j], 97.5))} for j, name in enumerate(('dAUC', 'dbACC'))}

def main():
    start = time.perf_counter(); tee = Tee(sys.stdout, OUT/'context_history_extend.log'); sys.stdout = tee
    try:
        print('[recover] rebuilding D_exclude_current and A_include_current shared OOF', flush=True)
        d = np.load(OUT/'highdim_deep_features.npz'); X = d['X'].astype(np.float64); y = d['y'].astype(int); g = d['groups']
        E = np.load(OUT/'emg_features_v2.npz')['E'].astype(np.float64)
        base = np.hstack([X, E]); base = base[:, X.std(0) > 1e-8] if False else np.hstack([X[:, X.std(0) > 1e-8], E])
        Z = subject_quantile(base, g, K0)
        pos = np.empty(len(y), int)
        for s in np.unique(g): pos[np.where(g == s)[0]] = np.arange(np.sum(g == s))
        mask = pos >= K0
        splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(Z, y, g))
        oofs = {}; results = {'meta': {'data': ['highdim_deep_features.npz', 'emg_features_v2.npz'], 'warmup_windows': K0,
            'strict_excludes_current': True, 'folds': 5, 'split_seed': SEED, 'eval_n': int(mask.sum()),
            'combos': {k: list(v) for k, v in COMBOS.items()}, 'bootstrap_reps': 2000,
            'comparison': 'D_exclude_current vs A_include_current (旧基线 0.8949 smooth5)'}, 'models': {}, 'combos': {}}
        with threadpool_limits(limits=4):
            for mode, ks in [('exclude_current', COMBOS['D']), ('include_current', COMBOS['A'])]:
                for k in ks:
                    mu, sd = causal_mean_std(Z, g, k, mode == 'exclude_current')
                    F = (Z-mu)/np.maximum(sd, .1)
                    key = f'd{k}_mean_std_{mode}'; oofs[key] = oof(F, y, g, mask, splits, key, start)
            for cname, mode in [('D', 'exclude_current'), ('A', 'include_current')]:
                p = np.mean([oofs[f'd{k}_mean_std_{mode}'] for k in COMBOS[cname]], axis=0)
                key = f'{cname}_{mode}'; oofs[key] = p
                results['combos'][key] = {'members': list(COMBOS[cname]), 'raw': metrics(p, y, mask), 'smooth5': metrics(smooth(p, g), y, mask)}
                print(f'[result] combo {key} raw AUC={results["combos"][key]["raw"]["AUC"]:.4f} bACC={results["combos"][key]["raw"]["bACC"]:.4f}; smooth5 AUC={results["combos"][key]["smooth5"]["AUC"]:.4f} bACC={results["combos"][key]["smooth5"]["bACC"]:.4f}', flush=True)
        new, old = smooth(oofs['D_exclude_current'], g), smooth(oofs['A_include_current'], g)
        results['bootstrap_D_exclude_vs_A_include'] = bootstrap(new, old, y, g, mask, 2000)
        results['old_baseline_smooth5_AUC'] = 0.8949
        results['new_smooth5_AUC'] = results['combos']['D_exclude_current']['smooth5']['AUC']
        results['exceeds_old_0.8949'] = bool(results['new_smooth5_AUC'] > .8949)
        results['seconds'] = time.perf_counter()-start; results['deviations'] = '无；仅重建共同 OOF，沿用原始数据、特征、折分和模型配置。'
        np.savez(OUT/'context_history_extend_oof.npz', y=y, groups=g, eval_mask=mask, **oofs)
        (OUT/'context_history_extend_results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'[done] seconds={results["seconds"]:.1f}', flush=True)
    finally:
        sys.stdout = tee; tee.close()

if __name__ == '__main__': main()
