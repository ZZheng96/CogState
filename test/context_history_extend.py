"""预注册历史长度扩展：严格排除当前窗，并保留含当前窗对照。"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
K0, SEED = 120, 0
HIST = (50, 100, 150, 200, 300, 400, 600, 800)
COMBOS = {'A': (100, 200, 400), 'B': (100, 200, 400, 800),
          'C': (150, 300, 600), 'D': (50, 100, 200, 400, 800)}
sys.path.insert(0, str(OUT))
from b_calib_variants import subject_quantile

class Tee:
    def __init__(self, stream, path):
        self.stream, self.file = stream, path.open('w', encoding='utf-8')
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

def metrics(v, y, g, mask):
    m = mask & np.isfinite(v)
    out = {'n': int(m.sum()), 'AUC': float(roc_auc_score(y[m], v[m])),
           'bACC': float(balanced_accuracy_score(y[m], v[m] >= .5))}
    dirs = {'R->T': [], 'T->R': []}
    for s in np.unique(g):
        ii = np.where(g == s)[0]; cuts = np.flatnonzero(np.r_[True, np.diff(y[ii]) != 0, True])
        blocks = [ii[a:b] for a, b in zip(cuts[:-1], cuts[1:])]
        for a, b in zip(blocks[:-1], blocks[1:]):
            sel = np.r_[a, b]; sel = sel[m[sel]]
            if len(sel) and len(np.unique(y[sel])) == 2:
                dirs['R->T' if y[a[0]] == 0 else 'T->R'].append(roc_auc_score(y[sel], v[sel]))
    out.update({key: (float(np.mean(val)) if val else None) for key, val in dirs.items()})
    return out

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

def bootstrap(a, b, y, g, mask, n=2000):
    subs = np.unique(g); pools = [np.where((g == s) & mask)[0] for s in subs]
    rng = np.random.default_rng(0); ds = []
    for _ in range(n):
        ix = np.concatenate([pools[j] for j in rng.integers(0, len(subs), len(subs))])
        ds.append((roc_auc_score(y[ix], a[ix])-roc_auc_score(y[ix], b[ix]),
                   balanced_accuracy_score(y[ix], a[ix] >= .5)-balanced_accuracy_score(y[ix], b[ix] >= .5)))
    d = np.asarray(ds)
    return {name: {'mean': float(np.mean(d[:, j])), 'p2.5': float(np.percentile(d[:, j], 2.5)),
                   'p97.5': float(np.percentile(d[:, j], 97.5))} for j, name in enumerate(('dAUC', 'dbACC'))}

def summarize(v, y, g, mask):
    return {'raw': metrics(v, y, g, mask), 'smooth5': metrics(smooth(v, g), y, g, mask)}

def main():
    start = time.perf_counter(); tee = Tee(sys.stdout, OUT/'context_history_extend.log'); sys.stdout = tee
    try:
        d = np.load(OUT/'highdim_deep_features.npz'); X = d['X'].astype(np.float64); y = d['y'].astype(int); g = d['groups']
        E = np.load(OUT/'emg_features_v2.npz')['E'].astype(np.float64)
        base = np.hstack([X[:, X.std(0) > 1e-8], E]); Z = subject_quantile(base, g, K0)
        pos = np.empty(len(y), int)
        for s in np.unique(g): pos[np.where(g == s)[0]] = np.arange(np.sum(g == s))
        mask = pos >= K0
        splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(Z, y, g))
        results = {'meta': {'data': ['highdim_deep_features.npz', 'emg_features_v2.npz'], 'warmup_windows': K0,
            'history_windows': list(HIST), 'strict_excludes_current': True, 'include_current_control': True,
            'folds': 5, 'split_seed': SEED, 'std_floor': .1, 'eval_n': int(mask.sum()),
            'combos': {k: list(v) for k, v in COMBOS.items()}, 'bootstrap_reps': 2000,
            'limitations': '历史长度和smooth5均超过6秒；依赖R/T交替；前120窗不训练不计分。'}, 'single_models': {}, 'combos': {}}
        oofs = {}; feature_cache = {}
        with threadpool_limits(limits=4):
            for include in (False, True):
                mode = 'include_current' if include else 'exclude_current'
                for k in HIST:
                    mu, sd = causal_mean_std(Z, g, k, not include)
                    name = f'd{k}_mean_std_{mode}'
                    F = (Z-mu)/np.maximum(sd, .1); feature_cache[name] = F
                    p = oof(F, y, g, mask, splits, name, start); oofs[name] = p
                    results['single_models'][name] = summarize(p, y, g, mask)
                    r = results['single_models'][name]['smooth5']
                    print(f'[result] {name} raw AUC={results["single_models"][name]["raw"]["AUC"]:.4f} bACC={results["single_models"][name]["raw"]["bACC"]:.4f}; smooth5 AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f} R->T={r["R->T"]:.4f} T->R={r["T->R"]:.4f}', flush=True)
            for mode in ('exclude_current', 'include_current'):
                for cname, ks in COMBOS.items():
                    key = f'{cname}_{mode}'; p = np.mean([oofs[f'd{k}_mean_std_{mode}'] for k in ks], axis=0)
                    oofs[key] = p; results['combos'][key] = {'members': list(ks), **summarize(p, y, g, mask)}
                    r = results['combos'][key]['smooth5']
                    print(f'[result] combo {key} raw AUC={results["combos"][key]["raw"]["AUC"]:.4f} bACC={results["combos"][key]["raw"]["bACC"]:.4f}; smooth5 AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f} R->T={r["R->T"]:.4f} T->R={r["T->R"]:.4f}', flush=True)
        strict = oofs['A_exclude_current']; z = logit(np.clip(strict, 1e-6, 1-1e-6))
        mu, sd = causal_mean_std(z[:, None], g, 200, True)
        zcal = expit((z-mu[:, 0])/np.maximum(sd[:, 0], .2)); oofs['score_z_calibration_exclude_current'] = zcal
        results['score_z_calibration'] = summarize(zcal, y, g, mask)
        print(f'[result] score_z_calibration smooth5 AUC={results["score_z_calibration"]["smooth5"]["AUC"]:.4f} bACC={results["score_z_calibration"]["smooth5"]["bACC"]:.4f}', flush=True)
        best_key = max((k for k in results['combos'] if k.endswith('exclude_current')), key=lambda k: results['combos'][k]['smooth5']['AUC'])
        best = smooth(oofs[best_key], g); old = smooth(oofs['A_include_current'], g)
        results['best_new'] = best_key; results['bootstrap_best_vs_old_baseline'] = bootstrap(best, old, y, g, mask, 2000)
        print(f'[bootstrap] {best_key} vs A_include_current 2000: {results["bootstrap_best_vs_old_baseline"]}', flush=True)
        np.savez(OUT/'context_history_extend_oof.npz', y=y, groups=g, eval_mask=mask, **oofs)
        results['seconds'] = time.perf_counter()-start; results['deviations'] = '无；按计划完成。'
        (OUT/'context_history_extend_results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'[done] seconds={results["seconds"]:.1f}', flush=True)
    finally:
        sys.stdout = tee; tee.close()

if __name__ == '__main__': main()
