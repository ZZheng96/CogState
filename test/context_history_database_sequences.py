"""Database historical-feature sensitivity to within-subject sequence order."""
import json, sys, time
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT, DB = ROOT / 'test', ROOT / 'database'
LOG, RES, OOF = OUT / 'context_history_database_sequences.log', OUT / 'context_history_database_sequences_results.json', OUT / 'context_history_database_sequences_oof.npz'
K0, SEED, NBOOT = 120, 0, 2000
D = (50, 100, 200, 400, 800)
T0 = time.perf_counter()

class Tee:
    def __init__(self, path): self.f = path.open('w', encoding='utf-8')
    def write(self, x): sys.__stdout__.write(x); sys.__stdout__.flush(); self.f.write(x); self.f.flush()
    def flush(self): sys.__stdout__.flush(); self.f.flush()
    def close(self): self.f.close()

def log(x): print(f'[{time.perf_counter()-T0:8.1f}s] {x}', flush=True)

def positions(g):
    p = np.empty(len(g), int)
    for s in np.unique(g):
        ii = np.flatnonzero(g == s); p[ii] = np.arange(len(ii))
    return p

def causal(z, g, k):
    mu = np.empty_like(z); sd = np.empty_like(z)
    for s in np.unique(g):
        ii = np.flatnonzero(g == s); v = z[ii]; t = np.arange(len(ii)); lo = np.maximum(t-k, 0); n = t-lo
        cs = np.vstack([np.zeros((1, v.shape[1])), np.cumsum(v, 0)])
        cs2 = np.vstack([np.zeros((1, v.shape[1])), np.cumsum(v*v, 0)])
        m = np.divide(cs[t]-cs[lo], n[:, None], out=np.zeros_like(v), where=n[:, None] > 0)
        q = np.divide(cs2[t]-cs2[lo], n[:, None], out=np.zeros_like(v), where=n[:, None] > 0)
        mu[ii] = m; sd[ii] = np.sqrt(np.maximum(q-m*m, 0))
    return mu, sd

def smooth(v, g, k=5):
    out = np.full(len(v), np.nan)
    for s in np.unique(g):
        ii = np.flatnonzero(g == s); x = v[ii]; good = np.isfinite(x)
        cs = np.cumsum(np.r_[0., np.where(good, x, 0.)]); ng = np.cumsum(np.r_[0, good.astype(int)])
        t = np.arange(len(ii)); lo = np.maximum(t-k+1, 0); n = ng[t+1]-ng[lo]
        out[ii] = np.divide(cs[t+1]-cs[lo], n, out=np.full(len(ii), np.nan), where=n > 0)
    return out

def metrics(v, y, g, mask):
    m = mask & np.isfinite(v); r = {'n': int(m.sum()), 'AUC': float(roc_auc_score(y[m], v[m])), 'bACC': float(balanced_accuracy_score(y[m], v[m] >= .5))}
    vals = []
    for s in np.unique(g):
        ii = m & (g == s)
        if len(np.unique(y[ii])) == 2: vals.append(float(roc_auc_score(y[ii], v[ii])))
    r['subject_AUC_median'] = float(np.median(vals)); r['subject_AUC_lt_0.5_frac'] = float(np.mean(np.asarray(vals) < .5)); r['n_subject_auc'] = len(vals)
    return r

def direction(v, y, g, mask):
    out = {'R->T': [], 'T->R': []}
    for s in np.unique(g):
        ii = np.flatnonzero(g == s); cuts = np.flatnonzero(np.r_[True, np.diff(y[ii]) != 0, True])
        blocks = [ii[a:b] for a, b in zip(cuts[:-1], cuts[1:])]
        for a, b in zip(blocks[:-1], blocks[1:]):
            z = np.r_[a, b]; z = z[mask[z] & np.isfinite(v[z])]
            if len(z) and len(np.unique(y[z])) == 2:
                out['R->T' if y[a[0]] == 0 else 'T->R'].append(roc_auc_score(y[z], v[z]))
    return {k: (float(np.mean(x)) if x else None) for k, x in out.items()}

def model():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_depth=3, l2_regularization=1., random_state=0)

def run_oof(F, y, g, mask, splits, tag):
    p = np.full(len(y), np.nan)
    for fold, (tr, te) in enumerate(splits, 1):
        tr, ev = tr[mask[tr]], te[mask[te]]; tic = time.perf_counter()
        m = model().fit(F[tr], y[tr]); p[ev] = m.predict_proba(F[ev])[:, 1]
        log(f'{tag} fold={fold}/5 train={len(tr)} test={len(ev)} elapsed={time.perf_counter()-tic:.1f}s')
    return p

def arrangement_orders(y, g):
    rng = np.random.default_rng(0); out = {}
    for name in ('RT_blocks', 'TR_blocks', 'RT_alternating_blocks_40', 'RT_alternating_blocks_200', 'RT_alternating_blocks_800', 'label_random', 'within_label_random'):
        pieces = []
        for s in np.unique(g):
            ii = np.flatnonzero(g == s); r, t = ii[y[ii] == 0], ii[y[ii] == 1]
            if name == 'RT_blocks': q = np.r_[r, t]
            elif name == 'TR_blocks': q = np.r_[t, r]
            elif name.startswith('RT_alternating_blocks_'):
                k = int(name.rsplit('_', 1)[1]); q = []; a = b = 0
                while a < len(r) or b < len(t):
                    q.extend(r[a:a+k]); a += k; q.extend(t[b:b+k]); b += k
                q = np.asarray(q)
            elif name == 'label_random': q = rng.permutation(ii)
            else: q = np.r_[rng.permutation(r), rng.permutation(t)]
            pieces.append(q)
        out[name] = np.concatenate(pieces)
    return out

def sequence_stats(y, g):
    result = {}
    for s in np.unique(g):
        q = y[g == s]; runs = np.flatnonzero(np.r_[True, np.diff(q) != 0]); ends = np.r_[runs[1:], len(q)]
        result[str(s)] = {'first': int(q[0]), 'last': int(q[-1]), 'runs': int(len(runs)), 'longest_run': int(np.max(ends-runs))}
    vals = list(result.values())
    return {'first_labels': sorted({v['first'] for v in vals}), 'last_labels': sorted({v['last'] for v in vals}), 'runs_min': int(min(v['runs'] for v in vals)), 'runs_median': float(np.median([v['runs'] for v in vals])), 'runs_max': int(max(v['runs'] for v in vals)), 'longest_run_min': int(min(v['longest_run'] for v in vals)), 'longest_run_median': float(np.median([v['longest_run'] for v in vals])), 'longest_run_max': int(max(v['longest_run'] for v in vals)), 'per_subject': result}

def bootstrap(a, b, y, g, mask):
    subs = np.unique(g); pools = [np.flatnonzero((g == s) & mask) for s in subs]; rng = np.random.default_rng(0); da, db = [], []
    for i in range(NBOOT):
        ix = np.concatenate([pools[j] for j in rng.integers(0, len(subs), len(subs))])
        da.append(roc_auc_score(y[ix], a[ix])-roc_auc_score(y[ix], b[ix]))
        db.append(balanced_accuracy_score(y[ix], a[ix] >= .5)-balanced_accuracy_score(y[ix], b[ix] >= .5))
        if (i+1) % 200 == 0: log(f'bootstrap {i+1}/{NBOOT}')
    return {'dAUC': {'point': float(roc_auc_score(y[mask], a[mask])-roc_auc_score(y[mask], b[mask])), 'ci95': [float(np.percentile(da, 2.5)), float(np.percentile(da, 97.5))]}, 'dbACC': {'point': float(balanced_accuracy_score(y[mask], a[mask] >= .5)-balanced_accuracy_score(y[mask], b[mask] >= .5)), 'ci95': [float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))]}}

def main():
    tee = Tee(LOG); sys.stdout = tee
    try:
        if RES.exists() or OOF.exists(): raise FileExistsError('refusing to overwrite sequence artifacts')
        fs = []; ys = []; gs = []
        for p in sorted(DB.glob('windows/*/*.npz')):
            q = np.load(p); fs.append(np.hstack([q['f'], q['e']])); ys.append(q['label']); gs.append(q['subject'])
        F0 = np.vstack(fs).astype(np.float64); y0 = np.concatenate(ys).astype(int); g0 = np.concatenate(gs).astype(str)
        keep = np.std(F0, 0) > 1e-8; F0 = F0[:, keep]
        sys.path.insert(0, str(OUT)); from b_calib_variants import subject_quantile
        orders = arrangement_orders(y0, g0); results = {'meta': {'data': 'database/windows/{relax,task}/*.npz f+e', 'n_windows': len(y0), 'n_subjects': len(np.unique(g0)), 'per_subject_counts': {'R': 8012, 'T': 8055}, 'warmup_windows': K0, 'history_excludes_current': True, 'folds': 'StratifiedGroupKFold(5, shuffle=True, random_state=0)', 'D': list(D), 'smooth_window': 5, 'bootstrap_subject_reps': NBOOT}, 'arrangements': {}}
        log(f'loaded n={len(y0)} subjects={len(np.unique(g0))} features={F0.shape[1]}')
        all_oof = {}
        with threadpool_limits(limits=4):
            for name, perm in orders.items():
                F, y, g = F0[perm], y0[perm], g0[perm]; mask = positions(g) >= K0
                Z = subject_quantile(F, g, K0); splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Z, y, g)); probs = []
                for k in D:
                    mu, sd = causal(Z, g, k); probs.append(run_oof((Z-mu)/np.maximum(sd, .1), y, g, mask, splits, f'{name}/D{k}'))
                p = np.nanmean(probs, axis=0); sm = smooth(p, g, 5)
                results['arrangements'][name] = {'sequence': sequence_stats(y, g), 'window': metrics(p, y, g, mask), 'smooth5': metrics(sm, y, g, mask), 'direction_smooth5': direction(sm, y, g, mask)}
                all_oof[name] = {'y': y, 'groups': g, 'eval_mask': mask, 'D_prob': p, 'D_smooth5': sm}
                RES.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8'); log(f'{name} complete window AUC={results["arrangements"][name]["window"]["AUC"]:.4f} smooth5 AUC={results["arrangements"][name]["smooth5"]["AUC"]:.4f}')
        a, b = all_oof['RT_blocks'], all_oof['label_random']; results['RT_blocks_vs_label_random_bootstrap'] = bootstrap(a['D_smooth5'], b['D_smooth5'], a['y'], a['groups'], a['eval_mask'])
        save = {}
        for name, q in all_oof.items():
            for key, val in q.items(): save[f'{name}__{key}'] = val
        np.savez_compressed(OOF, **save); results['elapsed_s'] = time.perf_counter()-T0; RES.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8'); log(f'done elapsed={results["elapsed_s"]:.1f}s')
    finally:
        sys.stdout = tee; tee.close()

if __name__ == '__main__': main()
