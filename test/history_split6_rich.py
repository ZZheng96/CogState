"""比较 6x1 秒子窗的丰富聚合，并接入因果历史模型。"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.signal import welch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
FS, K0, Q, FLOOR = 250., 120, 10**6, .1
SCALES, SMOOTHS, SEED = (50, 100, 200, 400, 800), (1, 5, 10, 20), 0
T0 = time.perf_counter()
sys.path.insert(0, str(OUT))
from b_calib_variants import subject_quantile
from broadband_shape_window6 import compute_features
from memory_stream_window6 import subject_median_threshold


class Tee:
    def __init__(self, stream, path):
        self.stream = stream; self.file = path.open('w', encoding='utf-8')
    def write(self, text): self.stream.write(text); self.file.write(text); self.file.flush()
    def flush(self): self.stream.flush(); self.file.flush()
    def close(self): self.file.close()


def log(text): print(f'[{time.perf_counter() - T0:7.1f}s] {text}', flush=True)


def one(raw):
    L = raw.shape[1]
    f, P = welch(raw, fs=FS, nperseg=min(500, L), noverlap=min(250, L // 2), axis=1, detrend='constant')
    fam = compute_features(raw.astype(np.float64), f, P)
    X = np.hstack([fam[k][0] for k in fam])
    df = f[1] - f[0]
    bands = ((1, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 45), (52, 100))
    X = np.hstack([X, np.stack([np.log(P[:, (f >= a) & (f < b)].sum(1) * df + 1e-30) for a, b in bands], 1)])
    X[~np.isfinite(X)] = 0.
    return X.astype(np.float32)


def split_features(raw):
    parts = []
    for j in range(6):
        parts.append(one(raw[:, j * 250:(j + 1) * 250]))
        log(f'feature part {j + 1}/6')
    return np.stack(parts, 1).astype(np.float64)


def aggregate(A):
    """Return the four requested within-window aggregations."""
    names = ('mean_std', 'mean_std_min_max', 'mean_std_p10_p90', 'mean_std_slope_diff')
    mean, sd = A.mean(1), A.std(1)
    out = {
        names[0]: np.hstack([mean, sd]),
        names[1]: np.hstack([mean, sd, A.min(1), A.max(1)]),
        names[2]: np.hstack([mean, sd, np.percentile(A, 10, axis=1), np.percentile(A, 90, axis=1)]),
    }
    t = np.arange(6, dtype=np.float64); tc = t - t.mean()
    slope = np.einsum('t,ntd->nd', tc, A - mean[:, None, :]) / np.sum(tc * tc)
    diff = np.diff(A, axis=1)
    out[names[3]] = np.hstack([mean, sd, slope, diff.mean(1), diff.std(1)])
    return out


def exact(z, g, k):
    zi = np.round(z * Q).astype(np.int64)
    mean, sd = np.zeros_like(z), np.zeros_like(z)
    for s in np.unique(g):
        ii = np.where(g == s)[0]; v = zi[ii]
        cs = np.vstack([np.zeros((1, v.shape[1]), np.int64), np.cumsum(v, 0)])
        cs2 = np.vstack([np.zeros((1, v.shape[1]), np.int64), np.cumsum(v * v, 0)])
        t = np.arange(len(ii)); lo = np.maximum(t - k, 0); n = (t - lo)[:, None]
        S1, S2 = cs[t] - cs[lo], cs2[t] - cs2[lo]
        var = n * S2 - S1 * S1; nn = np.maximum(n, 1).astype(float)
        mean[ii] = np.where(n > 0, S1 / nn / Q, 0.)
        sd[ii] = np.where(n > 1, np.sqrt(np.maximum(var, 0).astype(float)) / nn / Q, 0.)
    return mean, sd


def smooth(v, g, k):
    out = np.full_like(v, np.nan)
    for s in np.unique(g):
        ii = np.where(g == s)[0]; x = v[ii]
        for t in range(len(ii)):
            w = x[max(0, t - k + 1):t + 1]; w = w[np.isfinite(w)]
            if len(w): out[ii[t]] = w.mean()
    return out


def metrics(v, y, g, mask):
    out = {}
    for k in SMOOTHS:
        p = smooth(v, g, k); m = mask & np.isfinite(p)
        med = subject_median_threshold(p, g, mask)
        out[f's{k}'] = {'AUC': float(roc_auc_score(y[m], p[m])),
                        'bACC_med': float(balanced_accuracy_score(y[m], p[m] >= med[m]))}
    return out


def model():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_depth=3,
                                          l2_regularization=1., random_state=0)


def main():
    tee = Tee(sys.stdout, OUT / 'history_split6_rich.log'); sys.stdout = tee
    try:
        d = np.load(OUT / 'dense_window_protocol_cache.npz')
        raw = np.load(OUT / 'dense_window_protocol_raw6.npy', mmap_mode='r')
        y, g = d['label'].astype(int), d['subject']; c = np.where(d['canon'])[0]
        yc, gc = y[c], g[c]
        ref = np.load(OUT / 'highdim_deep_features.npz')
        assert np.array_equal(yc, ref['y']) and np.array_equal(gc, ref['groups'])
        pos = np.empty(len(y), int)
        for s in np.unique(g):
            ii = np.where(g == s)[0]; pos[ii] = np.arange(len(ii))
        mask = pos[c] >= K0
        log('extracting dense 6x1s features')
        A = split_features(raw)
        aggs = aggregate(A)
        splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(np.zeros((len(yc), 1)), yc, gc))
        old = np.load(OUT / 'history_leakfree_oof.npz')
        assert np.array_equal(old['y'], yc) and np.array_equal(old['groups'], gc)
        oldD = old['D_hgb']
        res = {'meta': {'aggregations': {k: int(v.shape[1]) for k, v in aggs.items()}, 'warmup': K0,
                        'scales': list(SCALES), 'smooths': list(SMOOTHS), 'quant': Q,
                        'exact_integer_rolling': True, 'strict_subject_isolation': True,
                        'history': 'causal past windows only'}, 'variants': {}}
        all_p = {}
        with threadpool_limits(limits=4):
            for name, F0 in aggs.items():
                log(f'calibrating {name} dim={F0.shape[1]}')
                Z = subject_quantile(F0, g, K0)[c]
                for k in SCALES:
                    mu, sd = exact(Z, gc, k)
                    F = np.round((Z - mu) / np.maximum(sd, FLOOR), 6)
                    p = np.full(len(yc), np.nan)
                    for fold, (tr, te) in enumerate(splits, 1):
                        tr, te = tr[mask[tr]], te[mask[te]]
                        p[te] = model().fit(F[tr], yc[tr]).predict_proba(F[te])[:, 1]
                        log(f'{name} d{k} fold {fold}/5 ntr={len(tr)}')
                    all_p[f'{name}__d{k}'] = p
                D = np.nanmean(np.stack([all_p[f'{name}__d{k}'] for k in SCALES]), axis=0)
                all_p[f'{name}__D'] = D
                res['variants'][name] = {'D': metrics(D, yc, gc, mask)}
                log(f'{name} newD {res["variants"][name]["D"]}')
                dlogit = np.log(np.clip(D, 1e-6, 1 - 1e-6) / (1 - np.clip(D, 1e-6, 1 - 1e-6)))
                ologit = np.log(np.clip(oldD, 1e-6, 1 - 1e-6) / (1 - np.clip(oldD, 1e-6, 1 - 1e-6)))
                for fusion, v in (('score', (D + oldD) / 2),
                                  ('logit', 1 / (1 + np.exp(-(dlogit + ologit) / 2)))):
                    valid = np.isfinite(D) & np.isfinite(oldD)
                    v[~valid] = np.nan
                    key = f'{name}__fusion_{fusion}'; all_p[key] = v
                    res['variants'][name][f'fusion_{fusion}'] = metrics(v, yc, gc, mask)
                    log(f'{key} {res["variants"][name][f"fusion_{fusion}"]}')
        res['seconds'] = time.perf_counter() - T0
        np.savez_compressed(OUT / 'history_split6_rich_oof.npz', y=yc, groups=gc, eval_mask=mask, **all_p)
        (OUT / 'history_split6_rich_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
        log(f'done {res["seconds"]:.0f}s')
    finally:
        sys.stdout = tee.stream; tee.close()


if __name__ == '__main__': main()
