"""Causal history model with current-window spectral and random-statistics features."""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from b_calib_variants import subject_quantile
from context_history_extend import Tee
from memory_stream_window6 import subject_median_threshold

FS, K0, Q, FLOOR, SEED = 250., 120, 10**6, .1, 0
SCALES, SMOOTHS = (50, 100, 200, 400, 800), (1, 5, 10, 20)
T0 = time.perf_counter()


def log(msg):
    print(f'[{time.perf_counter() - T0:7.1f}s] {msg}', flush=True)


def spectral_random(raw):
    """Features use only samples in the current 6-second window."""
    x = raw.astype(np.float64)
    f, P = welch(x, fs=FS, nperseg=500, noverlap=250, axis=1, detrend='constant')
    df = f[1] - f[0]
    bands = ((1, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 45), (52, 100))
    bp = np.stack([P[:, (f >= a) & (f < b)].sum(1) * df + 1e-30 for a, b in bands], 1)
    total = P[:, (f >= 1) & (f < 100)].sum(1) * df + 1e-30
    rel = bp / total[:, None]
    q = P[:, (f >= 1) & (f < 100)] + 1e-30
    q = q / q.sum(1, keepdims=True)
    ent = -(q * np.log(q)).sum(1) / np.log(q.shape[1])
    flat = np.exp(np.log(q).mean(1)) / q.mean(1)
    ff = f[(f >= 1) & (f < 100)]
    centroid = (q * ff).sum(1)
    svar = (q * (ff[None, :] - centroid[:, None]) ** 2).sum(1)
    sskew = (q * (ff[None, :] - centroid[:, None]) ** 3).sum(1) / np.maximum(svar, 1e-12) ** 1.5
    slx = np.log(ff); sly = np.log(P[:, (f >= 1) & (f < 100)] + 1e-30)
    slx0 = slx - slx.mean(); slope = (sly * slx0).sum(1) / (slx0 * slx0).sum()
    ratios = np.stack([bp[:, i] / bp[:, j] for i, j in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6))], 1)
    mean = x.mean(1); sd = x.std(1)
    ac = np.stack([np.sum((x[:, :-lag] - mean[:, None]) * (x[:, lag:] - mean[:, None]), 1) /
                   np.maximum(np.sum((x - mean[:, None]) ** 2, 1), 1e-30) for lag in (1, 2, 5, 10)], 1)
    zc = ((x[:, 1:] - mean[:, None]) * (x[:, :-1] - mean[:, None]) < 0).mean(1)
    cv = sd / np.maximum(np.abs(mean), 1e-6)
    sk = skew(x, axis=1, bias=False); ku = kurtosis(x, axis=1, fisher=True, bias=False)
    F = np.hstack([np.log(bp), rel, ent[:, None], flat[:, None], centroid[:, None], svar[:, None],
                   sskew[:, None], slope[:, None], np.log(ratios), ac, zc[:, None], cv[:, None],
                   sk[:, None], ku[:, None]])
    F[~np.isfinite(F)] = 0.
    return F.astype(np.float64)


def exact_mean_std(z, g, k):
    zi = np.round(z * Q).astype(np.int64)
    mean = np.zeros_like(z); std = np.zeros_like(z)
    for s in np.unique(g):
        ii = np.where(g == s)[0]; v = zi[ii]
        cs = np.vstack([np.zeros((1, v.shape[1]), np.int64), np.cumsum(v, 0)])
        cs2 = np.vstack([np.zeros((1, v.shape[1]), np.int64), np.cumsum(v * v, 0)])
        t = np.arange(len(ii)); lo = np.maximum(t - k, 0); n = (t - lo)[:, None]
        s1, s2 = cs[t] - cs[lo], cs2[t] - cs2[lo]
        var_num = n * s2 - s1 * s1; nn = np.maximum(n, 1).astype(float)
        mean[ii] = np.where(n > 0, s1 / nn / Q, 0.)
        std[ii] = np.where(n > 1, np.sqrt(np.maximum(var_num, 0).astype(float)) / nn / Q, 0.)
    return mean, std


def smooth(v, g, k):
    out = np.full_like(v, np.nan)
    for s in np.unique(g):
        ii = np.where(g == s)[0]; x = v[ii]
        for t in range(len(ii)):
            w = x[max(0, t-k+1):t+1]; w = w[np.isfinite(w)]
            if len(w): out[ii[t]] = w.mean()
    return out


def metrics(v, y, g, mask):
    out = {}
    for k in SMOOTHS:
        p = smooth(v, g, k); m = mask & np.isfinite(p)
        med = subject_median_threshold(p, g, mask)
        out[f's{k}'] = {'AUC': float(roc_auc_score(y[m], p[m])),
                        'bACC': float(balanced_accuracy_score(y[m], p[m] >= med[m]))}
    return out


def model():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_depth=3,
                                          l2_regularization=1., random_state=0)


def fuse(a, b, mode):
    if mode == 'score': return (a + b) / 2
    la = np.log(np.clip(a, 1e-6, 1-1e-6) / (1-np.clip(a, 1e-6, 1-1e-6)))
    lb = np.log(np.clip(b, 1e-6, 1-1e-6) / (1-np.clip(b, 1e-6, 1-1e-6)))
    return 1 / (1 + np.exp(-(la + lb) / 2))


def main():
    tee = Tee(sys.stdout, OUT / 'history_spectral_random.log'); sys.stdout = tee
    try:
        d = np.load(OUT / 'dense_window_protocol_cache.npz'); raw = np.load(OUT / 'dense_window_protocol_raw6.npy', mmap_mode='r')
        y0, g0 = d['label'].astype(int), d['subject']; c = np.where(d['canon'])[0]
        y, g = y0[c], g0[c]
        ref = np.load(OUT / 'highdim_deep_features.npz')
        assert np.array_equal(y, ref['y']) and np.array_equal(g, ref['groups'])
        log(f'canonical aligned n={len(c)} raw_shape={raw.shape}')
        F0 = spectral_random(raw[c]); log(f'current-window features dim={F0.shape[1]}')
        pos = np.empty(len(y), int)
        for s in np.unique(g):
            ii = np.where(g == s)[0]; pos[ii] = np.arange(len(ii))
        mask = pos >= K0
        splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(F0, y, g))
        old = np.load(OUT / 'history_leakfree_oof.npz'); assert np.array_equal(old['y'], y) and np.array_equal(old['groups'], g)
        old_scores = {'old_D_hgb': old['D_hgb'], 'old_best_fusion': old['D_hgb+single6_memz']}
        res = {'meta': {'feature_dim': int(F0.shape[1]), 'warmup': K0, 'scales': list(SCALES), 'smooths': list(SMOOTHS),
                        'quant': Q, 'std_floor': FLOOR, 'welch': {'fs': FS, 'nperseg': 500, 'noverlap': 250},
                        'history': 'exact integer rolling statistics over past windows only', 'canonical_aligned': True},
               'single': {}, 'fusion': {}}
        P = {}
        with threadpool_limits(limits=4):
            Z = subject_quantile(F0, g, K0)
            for k in SCALES:
                mu, sd = exact_mean_std(Z, g, k); F = np.round((Z - mu) / np.maximum(sd, FLOOR), 6)
                p = np.full(len(y), np.nan)
                for fold, (tr, te) in enumerate(splits, 1):
                    tr, te = tr[mask[tr]], te[mask[te]]
                    p[te] = model().fit(F[tr], y[tr]).predict_proba(F[te])[:, 1]
                    log(f'd{k} fold {fold}/5 ntr={len(tr)}')
                P[f'd{k}_hgb'] = p; res['single'][f'd{k}_hgb'] = metrics(p, y, g, mask)
                log(f'd{k}_hgb {res["single"][f"d{k}_hgb"]}')
        P['D_hgb'] = np.nanmean(np.stack([P[f'd{k}_hgb'] for k in SCALES]), 0)
        res['fusion']['D_hgb'] = metrics(P['D_hgb'], y, g, mask)
        for oname, op in old_scores.items():
            for mode in ('score', 'logit'):
                v = fuse(P['D_hgb'], op, mode); v[~(np.isfinite(P['D_hgb']) & np.isfinite(op))] = np.nan
                key = f'D_hgb+{oname}_{mode}'; P[key] = v; res['fusion'][key] = metrics(v, y, g, mask)
        for key, val in res['fusion'].items(): log(f'{key} {val}')
        np.savez_compressed(OUT / 'history_spectral_random_oof.npz', y=y, groups=g, eval_mask=mask, **P)
        res['seconds'] = time.perf_counter() - T0
        (OUT / 'history_spectral_random_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
        log(f'done seconds={res["seconds"]:.1f}')
    finally:
        sys.stdout = tee.stream; tee.close()


if __name__ == '__main__': main()
