"""严格因果历史协议下的 6x1 秒低维谱动态实验。"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import welch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from b_calib_variants import subject_quantile
from context_history_extend import Tee

FS, K0, Q, FLOOR, SEED = 250., 120, 10**6, .1, 0
SCALES, SMOOTHS = (50, 100, 200, 400, 800), (1, 5, 10, 20)
BANDS = ((1, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 45), (52, 100))
AGGREGATIONS = ('mean_std', 'mean_std_p10_p90_slope_diff')
FUSION_MODES = ('score', 'logit')
MIN_HISTORY = 20
T0 = time.perf_counter()


def log(msg):
    print(f'[elapsed {time.perf_counter() - T0:7.1f}s] {msg}', flush=True)


def subwindow_spectrum(x):
    """One-second window -> 7 log powers + 7 relative powers + entropy/flatness."""
    f, p = welch(x.astype(np.float64), fs=FS, nperseg=250, noverlap=125,
                 axis=1, detrend='constant')
    df = f[1] - f[0]
    bp = np.stack([p[:, (f >= a) & (f < b)].sum(1) * df + 1e-30
                   for a, b in BANDS], axis=1)
    broad = (f >= 1) & (f < 100)
    total = p[:, broad].sum(1) * df + 1e-30
    rel = bp / total[:, None]
    q = p[:, broad] + 1e-30
    q /= q.sum(1, keepdims=True)
    entropy = -(q * np.log(q)).sum(1) / np.log(q.shape[1])
    flatness = np.exp(np.log(q).mean(1)) / q.mean(1)
    features = np.hstack([np.log(bp), rel, entropy[:, None], flatness[:, None]])
    features[~np.isfinite(features)] = 0.
    return features


def spectral_dynamics(raw, canonical):
    parts = []
    for j in range(6):
        x = np.asarray(raw[canonical, j * 250:(j + 1) * 250])
        parts.append(subwindow_spectrum(x))
        log(f'Welch subwindow {j + 1}/6 dim={parts[-1].shape[1]}')
    return np.stack(parts, axis=1)


def aggregate(a):
    """Only the two preregistered aggregations are constructed."""
    mean, std = a.mean(1), a.std(1)
    t = np.arange(6, dtype=np.float64)
    tc = t - t.mean()
    slope = np.einsum('t,ntd->nd', tc, a - mean[:, None, :]) / np.sum(tc * tc)
    diff = np.diff(a, axis=1)
    return {
        'mean_std': np.hstack([mean, std]),
        'mean_std_p10_p90_slope_diff': np.hstack([
            mean, std, np.percentile(a, 10, axis=1), np.percentile(a, 90, axis=1),
            slope, diff.mean(1), np.abs(diff).mean(1)]),
    }


def exact_mean_std(z, groups, k):
    """Past k windows only; integer accumulation prevents floating-point residuals."""
    zi = np.round(z * Q).astype(np.int64)
    mean, std = np.zeros_like(z), np.zeros_like(z)
    for subject in np.unique(groups):
        ii = np.where(groups == subject)[0]
        v = zi[ii]
        cs = np.vstack([np.zeros((1, v.shape[1]), np.int64), np.cumsum(v, 0)])
        cs2 = np.vstack([np.zeros((1, v.shape[1]), np.int64), np.cumsum(v * v, 0)])
        t = np.arange(len(ii))
        lo = np.maximum(t - k, 0)
        n = (t - lo)[:, None]
        s1, s2 = cs[t] - cs[lo], cs2[t] - cs2[lo]
        var_num = n * s2 - s1 * s1
        nn = np.maximum(n, 1).astype(float)
        mean[ii] = np.where(n > 0, s1 / nn / Q, 0.)
        std[ii] = np.where(n > 1,
                           np.sqrt(np.maximum(var_num, 0).astype(float)) / nn / Q, 0.)
    return mean, std


def causal_smooth(v, groups, k):
    out = np.full_like(v, np.nan)
    for subject in np.unique(groups):
        ii = np.where(groups == subject)[0]
        x = v[ii]
        for t in range(len(ii)):
            history = x[max(0, t - k + 1):t + 1]
            history = history[np.isfinite(history)]
            if len(history):
                out[ii[t]] = history.mean()
    return out


def causal_median_threshold(v, groups):
    """Median of prior scores only; no current/future score and no labels."""
    out = np.full(len(v), .5)
    for subject in np.unique(groups):
        ii = np.where(groups == subject)[0]
        x = v[ii]
        for t, i in enumerate(ii):
            history = x[:t]
            history = history[np.isfinite(history)]
            if len(history) >= MIN_HISTORY:
                out[i] = np.median(history)
    return out


def classification_metrics(y, groups, mask, score, pred):
    m = mask & np.isfinite(score)
    micro = float(balanced_accuracy_score(y[m], pred[m]))
    per_subject = []
    for subject in np.unique(groups[m]):
        q = m & (groups == subject)
        if np.unique(y[q]).size == 2:
            per_subject.append(balanced_accuracy_score(y[q], pred[q]))
    return {'subject_macro_bACC': float(np.mean(per_subject)),
            'window_micro_bACC': micro}


def evaluate(v, y, groups, mask):
    out = {}
    for k in SMOOTHS:
        score = causal_smooth(v, groups, k)
        m = mask & np.isfinite(score)
        median = causal_median_threshold(score, groups)
        out[f's{k}'] = {
            'AUC': float(roc_auc_score(y[m], score[m])),
            'fixed_0.5': classification_metrics(y, groups, mask, score, score >= .5),
            'causal_median': classification_metrics(y, groups, mask, score, score >= median),
        }
    return out


def model():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_depth=3,
                                          l2_regularization=1., random_state=SEED)


def fuse(a, b, mode):
    if mode == 'score':
        out = (a + b) / 2
    else:
        ac = np.clip(a, 1e-6, 1 - 1e-6)
        bc = np.clip(b, 1e-6, 1 - 1e-6)
        la = np.log(ac / (1 - ac))
        lb = np.log(bc / (1 - bc))
        out = 1 / (1 + np.exp(-(la + lb) / 2))
    out[~(np.isfinite(a) & np.isfinite(b))] = np.nan
    return out


def compact(metrics):
    return ' '.join(
        f'{s}:AUC={r["AUC"]:.4f}/macro05={r["fixed_0.5"]["subject_macro_bACC"]:.4f}'
        f'/micro05={r["fixed_0.5"]["window_micro_bACC"]:.4f}'
        f'/macroMed={r["causal_median"]["subject_macro_bACC"]:.4f}'
        f'/microMed={r["causal_median"]["window_micro_bACC"]:.4f}'
        for s, r in metrics.items())


def main():
    tee = Tee(sys.stdout, OUT / 'history_spectral_dynamics.log')
    sys.stdout = tee
    try:
        cache = np.load(OUT / 'dense_window_protocol_cache.npz')
        raw = np.load(OUT / 'dense_window_protocol_raw6.npy', mmap_mode='r')
        assert raw.shape[1] == 1500
        canonical = np.where(cache['canon'])[0]
        y = cache['label'][canonical].astype(int)
        groups = cache['subject'][canonical]
        ref = np.load(OUT / 'highdim_deep_features.npz')
        assert np.array_equal(y, ref['y']) and np.array_equal(groups, ref['groups'])
        assert np.array_equal(np.asarray(raw[canonical]), ref['raw'])
        log(f'canonical aligned n={len(y)} subjects={len(np.unique(groups))} raw={raw.shape}')

        a = spectral_dynamics(raw, canonical)
        aggregations = aggregate(a)
        assert tuple(aggregations) == AGGREGATIONS
        log('preregistered aggregations ' +
            ', '.join(f'{k}={v.shape[1]}' for k, v in aggregations.items()))

        pos = np.empty(len(y), int)
        for subject in np.unique(groups):
            ii = np.where(groups == subject)[0]
            pos[ii] = np.arange(len(ii))
        mask = pos >= K0
        splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(
            np.zeros((len(y), 1)), y, groups))

        old = np.load(OUT / 'history_leakfree_oof.npz')
        assert np.array_equal(old['y'], y) and np.array_equal(old['groups'], groups)
        assert np.array_equal(old['eval_mask'], mask)
        sources = {'old_best': old['D_hgb+single6_memz'], 'old_D_hgb': old['D_hgb']}
        split_path = OUT / 'history_split6_rich_oof.npz'
        if split_path.exists():
            split = np.load(split_path)
            assert np.array_equal(split['y'], y) and np.array_equal(split['groups'], groups)
            # split6 used the full-cache position for warmup, a superset of this canonical mask.
            assert np.all(~mask | split['eval_mask'])
            # Fixed before this run from the prior split6 experiment; no new search.
            sources['split6_best'] = split['mean_std_p10_p90__fusion_score']

        res = {
            'meta': {
                'feature_per_subwindow': int(a.shape[2]),
                'subwindow_feature_names': ['log_power_7', 'relative_power_7',
                                            'spectral_entropy', 'spectral_flatness'],
                'aggregation_dims': {k: int(v.shape[1]) for k, v in aggregations.items()},
                'preregistered_aggregations': list(AGGREGATIONS),
                'welch': {'fs': FS, 'nperseg': 250, 'noverlap': 125,
                          'bands': BANDS, 'broadband': [1, 100]},
                'warmup': K0, 'scales': list(SCALES), 'smooths': list(SMOOTHS),
                'quant': Q, 'std_floor': FLOOR, 'folds': 5, 'seed': SEED,
                'history': 'exact integer rolling statistics over past canonical windows only',
                'strict_subject_isolation': True, 'canonical_raw_aligned': True,
                'thresholds': ['fixed_0.5', 'causal_past_score_median'],
                'median_min_history': MIN_HISTORY,
                'fusion': 'fixed equal-weight score/logit; no OOF weight or threshold tuning',
                'fusion_sources': list(sources),
                'split6_score': 'mean_std_p10_p90__fusion_score',
                'evaluation_mask': 'canonical warmup mask used for every source and fusion',
                'old_best_note': 'real-valued cached score; fixed 0.5 is applied uniformly as preregistered',
            },
            'baselines': {}, 'variants': {},
        }
        saved = {'y': y, 'groups': groups, 'eval_mask': mask}
        for name, score in sources.items():
            saved[name] = score
            res['baselines'][name] = evaluate(score, y, groups, mask)
            log(f'baseline {name} {compact(res["baselines"][name])}')

        with threadpool_limits(limits=4):
            for name in AGGREGATIONS:
                f0 = aggregations[name]
                block = {'feature_dim': int(f0.shape[1]), 'scales': {}, 'fusion': {}}
                res['variants'][name] = block
                log(f'calibrating {name} dim={f0.shape[1]}')
                z = subject_quantile(f0, groups, K0)
                for scale in SCALES:
                    mu, sd = exact_mean_std(z, groups, scale)
                    features = np.round((z - mu) / np.maximum(sd, FLOOR), 6)
                    p = np.full(len(y), np.nan)
                    for fold, (tr, te) in enumerate(splits, 1):
                        tr, te = tr[mask[tr]], te[mask[te]]
                        p[te] = model().fit(features[tr], y[tr]).predict_proba(features[te])[:, 1]
                        log(f'{name} d{scale} fold {fold}/5 ntr={len(tr)} nte={len(te)}')
                    assert np.isfinite(p[mask]).all()
                    key = f'{name}__d{scale}_hgb'
                    saved[key] = p
                    block['scales'][f'd{scale}_hgb'] = evaluate(p, y, groups, mask)
                    log(f'{key} {compact(block["scales"][f"d{scale}_hgb"])}')
                d = np.mean(np.stack([saved[f'{name}__d{k}_hgb'] for k in SCALES]), axis=0)
                saved[f'{name}__D_hgb'] = d
                block['D_hgb'] = evaluate(d, y, groups, mask)
                log(f'{name}__D_hgb {compact(block["D_hgb"])}')
                for source_name, source in sources.items():
                    for mode in FUSION_MODES:
                        key = f'{name}__D_hgb+{source_name}_{mode}'
                        score = fuse(d, source, mode)
                        saved[key] = score
                        block['fusion'][f'D_hgb+{source_name}_{mode}'] = evaluate(
                            score, y, groups, mask)
                        log(f'{key} {compact(block["fusion"][f"D_hgb+{source_name}_{mode}"])}')

        np.savez_compressed(OUT / 'history_spectral_dynamics_oof.npz', **saved)
        res['seconds'] = time.perf_counter() - T0
        (OUT / 'history_spectral_dynamics_results.json').write_text(
            json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
        log(f'done seconds={res["seconds"]:.1f}')
    finally:
        sys.stdout = tee.stream
        tee.close()


if __name__ == '__main__':
    main()
