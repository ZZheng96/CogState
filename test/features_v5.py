"""A 支 v5：当前窗内的高分辨率对数谱与时频调制，不使用历史或顺序信息。"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.signal import welch, butter, sosfilt, hilbert, windows
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]; DB = ROOT/'database'; OUT = ROOT/'test'
sys.path[:0] = [str(OUT)]
from highdim_deep_ablation import FeatureTransform

FS = 250.
BANDS = ((1, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 45))
MOD_BANDS = ((.2, .5), (.5, 1), (1, 2), (2, 4), (4, 8))


def load():
    fs, es, xs, ys, gs = [], [], [], [], []
    for p in sorted((DB/'windows').glob('*/*.npz')):
        d = np.load(p)
        fs.append(d['f']); es.append(d['e']); xs.append(d['x']); ys.append(d['label']); gs.append(d['subject'])
    return (np.vstack(fs).astype(np.float64), np.vstack(es).astype(np.float64),
            np.vstack(xs).astype(np.float64), np.concatenate(ys).astype(int), np.concatenate(gs))


def names_and_slices(freq_w, freq_m):
    names = ([f'welch_log_{f:.1f}' for f in freq_w] +
             [f'multitaper_log_{f:.3f}' for f in freq_m])
    slices = {'welch': slice(0, len(freq_w)), 'multitaper': slice(len(freq_w), len(freq_w)+len(freq_m))}
    start = len(names)
    for bi, (lo, hi) in enumerate(BANDS):
        names.extend(f'env{bi}_logpower_t{j}' for j in range(12))
        slices[f'env{bi}_trajectory'] = slice(start, start+12); start += 12
        names.extend(f'env{bi}_mod_{mlo}_{mhi}' for mlo, mhi in MOD_BANDS)
        slices[f'env{bi}_modulation'] = slice(start, start+5); start += 5
    slices['spectral'] = slice(0, len(freq_w)+len(freq_m))
    slices['modulation'] = slice(len(freq_w)+len(freq_m), len(names))
    return names, slices


def extract(raw):
    """分批计算，避免多 taper 复谱和 Hilbert 中间数组占用过多内存。"""
    n = len(raw)
    fw = np.arange(0, FS/2+.01, .5)
    fm = np.fft.rfftfreq(raw.shape[1], 1/FS)
    wkeep = (fw >= 1) & (fw <= 45)
    mkeep = (fm >= 1) & (fm <= 45)
    names, slices = names_and_slices(fw[wkeep], fm[mkeep])
    out = np.empty((n, len(names)), dtype=np.float32)
    tap = windows.dpss(raw.shape[1], 3, Kmax=5)
    win = windows.hann(500, sym=False)
    sos = [butter(3, [lo, hi], fs=FS, btype='bandpass', output='sos') for lo, hi in BANDS]
    tick = time.perf_counter()
    for first in range(0, n, 128):
        x = raw[first:first+128]
        # 2 秒 Welch：0.5 Hz 分辨率，5 个重叠段平均。
        seg = np.lib.stride_tricks.sliding_window_view(x, 500, axis=1)[:, ::250, :]
        seg = (seg-seg.mean(axis=-1, keepdims=True))*win
        pw = np.mean(np.abs(np.fft.rfft(seg, axis=-1))**2, axis=1)
        # 全 6 秒多 taper：保留峰形，同时降低单一周期图方差。
        xc = (x-x.mean(1, keepdims=True))*win.mean()  # 仅为统一数值尺度，标签不可见
        pm = np.mean([np.abs(np.fft.rfft(xc*t, axis=1))**2 for t in tap], axis=0)
        row = [np.log(pw[:, wkeep]+1e-30), np.log(pm[:, mkeep]+1e-30)]
        for filt in sos:
            env = np.abs(hilbert(sosfilt(filt, x, axis=1), axis=1))
            ep = np.log(np.mean(env.reshape(len(x), 12, 125)**2, axis=2)+1e-30)
            # 包络降采样至 25 Hz，再做调制谱；仅描述当前窗内部动态。
            e25 = env.reshape(len(x), 150, 10).mean(2)
            e25 = e25-e25.mean(1, keepdims=True)
            mf = np.fft.rfftfreq(64, 1/25)
            mp = np.abs(np.fft.rfft(e25[:, :128].reshape(len(x), 2, 64), axis=2))**2
            mp = mp.mean(1)
            mods = [np.log(mp[:, (mf >= lo) & (mf < hi)].mean(1)+1e-30) for lo, hi in MOD_BANDS]
            row.extend([ep, np.column_stack(mods)])
        out[first:first+len(x)] = np.hstack(row)
        if first % 1280 == 0 or first+len(x) == n:
            print(f'[extract] {first+len(x)}/{n} elapsed={time.perf_counter()-tick:.0f}s', flush=True)
    return out, np.array(names), slices


def report(y, s, g):
    r = {'AUC': float(roc_auc_score(y, s)), 'bACC': float(balanced_accuracy_score(y, s >= .5))}
    per = [{'AUC': roc_auc_score(y[g == k], s[g == k]), 'bACC': balanced_accuracy_score(y[g == k], s[g == k] >= .5)}
           for k in np.unique(g) if len(set(y[g == k])) == 2]
    r.update({f'subject_{m}': float(np.mean([d[m] for d in per])) for m in ('AUC', 'bACC')})
    return r


def main():
    tick = time.perf_counter()
    F, E, raw, y, g = load()
    cache = OUT/'features_v5.npz'
    if cache.exists():
        d = np.load(cache); V, names = d['V'], d['names']
        fw = np.arange(0, FS/2+.01, .5); fw = fw[(fw >= 1) & (fw <= 45)]
        fm = np.fft.rfftfreq(raw.shape[1], 1/FS); fm = fm[(fm >= 1) & (fm <= 45)]
        _, slices = names_and_slices(fw, fm)
    else:
        V, names, slices = extract(raw)
        np.savez_compressed(cache, V=V, names=names)
    V = np.nan_to_num(V.astype(np.float64), nan=0., posinf=0., neginf=0.)
    print(f'[data] v5={V.shape} families={list(slices)}', flush=True)
    single = sorted([(roc_auc_score(y, V[:, i]), str(n)) for i, n in enumerate(names)], key=lambda z: -abs(z[0]-.5))
    for a, n in single[:12]:
        print(f'[single] {n:30s} AUC={a:.4f}', flush=True)

    V3 = np.load(OUT/'features_v3.npz')['V']
    base = np.hstack([F[:, F.std(0) > 1e-8], E, V3])
    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(V, y, g))

    def run(X, tag, make, kbest=0):
        oof = np.full(len(y), np.nan)
        for fold, (tr, te) in enumerate(splits, 1):
            A, T = X[tr], X[te]
            if kbest:
                tf = FeatureTransform(kbest, 'kbest').fit(A, y[tr])
                A, T = tf.transform(A).astype(np.float64), tf.transform(T).astype(np.float64)
            oof[te] = make().fit(A, y[tr]).predict_proba(T)[:, 1]
            print(f'[{tag}] fold={fold}/5 elapsed={time.perf_counter()-tick:.0f}s', flush=True)
        r = report(y, oof, g)
        print(f'[result] {tag} AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f} subjAUC={r["subject_AUC"]:.4f}', flush=True)
        return r, oof

    lr = lambda C=1: LogisticRegression(C=C, max_iter=2000, class_weight='balanced', random_state=0)
    res, oofs = {}, {}
    for family in ('welch', 'multitaper', 'modulation', 'spectral'):
        key = 'v5_'+family
        res[key], oofs[key] = run(V[:, slices[family]], key, lambda: lr(.1))
    res['v5_all_lr'], oofs['v5_all_lr'] = run(V, 'v5_all_lr', lambda: lr(.1))
    res['base_v5_lr'], oofs['base_v5_lr'] = run(np.hstack([base, V]), 'base_v5_lr', lambda: lr(1), kbest=300)
    gb = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_depth=3,
                                                l2_regularization=1., random_state=0)
    res['base_v5_histgb'], oofs['base_v5_histgb'] = run(np.hstack([base, V]), 'base_v5_histgb', gb)
    np.savez_compressed(OUT/'features_v5_oof.npz', y=y, groups=g, **oofs)
    res['_meta'] = {'seconds': time.perf_counter()-tick, 'n_features': int(V.shape[1]),
                    'single_feature_auc': {n: float(a) for a, n in single},
                    'families': {k: [int(v.start), int(v.stop)] for k, v in slices.items()}}
    (OUT/'features_v5_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[done] elapsed={time.perf_counter()-tick:.0f}s', flush=True)


if __name__ == '__main__':
    main()
