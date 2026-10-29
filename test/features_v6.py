"""A 支 v6：6 秒窗内部的动态过程。

现有特征基本是整窗汇总。这里补四类窗内时间结构，全部只用当前窗样本：
  - 6 个 1 秒子窗的频带功率轨迹：逐秒对数功率、跨秒变异系数、线性趋势；
  - 半窗对比：后半窗减前半窗的对数功率差与谱斜率差；
  - 跨频带包络耦合：频带对的包络相关（零延迟与 ±200 ms 内的最大绝对值）；
  - 爆发形态：alpha 包络的偏度、峰度、爆发率、平均时长、峰中位比。
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfilt, hilbert
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]; DB = ROOT/'database'; OUT = ROOT/'test'
sys.path[:0] = [str(OUT)]
from highdim_deep_ablation import FeatureTransform

FS = 250.
BANDS = ((1, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 45))
SEG = 250          # 1 秒
NSEG = 6
ENV_FS = 25        # 包络降采样频率
LAGS = (-5, -3, 0, 3, 5)   # ±200 ms @ 25 Hz
ALPHA = 2          # 8–13 Hz 的频带序号

SOS = [butter(3, [lo, hi], fs=FS, btype='bandpass', output='sos') for lo, hi in BANDS]


def env_ds(a, n=150):
    """把 6 秒包络降采样到 25 Hz，只取整段。"""
    return a[:n*10].reshape(n, 10).mean(1)


def features(x):
    x = np.asarray(x, dtype=float)
    seg_pow, envs, slope_seg = [], [], []
    for (lo, hi), sos in zip(BANDS, SOS):
        band = sosfilt(sos, x)
        p = band**2
        seg = p[:NSEG*SEG].reshape(NSEG, SEG).mean(1)
        seg_pow.append(np.log(seg+1e-20))
        envs.append(env_ds(np.abs(hilbert(band))))
        # 逐秒谱斜率（对数功率对对数频率的回归斜率）
        sl = []
        for s in range(NSEG):
            seg_sig = band[s*SEG:(s+1)*SEG]
            f = np.fft.rfftfreq(len(seg_sig), 1/FS)
            m = (f >= 1) & (f <= 45)
            sp = np.abs(np.fft.rfft(seg_sig*np.hanning(len(seg_sig))))**2
            sl.append(np.polyfit(np.log(f[m]+1e-20), np.log(sp[m]+1e-20), 1)[0])
        slope_seg.append(sl)
    seg_pow = np.array(seg_pow)          # (6 band, 6 seg)
    slope_seg = np.array(slope_seg)
    envs = np.array(envs)                # (6 band, 150)

    vals = [seg_pow.ravel()]
    # 跨秒变异与趋势
    vals.append(np.log(seg_pow.std(1)+1e-20))
    t = np.arange(NSEG)-(NSEG-1)/2
    vals.append(seg_pow @ t/(t@t))
    # 半窗对比
    vals.append(seg_pow[:, 3:].mean(1)-seg_pow[:, :3].mean(1))
    vals.append(slope_seg[:, 3:].mean(1)-slope_seg[:, :3].mean(1))

    # 跨频带包络耦合
    ec = envs-envs.mean(1, keepdims=True)
    ec = ec/(ec.std(1, keepdims=True)+1e-12)
    zero, best = [], []
    for i in range(len(BANDS)):
        for j in range(i+1, len(BANDS)):
            zero.append(float(np.mean(ec[i]*ec[j])))
            peak = max(abs(float(np.mean(ec[i][max(0, k):len(ec[i])+min(0, k)] *
                                   ec[j][max(0, -k):len(ec[j])-max(0, k)])))
                       for k in LAGS)
            best.append(peak)
    vals.append(np.array(zero))
    vals.append(np.array(best))

    # alpha 爆发形态
    a = envs[ALPHA]
    med = np.median(a)+1e-20
    z = (a-a.mean())/(a.std()+1e-12)
    on = a > 1.5*med
    durs, run = [], 0
    for v in on:
        if v:
            run += 1
        elif run:
            durs.append(run); run = 0
    if run:
        durs.append(run)
    vals.append(np.array([float(np.mean(z**3)), float(np.mean(z**4)-3),
                          float(on.mean()), float(np.mean(durs)) if durs else 0.,
                          float(a.max()/med)]))

    # 谱斜率的跨秒变异
    vals.append(np.log(slope_seg.std(1)+1e-12))
    return np.concatenate([np.asarray(v, dtype=float).ravel() for v in vals])


def names():
    out = [f'seg_logpow_b{b}_t{t}' for b in range(len(BANDS)) for t in range(NSEG)]
    out += [f'seg_cv_b{b}' for b in range(len(BANDS))]
    out += [f'seg_trend_b{b}' for b in range(len(BANDS))]
    out += [f'half_dlogpow_b{b}' for b in range(len(BANDS))]
    out += [f'half_dslope_b{b}' for b in range(len(BANDS))]
    pairs = [(i, j) for i in range(len(BANDS)) for j in range(i+1, len(BANDS))]
    out += [f'envcorr_b{i}_b{j}' for i, j in pairs]
    out += [f'envcorr_peak_b{i}_b{j}' for i, j in pairs]
    out += ['alpha_skew', 'alpha_kurt', 'alpha_burst_frac', 'alpha_burst_dur', 'alpha_peak_ratio']
    out += [f'slope_cv_b{b}' for b in range(len(BANDS))]
    return out


def load():
    fs, es, xs, ys, gs = [], [], [], [], []
    for p in sorted((DB/'windows').glob('*/*.npz')):
        d = np.load(p); fs.append(d['f']); es.append(d['e']); xs.append(d['x'])
        ys.append(d['label']); gs.append(d['subject'])
    F = np.vstack(fs).astype(np.float64); F = F[:, F.std(0) > 1e-8]
    return (np.hstack([F, np.vstack(es).astype(np.float64)]),
            np.vstack(xs).astype(np.float64), np.concatenate(ys).astype(int), np.concatenate(gs))


def report(y, s, g, proba=True):
    r = {'AUC': float(roc_auc_score(y, s))}
    r['bACC'] = float(balanced_accuracy_score(y, s >= .5)) if proba else None
    per = [roc_auc_score(y[g == k], s[g == k]) for k in np.unique(g) if len(set(y[g == k])) == 2]
    r['subject_AUC'] = float(np.mean(per))
    return r


def main():
    tick = time.perf_counter()
    base, raw, y, g = load()
    cache = OUT/'features_v6.npz'
    if cache.exists():
        V = np.load(cache)['V']
    else:
        V = np.array([features(x) for x in raw], dtype=np.float32)
        np.savez_compressed(cache, V=V, names=np.array(names()))
    V = np.nan_to_num(V.astype(np.float64), nan=0., posinf=0., neginf=0.)
    print(f'[v6] {base.shape} + {V.shape} elapsed={time.perf_counter()-tick:.0f}s', flush=True)

    NAMES = names()
    single = sorted([(roc_auc_score(y, V[:, i]), NAMES[i]) for i in range(V.shape[1])],
                    key=lambda z: -abs(z[0]-.5))
    for a, n in single[:10]:
        print(f'[single] {n:24s} AUC={a:.4f}', flush=True)

    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(base, y, g))

    def run(X, tag, make, proba=True, kbest=300):
        oof = np.full(len(y), np.nan)
        for fold, (tr, te) in enumerate(splits, 1):
            A, T = X[tr], X[te]
            if kbest:
                tf = FeatureTransform(kbest, 'kbest').fit(A, y[tr])
                A, T = tf.transform(A).astype(np.float64), tf.transform(T).astype(np.float64)
            m = make().fit(A, y[tr])
            oof[te] = m.predict_proba(T)[:, 1] if proba else m.decision_function(T)
            print(f'[{tag}] fold={fold}/5 elapsed={time.perf_counter()-tick:.0f}s', flush=True)
        r = report(y, oof, g, proba)
        print(f'[result] {tag} AUC={r["AUC"]:.4f} bACC={r["bACC"]} subjAUC={r["subject_AUC"]:.4f}', flush=True)
        return r, oof

    lr = lambda: LogisticRegression(C=1, max_iter=3000, class_weight='balanced', random_state=0)
    gb = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_depth=3,
                                                l2_regularization=1., random_state=0)
    lda = lambda: LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    svc = lambda: make_pipeline(StandardScaler(), SVC(C=1, gamma='scale', kernel='rbf', cache_size=2048))

    res, oofs = {}, {}
    res['v6_only'], oofs['v6_only'] = run(V, 'v6_only', lr, kbest=0)
    for tag, make, proba in (('lr', lr, True), ('lda', lda, False),
                             ('histgb', gb, True), ('svc', svc, False)):
        key = f'base_v6_{tag}'
        res[key], oofs[key] = run(np.hstack([base, V]), key, make, proba,
                                  0 if tag == 'histgb' else 300)

    np.savez_compressed(OUT/'features_v6_oof.npz', y=y, groups=g, **oofs)
    res['_meta'] = {'seconds': time.perf_counter()-tick, 'n_features': int(V.shape[1]),
                    'single_feature_auc': {n: float(a) for a, n in single}}
    (OUT/'features_v6_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[done] elapsed={time.perf_counter()-tick:.0f}s', flush=True)


if __name__ == '__main__':
    main()
