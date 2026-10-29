"""从数据库的原始窗补算"alpha 峰 / 爆发动态 / 谱形状"特征，检验是否带来新信息。

现有 878 维里 alpha 只有固定 8–13 Hz 的功率统计，没有个体峰位置、也没有爆发的时间结构。
训练与评估全部按受试者分组，且只使用当前窗口内的样本。
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.signal import welch, butter, sosfilt, hilbert
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]; DB = ROOT/'database'; OUT = ROOT/'test'
sys.path[:0] = [str(OUT)]
from highdim_deep_ablation import FeatureTransform

FS = 250.


def features(x):
    f, p = welch(x, FS, nperseg=750, noverlap=375)
    band = (f >= 1) & (f <= 45)
    fb, pb = f[band], p[band]+1e-20
    # 个体 alpha 峰
    m = (f >= 7) & (f <= 13)
    iaf = f[m][np.argmax(p[m])] if m.sum() else 0.
    pf = np.max(p[m]) if m.sum() else 0.
    prom = pf/(np.median(p[m])+1e-20) if m.sum() else 0.
    # 1/f 斜率
    slope = np.polyfit(np.log(fb), np.log(pb), 1)[0]
    # 谱熵
    q = pb/pb.sum()
    sent = -np.sum(q*np.log(q))/np.log(len(q))
    # 频带功率与比值（对数域）
    def bp(lo, hi):
        k = (f >= lo) & (f < hi)
        return np.trapezoid(p[k], f[k])+1e-20
    d, th, al, be = bp(1, 4), bp(4, 8), bp(8, 13), bp(13, 30)
    ratios = [np.log(th/al), np.log(be/al), np.log(be/(al+th)), np.log(al/(d+th)), np.log((al+th)/be)]
    # alpha 爆发：0.5 s 子段上的包络功率序列
    sos = butter(3, [8, 13], fs=FS, btype='bandpass', output='sos')
    a = np.abs(hilbert(sosfilt(sos, x)))
    seg = len(a)//125
    if seg >= 2:
        env = a[:seg*125].reshape(seg, 125).mean(1)
        thr = np.median(env)
        on = env > thr
        burst_frac = float(on.mean())
        # 平均爆发时长（连续段长度，单位 0.5 s）
        durs, run = [], 0
        for v in on:
            if v:
                run += 1
            elif run:
                durs.append(run); run = 0
        if run:
            durs.append(run)
        burst_dur = float(np.mean(durs)) if durs else 0.
        burst_rate = len(durs)/6.
        env_cv = float(env.std()/(env.mean()+1e-20))
    else:
        burst_frac = burst_dur = burst_rate = env_cv = 0.
    # 二值化 Lempel-Ziv 复杂度
    s = (x > np.median(x)).astype(np.int8)
    lz = _lz(s)
    return [iaf, prom, slope, sent, *ratios, burst_frac, burst_dur, burst_rate, env_cv, lz,
            float(np.log(al)), float(np.log(be/(al+1e-20)))]


def _lz(s):
    """粗粒度 Lempel-Ziv 复杂度（归一化）。"""
    n = len(s)
    i, k, l, c, k_max = 0, 1, 1, 1, 1
    while True:
        if s[i+k-1] == s[l+k-1]:
            k += 1
            if l+k > n:
                c += 1
                break
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l+1 > n:
                    break
                i, k, k_max = 0, 1, 1
            else:
                k = 1
    return c*1.0/n


NAMES = ['iaf_freq', 'iaf_prom', 'slope_1_40', 'spec_entropy', 'log_theta_alpha', 'log_beta_alpha',
         'log_engage', 'log_alpha_delta', 'log_inv_engage', 'burst_frac', 'burst_dur', 'burst_rate',
         'env_cv', 'lz_complexity', 'log_alpha_power', 'log_beta_alpha2']


def load():
    fs, es, xs, ys, gs = [], [], [], [], []
    for p in sorted((DB/'windows').glob('*/*.npz')):
        d = np.load(p)
        fs.append(d['f']); es.append(d['e']); xs.append(d['x']); ys.append(d['label']); gs.append(d['subject'])
    return (np.vstack(fs).astype(np.float64), np.vstack(es).astype(np.float64),
            np.vstack(xs), np.concatenate(ys).astype(int), np.concatenate(gs))


def report(y, s, g):
    r = {'AUC': float(roc_auc_score(y, s)), 'bACC': float(balanced_accuracy_score(y, s >= .5))}
    per = [{'AUC': roc_auc_score(y[g == k], s[g == k]), 'bACC': balanced_accuracy_score(y[g == k], s[g == k] >= .5)}
           for k in np.unique(g) if len(set(y[g == k])) == 2]
    r.update({f'subject_{m}': float(np.mean([d[m] for d in per])) for m in ('AUC', 'bACC')})
    return r


def run(X, y, g, make, tag, kbest=300):
    oof = np.full(len(y), np.nan)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, g):
        if kbest:
            tf = FeatureTransform(kbest, 'kbest').fit(X[tr], y[tr])
            A, T = tf.transform(X[tr]).astype(np.float64), tf.transform(X[te]).astype(np.float64)
        else:
            A, T = X[tr], X[te]
        oof[te] = make().fit(A, y[tr]).predict_proba(T)[:, 1]
    r = report(y, oof, g)
    print(f'[{tag}] AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f} subjAUC={r["subject_AUC"]:.4f}', flush=True)
    return r


def main():
    tick = time.perf_counter()
    F, E, RAW, y, g = load()
    cache = OUT/'features_v3.npz'
    if cache.exists():
        V = np.load(cache)['V']
    else:
        V = np.array([features(x.astype(float)) for x in RAW], dtype=float)
        np.savez_compressed(cache, V=V, names=np.array(NAMES))
    print(f'[v3] V={V.shape} {time.perf_counter()-tick:.0f}s', flush=True)
    for i, n in enumerate(NAMES):
        print(f'   {n:20s} mean={np.nanmean(V[:, i]):8.3f} sd={np.nanstd(V[:, i]):7.3f} '
              f'corr_y={np.corrcoef(V[:, i], y)[0, 1]:+.3f}', flush=True)

    X = np.hstack([F[:, F.std(0) > 1e-8], E])
    lr = lambda: LogisticRegression(C=1, max_iter=2000, class_weight='balanced', random_state=0)
    res = {}
    res['v3_only_logistic'] = run(V, y, g, lr, '仅 v3 特征', kbest=0)
    res['base'] = run(X, y, g, lr, '基线 Logistic300')
    res['base_v3_logistic'] = run(np.hstack([X, V]), y, g, lr, '基线 + v3 · Logistic300')
    res['base_v3_histgb'] = run(np.hstack([X, V]), y, g,
                                lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_depth=3,
                                                                       l2_regularization=1., random_state=0),
                                '基线 + v3 · HistGB 全维', kbest=0)
    res['_meta'] = {'seconds': time.perf_counter()-tick, 'names': NAMES,
                    'corr_y': {n: float(np.corrcoef(V[:, i], y)[0, 1]) for i, n in enumerate(NAMES)}}
    (OUT/'features_v3_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[done] {time.perf_counter()-tick:.0f}s', flush=True)


if __name__ == '__main__':
    main()
