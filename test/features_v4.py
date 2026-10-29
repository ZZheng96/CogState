"""变换域特征：AR 模型、小波包、双谱/双相干。

现有 878 维集中在幅度谱（STFT/DWT/Hilbert 包络）。这里补三类不同性质的表征：
  - AR(p) 系数：谱包络的参数化（极点位置），经典 EEG 特征；
  - 小波包 level-4：16 个等宽子带的能量与熵（DWT 只沿近似分支分解，覆盖不同）；
  - 双谱 / 双相干：二次相位耦合，反映非线性相互作用，与时序相位有关。
"""
import json, sys, time
from pathlib import Path
import numpy as np
import pywt
from scipy.linalg import solve_toeplitz
from scipy.signal import butter, sosfilt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]; DB = ROOT/'database'; OUT = ROOT/'test'
sys.path[:0] = [str(OUT)]
from highdim_deep_ablation import FeatureTransform

FS = 250.
ORDER = 16
SOS = butter(3, [1, 45], fs=FS, btype='bandpass', output='sos')


def ar_features(x):
    e = sosfilt(SOS, x)
    r = np.correlate(e, e, 'full')[len(e)-1:][:ORDER+1]/len(e)
    r[0] *= 1.02  # 轻微对角加载，保证正定
    a = solve_toeplitz(r[:ORDER], -r[1:ORDER+1])
    err = r[0] + np.dot(a, r[1:ORDER+1])
    # 反射系数（Levinson 递推）
    refl = []
    ak = np.array([1.])
    for m in range(1, ORDER+1):
        km = -np.dot(ak, r[m:0:-1][:m])/max(np.dot(ak, ak), 1e-20)
        ak = np.r_[ak, 0.] + km*np.r_[0., ak[::-1]]
        refl.append(km)
    return np.r_[a, np.log(err+1e-20), refl]


def wp_features(x):
    wp = pywt.WaveletPacket(x, 'db4', maxlevel=4)
    nodes = []
    for i in range(16):
        path = bin(i)[2:].zfill(4).replace('0', 'a').replace('1', 'd')
        nodes.append(wp[path].data)
    energy = np.array([np.sum(c*c) for c in nodes])+1e-20
    rel = energy/energy.sum()
    ent = -np.sum(rel*np.log(rel))/np.log(16)
    return np.r_[np.log(energy), rel, ent]


def bicoherence(x):
    """分段平均的归一化双谱幅值 b(f1,f2)=|E[X1 X2 X3*]| / sqrt(E|X1|^2 E|X2|^2 E|X3|^2)。"""
    e = sosfilt(SOS, x)
    nseg, nfft = 6, 256
    step = max((len(e)-nfft)//max(nseg-1, 1), 1)
    Fs = np.fft.rfftfreq(nfft, 1/FS)
    nf = nfft//2+1
    idx = np.arange(nf)
    i, j = np.meshgrid(idx, idx, indexing='ij')
    k = i+j
    valid = k < nf
    acc = np.zeros((nf, nf), complex)
    p1 = np.zeros((nf, nf)); p2 = np.zeros((nf, nf)); p3 = np.zeros((nf, nf))
    for s in range(nseg):
        seg = e[s*step:s*step+nfft]
        if len(seg) < nfft:
            break
        seg = (seg-seg.mean())*np.hanning(len(seg))
        F = np.fft.rfft(seg)
        X1 = np.broadcast_to(F[:, None], (nf, nf))
        X2 = np.broadcast_to(F[None, :], (nf, nf))
        X3 = np.zeros((nf, nf), complex)
        X3[valid] = np.conj(F[k[valid]])
        acc += X1*X2*X3
        p1 += np.abs(X1)**2; p2 += np.abs(X2)**2; p3 += np.abs(X3)**2
    mag = np.abs(acc)/np.sqrt(np.maximum(p1*p2*p3, 1e-30))
    def region(lo1, hi1, lo2, hi2):
        m = ((Fs >= lo1) & (Fs < hi1))[:, None] & ((Fs >= lo2) & (Fs < hi2))[None, :]
        return float(np.mean(mag[m])) if m.sum() else 0.
    return np.array([region(4, 8, 4, 8), region(8, 13, 8, 13), region(4, 8, 8, 13),
                     region(8, 13, 13, 30), region(4, 8, 13, 30), region(1, 4, 8, 13),
                     float(np.mean(mag)), float(np.max(mag))])


NAMES = ([f'ar_coef_{i}' for i in range(ORDER)] + ['ar_log_err'] +
         [f'ar_refl_{i}' for i in range(1, ORDER+1)] +
         [f'wp_log_energy_{i}' for i in range(16)] + [f'wp_rel_{i}' for i in range(16)] + ['wp_entropy'] +
         ['bic_theta_theta', 'bic_alpha_alpha', 'bic_theta_alpha', 'bic_alpha_beta',
          'bic_theta_beta', 'bic_delta_alpha', 'bic_mean', 'bic_max'])


def features(x):
    return np.r_[ar_features(x), wp_features(x), bicoherence(x)]


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
        A, T = X[tr], X[te]
        if kbest:
            tf = FeatureTransform(kbest, 'kbest').fit(A, y[tr])
            A, T = tf.transform(A).astype(np.float64), tf.transform(T).astype(np.float64)
        oof[te] = make().fit(A, y[tr]).predict_proba(T)[:, 1]
    r = report(y, oof, g)
    print(f'[{tag}] AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f} subjAUC={r["subject_AUC"]:.4f}', flush=True)
    return r, oof


def main():
    tick = time.perf_counter()
    F, E, RAW, y, g = load()
    cache = OUT/'features_v4.npz'
    if cache.exists():
        V = np.load(cache)['V']
    else:
        V = np.array([features(x.astype(float)) for x in RAW], dtype=float)
        np.savez_compressed(cache, V=V, names=np.array(NAMES))
    print(f'[v4] V={V.shape} {time.perf_counter()-tick:.0f}s', flush=True)
    V = np.nan_to_num(V, nan=0., posinf=0., neginf=0.)
    single = [(n, float(roc_auc_score(y, V[:, i])), float(np.corrcoef(V[:, i], y)[0, 1])) for i, n in enumerate(NAMES)]
    single.sort(key=lambda t: -abs(t[1]-.5))
    print('[单特征区分度 top12]', flush=True)
    for n, a, c in single[:12]:
        print(f'   {n:22s} AUC={a:.3f} corr={c:+.3f}', flush=True)

    lr = lambda: LogisticRegression(C=1, max_iter=2000, class_weight='balanced', random_state=0)
    gb = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_depth=3,
                                                l2_regularization=1., random_state=0)
    base = np.hstack([F[:, F.std(0) > 1e-8], E])
    V3 = np.load(OUT/'features_v3.npz')['V']
    res, oofs = {}, {}
    res['v4_only'], oofs['v4'] = run(V, y, g, lr, '仅变换域 v4', kbest=0)
    res['base892'], oofs['base892'] = run(np.hstack([base, V3]), y, g, lr, '基线 892 · Logistic300')
    res['base_v4'], oofs['base_v4'] = run(np.hstack([base, V3, V]), y, g, lr, '基线 + v4 · Logistic300')
    res['base_v4_histgb'], oofs['base_v4_histgb'] = run(np.hstack([base, V3, V]), y, g, gb, '基线 + v4 · HistGB 全维', kbest=0)
    np.savez_compressed(OUT/'features_v4_oof.npz', y=y, groups=g, **oofs)
    res['_meta'] = {'seconds': time.perf_counter()-tick, 'names': NAMES,
                    'single_feature_auc': {n: a for n, a, _ in single}}
    (OUT/'features_v4_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[done] {time.perf_counter()-tick:.0f}s', flush=True)


if __name__ == '__main__':
    main()
