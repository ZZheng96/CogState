"""伪迹鲁棒的特征估计：把窗内均值功率换成中位数/截尾均值，并显式度量突发伪迹。

现有 causal_features 用 Welch 均值估计频带功率，单个眨眼或动作突发会把整窗的值抬高。
这里在同一批原始窗上重算，只改"如何汇总窗内的 1 秒子段"，用于隔离"鲁棒估计"本身的贡献。
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfilt, iirnotch, lfilter
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.stats import rankdata

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from b_calib_variants import subject_quantile
from singlem_eval import smooth, boot

FS = 250.
BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))
NAMES = [f'{lo}_{hi}' for lo, hi in BANDS]
K = 120


def preprocess(raw):
    """窗内因果链：去均值 -> 50 Hz 陷波 -> 1-45 Hz 带通（全部单向前向滤波）。"""
    b, a = iirnotch(50., 30., FS)
    sos = butter(4, [1./(FS/2), 45./(FS/2)], btype='band', output='sos')
    out = np.empty_like(raw, dtype=np.float64)
    for i, x in enumerate(raw):
        v = lfilter(b, a, x - x.mean())
        out[i] = sosfilt(sos, v)
    return out


def window_stats(filt, raw):
    """把 6 秒窗切成 6 个 1 秒子段，逐子段算频带功率，再分别用 mean/median/截尾均值汇总。"""
    n, L = filt.shape
    seg = filt.reshape(n, 6, L//6)
    win = np.hanning(L//6)
    P = np.abs(np.fft.rfft(seg*win, axis=2))**2          # (n,6,126)，1 Hz 分辨率
    bp = np.stack([P[:, :, lo:hi].sum(2) for lo, hi in BANDS], 2)   # (n,6,5)
    total = P[:, :, 1:45].sum(2)                          # (n,6)
    stat = {'mean': bp.mean(1), 'med': np.median(bp, 1),
            'trim': np.sort(bp, 1)[:, 1:5].mean(1)}
    tot = {'mean': total.mean(1), 'med': np.median(total, 1),
           'trim': np.sort(total, 1)[:, 1:5].mean(1)}
    # 突发伪迹：子段 RMS 相对本窗中位数的偏离，以及波峰因数
    rms = np.sqrt((seg**2).mean(2))
    burst = (rms > 2.5*np.median(rms, 1, keepdims=True)).mean(1)
    crest = np.abs(filt).max(1)/(np.sqrt((filt**2).mean(1))+1e-20)
    hf = np.log(np.var(np.diff(raw, axis=1), axis=1)+1e-20)
    art = np.column_stack([burst, np.log(np.clip(crest, 1e-6, None)), hf])
    return stat, tot, art


def evaluate(F, y, g, pos, calib, seed=0):
    mask = pos >= K
    oof = np.full(len(y), np.nan)
    splits = StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(F, y, g)
    for tr, te in splits:
        A = subject_quantile(F, g, K) if calib == 'feat' else F
        m = make_pipeline(StandardScaler(), SVC(C=1., gamma='scale', cache_size=2048)).fit(A[tr], y[tr])
        s = m.decision_function(A[te]) if calib == 'feat' else m.decision_function(F[te])
        oof[te] = s
    if calib == 'score':
        oof = subject_quantile(oof[:, None], g, K)[:, 0]
    v = smooth(oof, g, 5)
    return {'AUC': float(roc_auc_score(y[mask], v[mask])),
            'bACC': float(balanced_accuracy_score(y[mask], v[mask] >= np.median(v[mask]))),
            'raw_score': v}


def main():
    tick = time.perf_counter()
    d = np.load(OUT/'highdim_deep_features.npz')
    raw, y, g = d['raw'].astype(np.float64), d['y'].astype(int), d['groups']
    pos = np.empty(len(y), int)
    for s in np.unique(g):
        i = np.where(g == s)[0]; pos[i] = np.arange(len(i))
    mask = pos >= K
    filt = preprocess(raw)
    print(f'[prep] 窗内因果滤波完成 {time.perf_counter()-tick:.0f}s', flush=True)
    stat, tot, art = window_stats(filt, raw)
    print(f'[prep] 子段统计完成 {time.perf_counter()-tick:.0f}s', flush=True)

    pool = {}
    for f in ('b_branch_oof.npz', 'b_calib_variants_oof.npz', 'b_calib2_oof.npz'):
        z = np.load(OUT/f); pool.update({k: z[k] for k in z.files if k not in ('y', 'groups')})
    P3 = ['quant120_histgb', 'quant120w_histgb', 'quant120_svc', 'base_svc']
    p3 = smooth(np.mean([rankdata(pool[k])/len(y) for k in P3], 0), g, 5)
    base = {'AUC': float(roc_auc_score(y[mask], p3[mask])),
            'bACC': float(balanced_accuracy_score(y[mask], p3[mask] >= np.median(p3[mask])))}
    res = {'base_p3': base, 'burst_rate': float(art[:, 0].mean()), 'variants': {}}
    print(f'[base] P3 AUC={base["AUC"]:.4f} bACC={base["bACC"]:.4f} 突发子段比例={art[:,0].mean():.3f}', flush=True)

    sets = {}
    for st in ('mean', 'med', 'trim'):
        sets[f'{st}_5'] = np.log(stat[st]+1e-20)
        sets[f'{st}_5rel'] = np.hstack([np.log(stat[st]+1e-20),
                                        stat[st]/(tot[st][:, None]+1e-20)])
    sets['all15'] = np.hstack([np.log(stat[st]+1e-20) for st in ('mean', 'med', 'trim')])
    sets['all15_art'] = np.hstack([sets['all15'], art])
    sets['art_only'] = art

    for name, F in sets.items():
        for calib in ('feat', 'score'):
            r = evaluate(F, y, g, pos, calib)
            b = boot(r['raw_score'], p3, y, g, mask)
            res['variants'][f'{name}_{calib}'] = {'AUC': r['AUC'], 'bACC': r['bACC'], 'vs_p3': b,
                                                  'n_features': int(F.shape[1])}
            print(f'[result] {name:12s} calib={calib:5s} AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f} '
                  f'Δ={b["mean"]:+.4f} [{b["p2.5"]:+.4f},{b["p97.5"]:+.4f}] ({time.perf_counter()-tick:.0f}s)',
                  flush=True)
            (OUT/'robust_features_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                                            encoding='utf-8')

    best = max(res['variants'], key=lambda k: res['variants'][k]['AUC'])
    print(f'[best] {best} AUC={res["variants"][best]["AUC"]:.4f}', flush=True)
    r = evaluate(sets[best.rsplit("_", 1)[0]], y, g, pos, best.rsplit("_", 1)[1])
    for w in (.25, .5):
        mix = (1-w)*rankdata(p3)/len(y) + w*rankdata(r['raw_score'])/len(y)
        b = boot(mix, p3, y, g, mask)
        res['variants'][f'mix_{w}'] = {'AUC': float(roc_auc_score(y[mask], mix[mask])),
                                       'bACC': float(balanced_accuracy_score(y[mask], mix[mask] >= np.median(mix[mask]))),
                                       'vs_p3': b}
        print(f'[mix] w={w} AUC={res["variants"][f"mix_{w}"]["AUC"]:.4f} '
              f'Δ={b["mean"]:+.4f} [{b["p2.5"]:+.4f},{b["p97.5"]:+.4f}]', flush=True)

    res['seconds'] = time.perf_counter()-tick
    np.savez_compressed(OUT/'robust_features.npz', y=y, groups=g,
                        art=art, **{f'logbp_{st}': np.log(stat[st]+1e-20) for st in stat})
    (OUT/'robust_features_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
