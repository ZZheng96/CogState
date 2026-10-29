"""高频肌电/伪迹特征的细化版本 + 深度模型 + 集成。

第一版 15 个特征已把 858 维池从 0.6572 提到 0.6770，说明 1-45 Hz 之外的频段有独立信息。
这里加密频段、补谱斜率与爆发比例，再接大 MLP 与 rank 集成。
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.signal import welch, butter, sosfilt, find_peaks
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
import torch

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT/'test'
sys.path[:0] = [str(ROOT/'src'), str(OUT)]
from causal_history import rebuild_meta, report, running_stats
from highdim_deep_ablation import FeatureTransform, fit_network, predict

FS = 250.
FINE = ((40, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 100), (100, 120))


def features_v2(x):
    f, p = welch(x, FS, nperseg=500, noverlap=250)
    tot = np.trapezoid(p[(f >= 1) & (f < 120)], f[(f >= 1) & (f < 120)]) + 1e-20
    pk, rl = [], []
    for lo, hi in FINE:
        v = np.trapezoid(p[(f >= lo) & (f < hi)], f[(f >= lo) & (f < hi)])
        pk.append(np.log(v+1e-20)); rl.append(v/tot)
    # 40-120 Hz 对数谱斜率（肌电越强，斜率越平）
    m = (f >= 40) & (f < 120)
    slope = np.polyfit(np.log(f[m]), np.log(p[m]+1e-20), 1)[0]
    sos = butter(4, 40, fs=FS, btype='highpass', output='sos')
    hp = sosfilt(sos, x)
    sd = hp.std()+1e-20
    sos_low = butter(3, [.5, 4], fs=FS, btype='bandpass', output='sos')
    slow = sosfilt(sos_low, x)
    peaks, prop = find_peaks(slow, prominence=2.5*(slow.std()+1e-20), width=4)
    widths = prop.get('widths', np.array([0.]))
    return pk + rl + [slope, np.log(sd+1e-20), float(np.mean((hp/sd)**4)),
                      float(np.mean(np.abs(hp) > 3*sd)), len(peaks),
                      float(np.mean(widths))/FS if len(peaks) else 0.]


NAMES = ([f'emg_log_{lo}_{hi}' for lo, hi in FINE] + [f'emg_rel_{lo}_{hi}' for lo, hi in FINE] +
         ['emg_slope_40_120', 'emg_log_sd', 'emg_kurtosis_hp', 'emg_burst_frac', 'blink_count', 'blink_width'])


def evaluate_set(Xall, y, groups, order, tag, device):
    oof = {'logistic': np.full(len(y), np.nan), 'mlp': np.full(len(y), np.nan)}
    for fold, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Xall, y, groups), 1):
        t0 = time.perf_counter()
        mu0, var0 = Xall[tr].mean(0), Xall[tr].var(0)+1e-20
        Xn = np.empty_like(Xall, dtype=float)
        for ids in order:
            m, s = running_stats(Xall[ids].astype(float), 5, mu0, var0); Xn[ids] = (Xall[ids]-m)/s
        a, b = next(StratifiedGroupKFold(4, shuffle=True, random_state=fold).split(Xn[tr], y[tr], groups[tr]))
        inner, val = tr[a], tr[b]
        tf = FeatureTransform(300, 'kbest').fit(Xn[inner], y[inner])
        A, V, T = tf.transform(Xn[inner]), tf.transform(Xn[val]), tf.transform(Xn[te])
        lr = LogisticRegression(C=1, max_iter=2000, class_weight='balanced', random_state=0).fit(A, y[inner])
        oof['logistic'][te] = lr.predict_proba(T)[:, 1]
        net, info = fit_network('large', A, y[inner], V, y[val], device, fold)
        oof['mlp'][te] = predict(net, T, device)
        print(f'[{tag} fold {fold}/5] log={roc_auc_score(y[te], oof["logistic"][te]):.4f} '
              f'mlp={roc_auc_score(y[te], oof["mlp"][te]):.4f} {time.perf_counter()-t0:.0f}s', flush=True)
    return oof


def main():
    tick = time.perf_counter()
    F = np.load(OUT/'highdim_deep_features.npz')
    X, raw, y, groups = F['X'], F['raw'], F['y'], F['groups']
    meta = rebuild_meta()
    end = np.array([m[2] for m in meta])
    cache = OUT/'emg_features_v2.npz'
    if cache.exists():
        E = np.load(cache)['E']
    else:
        E = np.array([features_v2(x) for x in raw], dtype=float)
        np.savez_compressed(cache, E=E, names=np.array(NAMES))
    # 稳健性：去掉 40-60 Hz（可能混入 50 Hz 工频），只保留 60 Hz 以上
    keep = [i for i, n in enumerate(NAMES) if not n.endswith('40_50') and not n.endswith('50_60')]
    XE = np.hstack([X, E]); XE_hf = np.hstack([X, E[:, keep]])
    print(f'[features] E={E.shape} full={XE.shape} hf-only={XE_hf.shape} {time.perf_counter()-tick:.0f}s', flush=True)

    subjects = np.unique(groups)
    order = [np.where(groups == s)[0][np.argsort(end[groups == s])] for s in subjects]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    full = evaluate_set(XE, y, groups, order, 'full', device)
    hf = evaluate_set(XE_hf, y, groups, order, 'hf', device)
    oof = {'emg_v2_logistic': full['logistic'], 'emg_v2_mlp': full['mlp'],
           'emg_hf_logistic': hf['logistic'], 'emg_hf_mlp': hf['mlp']}

    res = {}
    for k, v in oof.items():
        res[k] = report(y, v, groups, .5)
        print(f'[{k}] AUC={res[k]["AUC"]:.4f} bACC={res[k]["bACC"]:.4f} subjAUC={res[k]["subject_AUC"]:.4f}', flush=True)
    np.savez_compressed(OUT/'emg_v2_oof.npz', y=y, groups=groups, **oof)

    # 与既有最优成员做嵌套集成
    A0 = np.load(OUT/'online_deep_oof.npz'); B0 = np.load(OUT/'highdim_deep_oof.npz')
    cand = {'emg_v2_mlp': oof['emg_v2_mlp'], 'emg_v2_logistic': oof['emg_v2_logistic'],
            'emg_hf_mlp': oof['emg_hf_mlp'], 'emg_hf_logistic': oof['emg_hf_logistic'],
            'online_mlp300': A0['online_mlp300'], 'online_logistic300': A0['online_logistic300'],
            'large_mlp300': B0['large_mlp300']}
    R = {k: rankdata(v)/len(y) for k, v in cand.items()}
    import itertools
    combos, subs_list = {}, []
    for r in range(1, 5):
        for sub in itertools.combinations(cand, r):
            combos['+'.join(sub)] = report(y, np.mean([R[k] for k in sub], axis=0), groups, .5)
            subs_list.append(sub)
    best = max(combos, key=lambda k: combos[k]['AUC'])
    print(f'[best combo] {best} AUC={combos[best]["AUC"]:.4f} bACC={combos[best]["bACC"]:.4f}', flush=True)

    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(R['emg_v2_mlp'], y, groups))
    nested = []
    for f, (tr, te) in enumerate(folds, 1):
        b, ba = None, -1
        for sub in subs_list:
            a = roc_auc_score(y[tr], np.mean([R[k] for k in sub], axis=0)[tr])
            if a > ba:
                ba, b = a, sub
        s = np.mean([R[k] for k in b], axis=0)
        nested.append({'fold': f, 'combo': '+'.join(b), 'AUC': float(roc_auc_score(y[te], s[te]))})
        print(f'[nested {f}/5] {"+".join(b)} AUC={nested[-1]["AUC"]:.4f}', flush=True)

    res['combos'] = combos
    res['best_full_combo'] = best
    res['nested'] = nested
    res['nested_mean_AUC'] = float(np.mean([d['AUC'] for d in nested]))
    print(f'[nested mean] AUC={res["nested_mean_AUC"]:.4f}', flush=True)
    res['_meta'] = {'seconds': time.perf_counter()-tick, 'names': NAMES,
                    'corr_y': {n: float(np.corrcoef(E[:, i], y)[0, 1]) for i, n in enumerate(NAMES)}}
    (OUT/'emg_v2_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[done] {time.perf_counter()-tick:.0f}s', flush=True)


if __name__ == '__main__':
    main()
