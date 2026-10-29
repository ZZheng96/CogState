"""补齐 858 维特征池缺失的两类线索：高频肌电功率与眨眼/眼动事件。

现有特征池的谱特征全部来自 1-45 Hz 滤波后的信号，45 Hz 以上的肌电只能通过原始波形的
标准差/峰度间接体现。单通道下肌电张力与眨眼率是 relax/task 的强线索，这里直接从缓存
的原始 6 秒窗（16067×1500 @250Hz）补算，并与现有池拼接做受试者隔离评估。
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.signal import welch, butter, sosfilt, find_peaks
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT/'test'
sys.path[:0] = [str(ROOT/'src'), str(OUT)]
from causal_history import rebuild_meta, report, running_stats
from highdim_deep_ablation import FeatureTransform, fit_network, predict

FS = 250.
BANDS = ((45, 60), (60, 80), (80, 100), (100, 120))


def emg_blink(x):
    """单窗的肌电/眨眼特征。"""
    f, p = welch(x, FS, nperseg=500, noverlap=250)
    total = np.trapezoid(p[(f >= 1) & (f < 120)], f[(f >= 1) & (f < 120)]) + 1e-20
    out = [np.log(np.trapezoid(p[(f >= lo) & (f < hi)], f[(f >= lo) & (f < hi)])+1e-20) for lo, hi in BANDS]
    out += [np.trapezoid(p[(f >= lo) & (f < hi)], f[(f >= lo) & (f < hi)])/total for lo, hi in BANDS]
    d = np.diff(x)
    out += [np.log(np.var(d)+1e-20), np.log(np.mean(np.abs(d))+1e-20),
            np.log(np.percentile(np.abs(x), 99)+1e-20)]
    # 眨眼：0.5-4 Hz 慢波上的大突起
    sos = butter(3, [.5, 4], fs=FS, btype='bandpass', output='sos')
    slow = sosfilt(sos, x)
    sd = slow.std()+1e-20
    peaks, prop = find_peaks(slow, prominence=2.5*sd, width=4)
    widths = prop.get('widths', np.array([0.]))
    out += [len(peaks), float(np.mean(np.abs(slow[peaks]))/sd) if len(peaks) else 0.,
            float(np.mean(widths))/FS if len(peaks) else 0., float(np.std(widths)/FS) if len(peaks) else 0.]
    return out


NAMES = ([f'emg_log_{lo}_{hi}' for lo, hi in BANDS] + [f'emg_rel_{lo}_{hi}' for lo, hi in BANDS] +
         ['raw_log_diffvar', 'raw_log_absdiff', 'raw_log_p99',
          'blink_count', 'blink_amp_sd', 'blink_width', 'blink_width_sd'])


def evaluate(X, y, groups, end, tag, deep=False):
    subjects = np.unique(groups)
    order = [np.where(groups == s)[0][np.argsort(end[groups == s])] for s in subjects]
    oof = np.full(len(y), np.nan)
    for fold, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, groups), 1):
        mu0, var0 = X[tr].mean(0), X[tr].var(0)+1e-20
        Xn = np.empty_like(X, dtype=float)
        for ids in order:
            m, s = running_stats(X[ids].astype(float), 5, mu0, var0); Xn[ids] = (X[ids]-m)/s
        t = FeatureTransform(300, 'kbest').fit(Xn[tr], y[tr])
        clf = LogisticRegression(C=1, max_iter=2000, class_weight='balanced', random_state=0).fit(t.transform(Xn[tr]), y[tr])
        oof[te] = clf.predict_proba(t.transform(Xn[te]))[:, 1]
        print(f'[{tag} fold {fold}/5] AUC={roc_auc_score(y[te], oof[te]):.4f}', flush=True)
    r = report(y, oof, groups, .5)
    print(f'[{tag}] AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f} subjAUC={r["subject_AUC"]:.4f}', flush=True)
    return r, oof


def main():
    tick = time.perf_counter()
    F = np.load(OUT/'highdim_deep_features.npz')
    X, raw, y, groups = F['X'], F['raw'], F['y'], F['groups']
    meta = rebuild_meta()
    end = np.array([m[2] for m in meta])

    cache = OUT/'emg_blink_features.npz'
    if cache.exists():
        E = np.load(cache)['E']
    else:
        E = np.array([emg_blink(x) for x in raw], dtype=float)
        np.savez_compressed(cache, E=E)
    print(f'[extract] E={E.shape} {time.perf_counter()-tick:.0f}s', flush=True)
    for i, n in enumerate(NAMES):
        print(f'  {n:22s} mean={E[:, i].mean():9.4f} sd={E[:, i].std():8.4f} '
              f'corr_y={np.corrcoef(E[:, i], y)[0, 1]:+.3f}', flush=True)

    res = {}
    res['仅肌电/眨眼 (22维)'], _ = evaluate(E, y, groups, end, 'emg_only')
    res['858维池 (基线)'], o1 = evaluate(X, y, groups, end, 'pool858')
    res['858 + 肌电/眨眼'], o2 = evaluate(np.hstack([X, E]), y, groups, end, 'pool858+emg')

    from scipy.stats import rankdata
    R = lambda v: rankdata(v)/len(v)
    res['集成: 858 + (858+EMG)'] = report(y, .5*R(o1)+.5*R(o2), groups, .5)
    print(f'[集成] AUC={res["集成: 858 + (858+EMG)"]["AUC"]:.4f} bACC={res["集成: 858 + (858+EMG)"]["bACC"]:.4f}', flush=True)

    np.savez_compressed(OUT/'emg_blink_oof.npz', y=y, groups=groups, pool858=o1, pool858_emg=o2)
    res['_meta'] = {'seconds': time.perf_counter()-tick, 'names': NAMES,
                    'corr_y': {n: float(np.corrcoef(E[:, i], y)[0, 1]) for i, n in enumerate(NAMES)}}
    (OUT/'emg_blink_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[done] {time.perf_counter()-tick:.0f}s', flush=True)


if __name__ == '__main__':
    main()
