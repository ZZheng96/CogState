"""窗口长度消融：3s / 4s 窗（等长跳步），与 6s 基线使用同一套 858 维特征与在线标准化流程。

窗口越短 -> 窗数越多、切换瞬态占比越小，但频谱分辨率下降。用于回答"6 秒窗是否最优"。
"""
import json, sys, time
from pathlib import Path
import numpy as np
import mne
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT/'test'
sys.path[:0] = [str(ROOT/'src'), str(OUT)]
from cogstate import load_trials, bad_intervals, valid_window
import nonstationary_ablation as nsa
import highdim_deep_ablation as hd
from causal_history import report, running_stats
from highdim_deep_ablation import FeatureTransform

# 原特征函数硬性要求 6 秒窗；这里只放开长度检查，特征定义本身不变。
_patch = lambda x, fs=250.: np.asarray(x, dtype=float)
nsa.check_window = _patch
hd.check_window = _patch
highdim_features = hd.highdim_features


def extract(w):
    cache = OUT/f'window{int(w)}s_features.npz'
    if cache.exists():
        d = np.load(cache)
        print(f'[cache] window {w}s -> {d["X"].shape}', flush=True)
        return d['X'], d['y'], d['groups'], d['end']
    X, y, groups, end = [], [], [], []
    paths = sorted((ROOT/'EEG_dataset').glob('*/*_events.csv'))
    tick = time.perf_counter()
    for i, csv in enumerate(paths, 1):
        bdf = next(csv.parent.glob('*.bdf'))
        raw = mne.io.read_raw_bdf(str(bdf), preload=True, verbose='ERROR')
        fs, bad = float(raw.info['sfreq']), bad_intervals(raw)
        for tr in load_trials(str(csv), str(bdf)):
            t = tr.start+w
            while t <= tr.end+1e-6:
                if valid_window(t-w, t, bad, raw.n_times/fs):
                    x = raw.get_data(start=round((t-w)*fs), stop=round(t*fs))[0]
                    if len(x) == round(w*fs) and np.isfinite(x).all():
                        X.append(highdim_features(x, fs)); y.append(tr.label)
                        groups.append(tr.subject); end.append(t)
                t += w
        raw.close()
        print(f'[extract {w}s {i}/{len(paths)}] windows={len(y)} elapsed={time.perf_counter()-tick:.0f}s', flush=True)
    X, y, groups, end = np.asarray(X), np.asarray(y), np.asarray(groups), np.asarray(end)
    np.savez_compressed(cache, X=X, y=y, groups=groups, end=end)
    return X, y, groups, end


def evaluate(X, y, groups, end, online, tag):
    streams = [np.where(groups == s)[0][np.argsort(end[groups == s])] for s in np.unique(groups)]
    oof = np.full(len(y), np.nan)
    for fold, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, groups), 1):
        if online:
            mu0, var0 = X[tr].mean(0), X[tr].var(0)+1e-20
            Xn = np.empty_like(X, dtype=float)
            for ids in streams:
                m, s = running_stats(X[ids].astype(float), 5, mu0, var0); Xn[ids] = (X[ids]-m)/s
        else:
            Xn = X.astype(float)
        t = FeatureTransform(300, 'kbest').fit(Xn[tr], y[tr])
        clf = LogisticRegression(C=1, max_iter=2000, class_weight='balanced', random_state=0).fit(t.transform(Xn[tr]), y[tr])
        oof[te] = clf.predict_proba(t.transform(Xn[te]))[:, 1]
        print(f'[{tag} fold {fold}/5] AUC={roc_auc_score(y[te], oof[te]):.4f}', flush=True)
    r = report(y, oof, groups, .5)
    print(f'[{tag}] n={len(y)} AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f} subjAUC={r["subject_AUC"]:.4f}', flush=True)
    return r


def main():
    tick = time.perf_counter()
    res = {}
    for w in (3., 4.):
        X, y, groups, end = extract(w)
        res[f'{int(w)}s'] = evaluate(X, y, groups, end, True, f'window {int(w)}s + online')
        res[f'{int(w)}s']['n_windows'] = int(len(y))
    res['meta'] = {'seconds': time.perf_counter()-tick, 'note': '6s 基线同流程为 AUC 0.6547 / bACC 0.6176'}
    (OUT/'window_ablation_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[done] {time.perf_counter()-tick:.0f}s', flush=True)


if __name__ == '__main__':
    main()
