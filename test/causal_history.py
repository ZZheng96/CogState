"""利用"过去窗口"的因果历史信息：在线自适应（合规）与跨窗平滑（感受野>6s，仅作参照）。

复用 highdim_deep_features.npz / highdim_deep_oof.npz，不重算特征。
"""
import json, sys, time
from pathlib import Path
import numpy as np, mne
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT/'test'
sys.path[:0] = [str(ROOT/'src'), str(OUT)]
from cogstate import load_trials, bad_intervals, valid_window
from highdim_deep_ablation import FeatureTransform


def rebuild_meta():
    """按 extract() 相同顺序重建每个窗口的 subject / trial_id / end。"""
    meta = []
    for csv in sorted((ROOT/'EEG_dataset').glob('*/*_events.csv')):
        bdf = next(csv.parent.glob('*.bdf'))
        raw = mne.io.read_raw_bdf(str(bdf), preload=False, verbose='ERROR')
        fs, bad = float(raw.info['sfreq']), bad_intervals(raw)
        for tr in load_trials(str(csv), str(bdf)):
            end = tr.start+6
            while end <= tr.end+1e-6:
                if valid_window(end-6, end, bad, raw.n_times/fs):
                    meta.append((tr.subject, tr.trial_id, end, tr.label))
                end += 6
        raw.close()
    return meta


def score(y, s, thr):
    return {'AUC': float(roc_auc_score(y, s)), 'bACC': float(balanced_accuracy_score(y, s >= thr))}


def report(y, s, groups, thr):
    r = score(y, s, thr)
    subj = [score(y[groups == g], s[groups == g], thr) for g in np.unique(groups) if len(set(y[groups == g])) == 2]
    r.update({f'subject_{m}': float(np.mean([d[m] for d in subj])) for m in ('AUC', 'bACC')})
    return r


def running_stats(v, n0, mu0, var0):
    """含当前窗的因果累计均值/方差，带先验伪计数 n0。v: (n,) 或 (n,d)。"""
    n = np.arange(1, len(v)+1).reshape(-1, *([1]*(v.ndim-1)))
    m = (n0*mu0 + np.cumsum(v, 0))/(n0+n)
    ex2 = (n0*(var0+mu0**2) + np.cumsum(v**2, 0))/(n0+n)
    return m, np.sqrt(np.maximum(ex2-m**2, 1e-12))


def per_subject(fn, v, order_by_subject):
    out = np.empty_like(v, dtype=float)
    for ids in order_by_subject:
        out[ids] = fn(v[ids])
    return out


def ema(v, a):
    out, acc = np.empty(len(v)), v[0]
    for i, x in enumerate(v):
        acc = a*acc+(1-a)*x; out[i] = acc
    return out


def main():
    tick = time.perf_counter()
    F, P = np.load(OUT/'highdim_deep_features.npz'), np.load(OUT/'highdim_deep_oof.npz')
    X, y, groups = F['X'], F['y'], F['groups']
    meta = rebuild_meta()
    assert len(meta) == len(y), (len(meta), len(y))
    assert all(m[0] == g and m[3] == t for m, g, t in zip(meta, groups, y))
    trial = np.array([f'{m[0]}#{m[1]}' for m in meta]); end = np.array([m[2] for m in meta])
    # 每个受试者按时间排序的连续流（跨 trial 不重置，测试时不知道 trial 边界）
    streams = [np.where(groups == g)[0][np.argsort(end[groups == g], kind='stable')] for g in np.unique(groups)]
    trials = [np.where(trial == t)[0][np.argsort(end[trial == t])] for t in np.unique(trial)]
    print(f'[meta] {len(y)} windows, {len(streams)} subjects, {len(trials)} trials, {time.perf_counter()-tick:.1f}s', flush=True)

    results = {}
    for name in ('large_mlp300', 'logistic300'):
        p = np.clip(P[name], 1e-6, 1-1e-6); z = np.log(p/(1-p))
        R = {'baseline (合规)': report(y, p, groups, .5)}
        # A. 在线分数标准化：用本受试者"过去+当前"窗口的无标签分数均值/方差去偏（合规：在线自适应）
        for n0 in (5, 20):
            def center(v, n0=n0):
                m, s = running_stats(v, n0, 0., 1.); return (v-m)/s
            R[f'online_score_z n0={n0} (合规)'] = report(y, per_subject(center, z, streams), groups, 0.)
        # B. 跨窗平滑（感受野 = 多个 6s 窗，超出 6s 上限，仅作参照）
        for a in (.5, .8, .9, .95):
            R[f'EMA a={a} 连续流 (违规参照)'] = report(y, per_subject(lambda v: ema(v, a), z, streams), groups, 0.)
            R[f'EMA a={a} 按trial重置 (oracle)'] = report(y, per_subject(lambda v: ema(v, a), z, trials), groups, 0.)
        for k in (2, 5, 10):
            mv = lambda v, k=k: np.convolve(np.r_[np.full(k-1, v[0]), v], np.ones(k)/k, 'valid')
            R[f'过去{k}窗均值 感受野{6*k}s (违规参照)'] = report(y, per_subject(mv, z, streams), groups, 0.)
        results[name] = R
        for k, v in R.items():
            print(f'[{name}] {k:32s} AUC={v["AUC"]:.4f} bACC={v["bACC"]:.4f} subjAUC={v["subject_AUC"]:.4f}', flush=True)

    # C. 在线特征标准化后重训 Logistic300（训练和测试都用各自受试者的因果累计统计，合规）
    for n0 in (5, 20):
        oof = np.full(len(y), np.nan)
        for fold, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, groups), 1):
            mu0, var0 = X[tr].mean(0), X[tr].var(0)+1e-20
            Xn = np.empty_like(X, dtype=float)
            for ids in streams:
                m, s = running_stats(X[ids].astype(float), n0, mu0, var0); Xn[ids] = (X[ids]-m)/s
            t = FeatureTransform(300, 'kbest').fit(Xn[tr], y[tr])
            clf = LogisticRegression(C=1, max_iter=2000, class_weight='balanced', random_state=0).fit(t.transform(Xn[tr]), y[tr])
            oof[te] = clf.predict_proba(t.transform(Xn[te]))[:, 1]
            print(f'[online_feature_z n0={n0}] fold {fold}/5 AUC={roc_auc_score(y[te], oof[te]):.4f} elapsed={time.perf_counter()-tick:.0f}s', flush=True)
        results['logistic300'][f'online_feature_z n0={n0} 重训 (合规)'] = report(y, oof, groups, .5)
        print(f'[online_feature_z n0={n0}] {results["logistic300"][f"online_feature_z n0={n0} 重训 (合规)"]}', flush=True)

    # 预热：按受试者流内位置统计在线标准化前后 AUC
    pos = np.empty(len(y), int)
    for ids in streams: pos[ids] = np.arange(len(ids))
    results['meta'] = {'n_windows': int(len(y)), 'seconds': time.perf_counter()-tick,
                       'stream_position_quartiles': np.percentile(pos, [25, 50, 75]).tolist()}
    (OUT/'causal_history_results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[done] {time.perf_counter()-tick:.1f}s', flush=True)


if __name__ == '__main__':
    main()
