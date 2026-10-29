"""Six-second endpoint-only nonstationary ablation; never reads future windows."""
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import numpy as np
import pywt
from scipy.signal import stft
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cogstate import BANDS, causal_features, load_trials, bad_intervals, valid_window
from preprocessing import PreprocessingConfig, causal_filter, quality_metrics, QUALITY_NAMES

OUT = ROOT / 'test'
SCALES = ('A5', 'D5', 'D4', 'D3', 'D2', 'D1')
STATS = ('mean', 'std', 'slope', 'max', 'min')
WAVE_NAMES = [f'dwt_{kind}_{s}' for kind in ('log_energy', 'relative', 'entropy') for s in SCALES] + [f'dwt_log_ratio_{SCALES[i]}_{SCALES[i+1]}' for i in range(5)]
TF_NAMES = [f'stft_{kind}_{lo}_{hi}_{stat}' for kind in ('log_power', 'relative') for lo, hi in BANDS for stat in STATS]


def check_window(x, fs):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) != round(6 * fs) or not np.isfinite(x).all():
        raise ValueError('requires exactly six seconds of finite single-channel samples')
    return x


def wavelet_features(filtered, fs=250.):
    x = check_window(filtered, fs)
    # Zero extension is synthetic, not reflected or observed future EEG. Coefficients
    # near either edge include this boundary assumption; no coefficient is trimmed.
    coeff = pywt.wavedec(x, 'db4', mode='zero', level=5)
    energies = np.array([np.sum(c*c) for c in coeff])
    tiny = np.finfo(float).tiny
    entropy = []
    for c, e in zip(coeff, energies):
        p = c*c / max(e, tiny)
        entropy.append(-np.sum(p * np.log(np.maximum(p, tiny))) / np.log(len(c)))
    loge = np.log(np.maximum(energies, tiny))
    return np.r_[loge, energies / max(energies.sum(), tiny), entropy, np.diff(-loge)]


def spectrogram(filtered, fs):
    return stft(filtered, fs=fs, window='hann', nperseg=round(fs),
                noverlap=round(fs / 2), detrend='constant', boundary=None, padded=False)


def timefreq_features(filtered, fs=250.):
    x = check_window(filtered, fs)
    f, t, z = spectrogram(x, fs)
    power = np.abs(z)**2
    bp = np.array([np.sum(power[(f >= lo) & (f < hi)], axis=0) for lo, hi in BANDS])
    tiny = np.finfo(float).tiny
    relative = bp / np.maximum(bp.sum(axis=0), tiny)
    centered = t - t.mean()
    feats = []
    for matrix in (np.log(np.maximum(bp, tiny)), relative):
        for row in matrix:
            feats.extend((row.mean(), row.std(), np.dot(centered, row) / np.dot(centered, centered), row.max(), row.min()))
    return np.array(feats)


def feature_blocks(x, fs=250.):
    x = check_window(x, fs)
    cfg = PreprocessingConfig(sfreq=fs)
    filtered = causal_filter(x, cfg)
    q = quality_metrics(x, fs, cfg)
    # Match artifact_ablation's existing 53D comparator, including its second
    # bandpass inside causal_features. New wavelet/STFT branches filter once.
    return {'raw44': causal_features(x, fs),
            'quality53': np.r_[causal_features(filtered, fs), [q[n] for n in QUALITY_NAMES]],
            'wavelet': wavelet_features(filtered, fs),
            'timefreq': timefreq_features(filtered, fs),
            'quality': np.array([q[n] for n in QUALITY_NAMES])}


def extract():
    rows = {k: [] for k in ('raw44', 'quality53', 'wavelet', 'timefreq', 'quality')}
    labels, groups, meta, example = [], [], [], None
    paths = sorted((ROOT / 'EEG_dataset').glob('*/*_events.csv'))
    for i, csv in enumerate(paths, 1):
        bdf = next(csv.parent.glob('*.bdf'))
        raw = mne.io.read_raw_bdf(str(bdf), preload=True, verbose='ERROR')
        fs = float(raw.info['sfreq'])
        bad = bad_intervals(raw)
        for trial in load_trials(str(csv), str(bdf)):
            end = trial.start + 6.
            while end <= trial.end + 1e-6:
                start = end - 6.
                if valid_window(start, end, bad, raw.n_times / fs):
                    x = raw.get_data(start=round(start*fs), stop=round(end*fs))[0]
                    if len(x) == round(6*fs) and np.isfinite(x).all():
                        blocks = feature_blocks(x, fs)
                        for key in rows:
                            rows[key].append(blocks[key])
                        labels.append(trial.label)
                        groups.append(trial.subject)
                        meta.append({'subject': trial.subject, 'trial_id': trial.trial_id, 'end': end})
                        if example is None:
                            example = (x.copy(), fs)
                end += 6.
        raw.close()
        print(f'[extract {i}/{len(paths)}] subject={csv.parent.name} windows={len(labels)}', flush=True)
    return {k: np.asarray(v) for k, v in rows.items()}, np.array(labels), np.array(groups), meta, example


def score(y, p):
    return {'AUC': float(roc_auc_score(y, p)), 'bACC': float(balanced_accuracy_score(y, p >= .5))}


def evaluate(X, y, groups, selected=False):
    steps = [VarianceThreshold(), StandardScaler()]
    if selected:
        steps.append(SelectKBest(f_classif, k=30))
    steps.append(LogisticRegression(C=1., max_iter=2000, class_weight='balanced', random_state=0))
    pipeline = make_pipeline(*steps)
    oof = np.full(len(y), np.nan)
    folds = []
    start = time.perf_counter()
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    for fold, (tr, va) in enumerate(cv.split(X, y, groups), 1):
        assert not set(groups[tr]) & set(groups[va])
        tick = time.perf_counter()
        model = clone(pipeline).fit(X[tr], y[tr])
        oof[va] = model.predict_proba(X[va])[:, 1]
        row = {'fold': fold, 'subjects': sorted(set(groups[va])), 'n': len(va),
               'n_features_fitted': int(model[-1].n_features_in_),
               'seconds': time.perf_counter()-tick, **score(y[va], oof[va])}
        if selected:
            retained = np.flatnonzero(model[0].get_support())
            row['selected_input_indices'] = retained[model[2].get_support()].tolist()
        folds.append(row)
        print(f'  fold {fold}/5 AUC={row["AUC"]:.6f} bACC={row["bACC"]:.6f}', flush=True)
    subject = {s: score(y[groups == s], oof[groups == s]) for s in sorted(set(groups))}
    return {'n_features': X.shape[1], 'oof': score(y, oof), 'folds': folds,
            'fold_mean': {k: float(np.mean([v[k] for v in folds])) for k in ('AUC', 'bACC')},
            'subject_macro': {k: float(np.mean([v[k] for v in subject.values()])) for k in ('AUC', 'bACC')},
            'subjects': subject, 'seconds': time.perf_counter()-start}, oof


def main():
    outputs = ['nonstationary_results.json', 'nonstationary_comparison.png',
               'nonstationary_diagnostics.png', 'nonstationary_timefreq.png', 'nonstationary_oof.npz']
    for name in outputs:
        if (OUT / name).exists():
            raise FileExistsError(f'Refusing to overwrite existing result: {OUT / name}')
    tick = time.perf_counter()
    blocks, y, groups, meta, example = extract()
    extraction_seconds = time.perf_counter()-tick
    joint = np.column_stack([blocks[k] for k in ('wavelet', 'timefreq', 'quality')])
    variants = {'raw44': blocks['raw44'], 'quality53': blocks['quality53'],
                'wavelet_only': blocks['wavelet'], 'timefreq_only': blocks['timefreq'],
                'wavelet_timefreq_quality': joint, 'selected30_logistic': joint}
    result = {'protocol': {'window_seconds': 6, 'stride_seconds': 6, 'n_samples': len(y),
              'n_subjects': len(set(groups)), 'cv': 'StratifiedGroupKFold(5, shuffle=True, random_state=0)',
              'filter': 'window-reset causal SOS 1-45Hz order4 + 50Hz notch Q30',
              'wavelet': 'db4 wavedec level5 zero extension; endpoint output, boundary coefficients retained',
              'timefreq': 'Hann STFT 1s, hop0.5s, boundary=None, padded=False; log-power and relative bandpower mean/std/slope/max/min',
              'quality53': 'existing artifact comparator: causal_features(causal_filter(x)) + 9 raw quality metrics',
              'classifier': 'VarianceThreshold + StandardScaler + optional SelectKBest(f_classif,k=30) + balanced LogisticRegression(C=1,max_iter=2000)',
              'leakage': 'all transforms fit only on training fold; all signal features emitted at six-second endpoint',
              'polarization': 'not applicable: single channel has no 2D/spatial polarization or multichannel covariance',
              'hilbert': 'not implemented; baseline44 already includes spectral entropy and centroid',
              'selection_caveat': 'best of these outer CV comparisons is exploratory, not independent test accuracy',
              'pywavelets_version': pywt.__version__},
              'feature_names': {'wavelet': WAVE_NAMES, 'timefreq': TF_NAMES, 'joint': WAVE_NAMES+TF_NAMES+list(QUALITY_NAMES)},
              'extraction_seconds': extraction_seconds, 'variants': {}}
    predictions = {}
    with threadpool_limits(limits=1):
        for name, X in variants.items():
            print(f'[model] {name} shape={X.shape}', flush=True)
            result['variants'][name], predictions[name] = evaluate(X, y, groups, name == 'selected30_logistic')
    result['best_by_subject_macro_AUC'] = max(result['variants'], key=lambda n: result['variants'][n]['subject_macro']['AUC'])
    names = list(variants)
    ix = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(12, 5))
    for offset, metric in ((-.18, 'AUC'), (.18, 'bACC')):
        ax.bar(ix+offset, [result['variants'][n]['subject_macro'][metric] for n in names], .36, label='Subject macro '+metric)
    ax.axhline(.9, ls='--', color='red', label='0.9 target')
    ax.set(ylim=(.4, 1), xticks=ix, xticklabels=names)
    plt.xticks(rotation=18, ha='right'); ax.legend(); fig.tight_layout()
    fig.savefig(OUT / outputs[1], dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].imshow(np.corrcoef(joint[:, :73], rowvar=False), vmin=-1, vmax=1, cmap='coolwarm', aspect='auto')
    axes[0].set(title='Wavelet / STFT feature correlation (descriptive)', xlabel='Feature index', ylabel='Feature index')
    axes[1].plot(np.std(joint, axis=0)); axes[1].set(yscale='symlog', title='Raw feature standard deviation', xlabel='Joint feature index')
    fig.tight_layout(); fig.savefig(OUT / outputs[2], dpi=160); plt.close(fig)
    x, fs = example
    filtered = causal_filter(x, PreprocessingConfig(sfreq=fs))
    f, t, z = spectrogram(filtered, fs)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].plot(np.arange(len(x))/fs, x); axes[0].set(title=f'Real six-second window: {meta[0]}', ylabel='Recorded amplitude')
    im = axes[1].pcolormesh(t, f, 10*np.log10(np.maximum(abs(z)**2, np.finfo(float).tiny)), shading='auto')
    axes[1].set(ylim=(1, 45), xlabel='Seconds within window (STFT frame centers)', ylabel='Hz')
    fig.colorbar(im, ax=axes[1], label='dB power'); fig.tight_layout()
    fig.savefig(OUT / outputs[3], dpi=160); plt.close(fig)
    np.savez(OUT / outputs[4], y=y, groups=groups, **predictions)
    result['runtime_seconds'] = time.perf_counter()-tick
    with (OUT / outputs[0]).open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print('COMPLETE', result['best_by_subject_macro_AUC'], 'seconds', result['runtime_seconds'], flush=True)


if __name__ == '__main__':
    main()
