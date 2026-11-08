"""E17 单窗 878 维特征（6 s，250 Hz）。

复制自（数值逐行一致）：
- src/cogstate.py: BANDS
- src/preprocessing.py: PreprocessingConfig, _validate_config, sos_filters, causal_filter, quality_metrics, QUALITY_NAMES
- test/08_diagnostics/nonstationary_ablation.py: check_window, wavelet_features, spectrogram,
  timefreq_features, feature_blocks（去掉了结果未被使用的 raw44/quality53 两个 causal_features 分支）
- test/03_deep/highdim_deep_ablation.py: STAT_NAMES, statistics, nonlinear, highdim_features
- test/01_features/emg_v2.py: FS, FINE, features_v2
- test/17_compliant/streaming.py: window_features
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pywt
from scipy.signal import butter, sosfilt, iirnotch, tf2sos, welch, stft, hilbert, find_peaks
from scipy.spatial import cKDTree

BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))


# ---------------------------------------------------------------- src/preprocessing.py
@dataclass(frozen=True)
class PreprocessingConfig:
    sfreq: float = 250.0
    low_hz: float = 1.0
    high_hz: float = 45.0
    mains_hz: Optional[float] = 50.0
    mains_q: float = 30.0
    filter_order: int = 4
    quality_mode: str = "keep"  # keep, features, or reject
    amplitude_limit: float = 500.0
    peak_to_peak_limit: float = 1000.0
    line_flat_tolerance: float = 1e-12


QUALITY_NAMES = (
    "amplitude_max", "peak_to_peak", "rms", "flat_fraction", "diff_energy",
    "spike_fraction", "low_freq_drift", "high_freq_ratio", "nan_fraction",
)


def _validate_config(c):
    if c.low_hz <= 0 or c.high_hz >= c.sfreq / 2 or c.low_hz >= c.high_hz:
        raise ValueError("invalid causal band-pass frequencies")
    if c.mains_hz is not None and not (0 < c.mains_hz < c.sfreq / 2):
        raise ValueError("mains_hz must be below Nyquist")
    if c.quality_mode not in {"keep", "features", "reject"}:
        raise ValueError("quality_mode must be keep, features, or reject")


def sos_filters(config: PreprocessingConfig):
    _validate_config(config)
    sos = [butter(config.filter_order, [config.low_hz, config.high_hz], btype="bandpass",
                  fs=config.sfreq, output="sos")]
    if config.mains_hz is not None:
        b, a = iirnotch(config.mains_hz, config.mains_q, fs=config.sfreq)
        sos.append(tf2sos(b, a))
    return np.concatenate(sos, axis=0)


def causal_filter(x, config=PreprocessingConfig(), zi=None):
    """Causal SOS filtering. If zi is supplied, returns (y, state)."""
    x = np.asarray(x, dtype=float).ravel()
    if not np.isfinite(x).all():
        raise ValueError("signal contains NaN or infinite values")
    sos = sos_filters(config)
    if zi is None:
        y = sosfilt(sos, x)
        return y
    zi = np.asarray(zi, dtype=float)
    y, state = sosfilt(sos, x, zi=zi)
    return y, state


def quality_metrics(x, sfreq=250.0, config=PreprocessingConfig()):
    """Compute finite-window quality metrics without labels or future samples."""
    raw = np.asarray(x, dtype=float).ravel()
    nan_fraction = float(1.0 - np.isfinite(raw).mean()) if raw.size else 1.0
    z = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    if not len(z):
        return {k: float("nan") for k in QUALITY_NAMES}
    d = np.diff(z, prepend=z[0])
    med = float(np.median(z))
    mad = float(np.median(np.abs(z - med)) * 1.4826 + 1e-12)
    spike_fraction = float(np.mean(np.abs(z - med) > 8.0 * mad))
    nper = min(len(z), max(32, int(2 * sfreq)))
    f, p = welch(z - np.mean(z), fs=sfreq, nperseg=nper, noverlap=0)
    total = float(np.trapezoid(p[(f >= 1) & (f < 45)], f[(f >= 1) & (f < 45)]) + 1e-12)
    low = float(np.trapezoid(p[(f >= 1) & (f < 4)], f[(f >= 1) & (f < 4)]))
    high = float(np.trapezoid(p[(f >= 30) & (f < 45)], f[(f >= 30) & (f < 45)]))
    return {
        "amplitude_max": float(np.max(np.abs(z))),
        "peak_to_peak": float(np.ptp(z)),
        "rms": float(np.sqrt(np.mean(z ** 2))),
        "flat_fraction": float(np.mean(np.abs(d) <= config.line_flat_tolerance)),
        "diff_energy": float(np.mean(d ** 2)),
        "spike_fraction": spike_fraction,
        "low_freq_drift": float(low / total),
        "high_freq_ratio": float(high / total),
        "nan_fraction": nan_fraction,
    }


# ---------------------------------------------------------------- nonstationary_ablation.py
def check_window(x, fs):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) != round(6 * fs) or not np.isfinite(x).all():
        raise ValueError('requires exactly six seconds of finite single-channel samples')
    return x


def wavelet_features(filtered, fs=250.):
    x = check_window(filtered, fs)
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
    return {'wavelet': wavelet_features(filtered, fs),
            'timefreq': timefreq_features(filtered, fs),
            'quality': np.array([q[n] for n in QUALITY_NAMES])}


# ---------------------------------------------------------------- highdim_deep_ablation.py
STAT_NAMES = ('mean', 'std', 'skew', 'kurtosis', 'trend', 'q10', 'q25', 'q50', 'q75', 'q90', 'min', 'max', 'entropy')


def statistics(x):
    x = np.asarray(x, dtype=float)
    sd = x.std()
    z = (x - x.mean()) / sd if sd > 1e-15 else np.zeros_like(x)
    t = np.linspace(-.5, .5, len(x))
    p = np.abs(x) / max(np.abs(x).sum(), 1e-30)
    return [x.mean(), sd, np.mean(z**3), np.mean(z**4)-3 if sd > 1e-15 else 0,
            np.dot(t, x) / max(np.dot(t, t), 1e-30), *np.quantile(x, [.1, .25, .5, .75, .9]),
            x.min(), x.max(), -np.sum(p*np.log(np.maximum(p, 1e-30)))/max(np.log(len(x)), 1)]


def nonlinear(x):
    x = (x-x.mean()) / max(x.std(), 1e-15)
    d, dd = np.diff(x), np.diff(x, 2)
    mobility = np.sqrt(np.var(d) / max(np.var(x), 1e-15))
    complexity = np.sqrt(np.var(dd)/max(np.var(d), 1e-15))/max(mobility, 1e-15)
    v = x[::5]
    counts = []
    for m in (2, 3):
        emb = np.lib.stride_tricks.sliding_window_view(v, m)[:len(v)-2]
        counts.append((cKDTree(emb).count_neighbors(cKDTree(emb), .2, p=np.inf)-len(emb))/2)
    sampen = -np.log((counts[1]+1)/(counts[0]+1))
    patterns = np.argsort(np.lib.stride_tricks.sliding_window_view(v, 3), axis=1, kind='stable')
    _, count = np.unique(patterns, axis=0, return_counts=True)
    p = count/count.sum()
    ac = [np.dot(x[:-lag], x[lag:])/max(np.dot(x, x), 1e-15) for lag in (1, 5, 25, 50, 125)]
    return [np.var(x), mobility, complexity, sampen, -np.sum(p*np.log(p))/np.log(6), np.mean(x[:-1]*x[1:] < 0), *ac]


def highdim_features(x, fs=250.):
    x = check_window(x, fs)
    b = feature_blocks(x, fs)
    values = list(np.r_[b['wavelet'], b['timefreq'], b['quality']])
    filtered = causal_filter(x, PreprocessingConfig(sfreq=fs))

    def add(row):
        values.extend(statistics(row))
    for seconds in (.5, 1., 2.):
        n = round(seconds*fs)
        f, _, z = stft(filtered, fs=fs, nperseg=n, noverlap=n//2, boundary=None, padded=False)
        bp = np.array([np.sum(abs(z[(f >= lo) & (f < hi)])**2, axis=0) for lo, hi in BANDS])
        for matrix in (np.log(np.maximum(bp, 1e-30)), bp/np.maximum(bp.sum(axis=0), 1e-30)):
            for row in matrix:
                add(row)
    for wave in ('db4', 'sym5', 'coif2'):
        for c in pywt.wavedec(filtered, wave, mode='zero', level=5):
            add(c)
    for lo, hi in BANDS:
        band_signal = sosfilt(butter(3, [lo, hi], fs=fs, btype='bandpass', output='sos'), filtered)
        analytic = hilbert(band_signal)
        add(np.abs(analytic))
        frequency = np.diff(np.unwrap(np.angle(analytic)))*fs/(2*np.pi)
        add(np.clip(frequency, -fs/2, fs/2))
    for signal in (filtered, x):
        values.extend(nonlinear(signal))
    result = np.asarray(values)
    if not np.isfinite(result).all():
        raise ValueError('nonfinite feature')
    return result


# ---------------------------------------------------------------- emg_v2.py
FS = 250.
FINE = ((40, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 100), (100, 120))


def features_v2(x):
    f, p = welch(x, FS, nperseg=500, noverlap=250)
    tot = np.trapezoid(p[(f >= 1) & (f < 120)], f[(f >= 1) & (f < 120)]) + 1e-20
    pk, rl = [], []
    for lo, hi in FINE:
        v = np.trapezoid(p[(f >= lo) & (f < hi)], f[(f >= lo) & (f < hi)])
        pk.append(np.log(v+1e-20)); rl.append(v/tot)
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


# ---------------------------------------------------------------- streaming.py
def window_features(x, fs):
    """单个 6 s 窗的 878 维特征。"""
    x = np.asarray(x, dtype=float)
    f = np.concatenate([highdim_features(x, fs), features_v2(x)]).astype(np.float32)
    return np.nan_to_num(f)
