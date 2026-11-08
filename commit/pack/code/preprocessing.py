"""Leakage-safe single-channel causal preprocessing and window quality tools."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi, iirnotch, tf2sos, welch, resample_poly


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


def initial_filter_state(config=PreprocessingConfig()):
    return np.zeros((len(sos_filters(config)), 2), dtype=float)


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


@dataclass
class QualityThresholds:
    limits: dict
    quantiles: tuple = (0.01, 0.99)

    @classmethod
    def fit(cls, metric_rows, quantiles=(0.01, 0.99)):
        """Fit unsupervised limits on training-fold windows only."""
        rows = list(metric_rows)
        if not rows:
            raise ValueError("cannot fit quality thresholds on no windows")
        limits = {}
        for name in QUALITY_NAMES:
            vals = np.asarray([r[name] for r in rows], dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals):
                limits[name] = tuple(float(v) for v in np.quantile(vals, quantiles))
        return cls(limits, quantiles)

    def flag(self, metrics):
        reasons = []
        for name, (lo, hi) in self.limits.items():
            value = float(metrics.get(name, np.nan))
            if not np.isfinite(value) or value < lo or value > hi:
                reasons.append(name)
        return bool(reasons), reasons


def apply_quality(metrics, mode="keep", thresholds=None, config=PreprocessingConfig()):
    if mode not in {"keep", "features", "reject"}:
        raise ValueError("invalid quality mode")
    fixed_bad = (metrics.get("nan_fraction", 1.0) > 0 or
                 metrics.get("amplitude_max", 0.0) > config.amplitude_limit or
                 metrics.get("peak_to_peak", 0.0) > config.peak_to_peak_limit)
    learned_bad, reasons = thresholds.flag(metrics) if thresholds is not None else (False, [])
    bad = fixed_bad or learned_bad
    if fixed_bad:
        reasons = ["fixed_amplitude_or_nan"] + reasons
    return {"keep": True if mode != "reject" else not bad,
            "bad": bad, "reasons": reasons,
            "features": np.asarray([metrics[n] for n in QUALITY_NAMES], dtype=np.float32)}


def preprocess_window(x, config=PreprocessingConfig()):
    x = np.asarray(x, dtype=float).ravel()
    metrics = quality_metrics(x, config.sfreq, config)
    result = apply_quality(metrics, config.quality_mode, None, config)
    if result["bad"] and config.quality_mode == "reject":
        return None, metrics, result
    return causal_filter(np.nan_to_num(x), config), metrics, result


class StreamingPreprocessor:
    """Chunk-safe causal processor; output is identical to latest-window processing."""
    def __init__(self, config=PreprocessingConfig(), window_seconds=6.0):
        self.config = config
        self.window = int(round(window_seconds * config.sfreq))
        self.raw = np.empty(0, dtype=float)
        self._state = initial_filter_state(config)

    def reset(self):
        self.raw = np.empty(0, dtype=float)
        self._state = initial_filter_state(self.config)

    def update(self, samples):
        samples = np.asarray(samples, dtype=float).ravel()
        self.raw = np.concatenate([self.raw, samples])[-self.window:]
        if len(self.raw) < self.window:
            return None
        y, metrics, quality = preprocess_window(self.raw, self.config)
        return {"signal": y, "metrics": metrics, "quality": quality}


def resample_offline_polyphase(x, up=4, down=5):
    """Offline complete-window anti-aliased polyphase resampling only."""
    x = np.asarray(x, dtype=float).ravel()
    if not np.isfinite(x).all():
        raise ValueError("signal contains NaN or infinite values")
    return resample_poly(x, up, down)


def resample_causal_fixed_delay(x, up=4, down=5, state=None):
    """Streaming-compatible FIR polyphase step with explicit state contract.

    This conservative implementation keeps a causal input history and emits only
    samples whose FIR support is available; it never uses future samples. It is
    intentionally separate from offline resample_poly.
    """
    x = np.asarray(x, dtype=float).ravel()
    if not np.isfinite(x).all():
        raise ValueError("signal contains NaN or infinite values")
    history = np.empty(0) if state is None else np.asarray(state, dtype=float).ravel()
    joined = np.concatenate([history, x])
    y = resample_poly(joined, up, down)
    keep = max(0, len(y) - len(resample_poly(history, up, down)))
    return y[-keep:] if keep else np.empty(0), joined[-max(64, down * 16):]
