from __future__ import annotations

import csv
import math
import re
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.signal import welch

FS = 250
EPS = 1e-12
BANDS = (
    (0.5, 2.0),
    (2.0, 4.0),
    (4.0, 6.0),
    (6.0, 8.0),
    (8.0, 10.0),
    (10.0, 13.0),
    (13.0, 16.0),
    (16.0, 20.0),
    (20.0, 25.0),
    (25.0, 30.0),
    (30.0, 35.0),
    (35.0, 40.0),
    (40.0, 45.0),
)


@dataclass(frozen=True)
class DecoderConfig:
    window_s: float = 4.0
    stride_s: float = 1.0
    welch_s: float = 2.0
    fs: int = FS

    def __post_init__(self) -> None:
        if not 0 < self.window_s <= 6:
            raise ValueError("窗口必须大于 0 且不超过 6 秒")
        if not 0 < self.stride_s <= self.window_s:
            raise ValueError("步长必须大于 0 且不超过窗口长度")
        if not 0 < self.welch_s <= self.window_s:
            raise ValueError("Welch 分段不能超过窗口长度")
        for value in (self.window_s, self.stride_s, self.welch_s):
            if not math.isclose(value * self.fs, round(value * self.fs)):
                raise ValueError("时间参数必须能转换为整数采样点")

    @property
    def window_samples(self) -> int:
        return round(self.window_s * self.fs)

    @property
    def stride_samples(self) -> int:
        return round(self.stride_s * self.fs)

    @property
    def welch_samples(self) -> int:
        return round(self.welch_s * self.fs)

    def to_dict(self) -> dict:
        return {
            "window_s": self.window_s,
            "stride_s": self.stride_s,
            "welch_s": self.welch_s,
            "fs": self.fs,
        }

    @classmethod
    def from_dict(cls, values: dict) -> "DecoderConfig":
        return cls(**values)


def parse_recording_start(path: str | Path) -> datetime:
    name = Path(path).name
    match = re.search(
        r"(\d{8})_(\d{6})\.(\d{3})_UTC([+-])(\d{2})(\d{2})", name
    )
    if match is None:
        raise ValueError(f"无法从文件名解析毫秒级起始时间: {name}")
    date, clock, millis, sign, hours, minutes = match.groups()
    offset = timedelta(hours=int(hours), minutes=int(minutes))
    if sign == "-":
        offset = -offset
    return datetime.strptime(date + clock, "%Y%m%d%H%M%S").replace(
        microsecond=int(millis) * 1000, tzinfo=timezone(offset)
    )


def read_events(path: str | Path) -> list[dict]:
    events = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            events.append(
                {
                    "time": datetime.fromisoformat(row["local_time"]),
                    "event": row["event"],
                    "phase": row["phase"],
                    "key": row.get("key", ""),
                }
            )
    return sorted(events, key=lambda item: item["time"])


def pair_trials(events: list[dict]) -> list[dict]:
    trials = []
    active = None
    for event in events:
        if event["event"] == "trial_start":
            if active is not None:
                raise ValueError("发现嵌套的 trial_start")
            if event["phase"] not in {"relax", "task"}:
                raise ValueError(f"未知 phase: {event['phase']}")
            active = event
        elif event["event"] == "trial_end":
            if active is None:
                raise ValueError("trial_end 没有对应的 trial_start")
            if event["phase"] != active["phase"]:
                raise ValueError("trial 起止 phase 不一致")
            if event["time"] <= active["time"]:
                raise ValueError("trial 时长必须为正")
            trials.append(
                {
                    "start": active["time"],
                    "end": event["time"],
                    "phase": active["phase"],
                }
            )
            active = None
    if active is not None:
        raise ValueError("最后一条 trial_start 没有配对结束事件")
    return trials


def labels_from_trials(
    trials: list[dict], start_time: datetime, n_samples: int, fs: int = FS
) -> np.ndarray:
    labels = np.full(n_samples, -1, dtype=np.int8)
    for trial in trials:
        start_s = (trial["start"] - start_time).total_seconds()
        end_s = (trial["end"] - start_time).total_seconds()
        start = max(0, int(np.ceil(start_s * fs - 1e-9)))
        end = min(n_samples, int(np.ceil(end_s * fs - 1e-9)))
        if end <= 0 or start >= n_samples:
            continue
        if end <= start:
            raise ValueError("trial 与 BDF 时间轴不一致")
        target = 0 if trial["phase"] == "relax" else 1
        overlap = labels[start:end]
        if np.any((overlap >= 0) & (overlap != target)):
            raise ValueError("发现重叠且标签冲突的 trial")
        labels[start:end] = target
    return labels


def load_bdf(path: str | Path) -> dict:
    import mne

    raw = mne.io.read_raw_bdf(path, preload=True, verbose="ERROR")
    fs = int(round(raw.info["sfreq"]))
    signal_uv = raw.get_data(picks=[0])[0].astype(np.float64) * 1e6
    valid = np.isfinite(signal_uv)
    bad_intervals = []
    for annotation in raw.annotations:
        if str(annotation["description"]).strip().upper() != "BAD_ACQ_SKIP":
            continue
        onset = float(annotation["onset"])
        duration = float(annotation["duration"])
        start = max(0, int(np.floor(onset * fs + 1e-9)))
        end = min(len(signal_uv), int(np.ceil((onset + duration) * fs - 1e-9)))
        if end > start:
            valid[start:end] = False
            bad_intervals.append((start, end))
    return {
        "signal_uv": signal_uv,
        "valid": valid,
        "fs": fs,
        "n_samples": len(signal_uv),
        "n_channels": len(raw.ch_names),
        "channel_names": list(raw.ch_names),
        "bad_intervals": bad_intervals,
        "start_time": parse_recording_start(path),
    }


def feature_names() -> list[str]:
    names = []
    for low, high in BANDS:
        names.append(f"log_power_{low:g}_{high:g}")
    for low, high in BANDS:
        names.append(f"log_relative_{low:g}_{high:g}")
    names.extend(
        [
            "log_variance",
            "log_line_length",
            "log_robust_range",
            "hjorth_mobility",
            "hjorth_complexity",
            "zero_crossing_rate",
            "spectral_centroid",
            "spectral_entropy",
            "alpha_peak_frequency",
            "log_theta_alpha_ratio",
            "log_beta_alpha_ratio",
        ]
    )
    return names


def _band_power(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    upper_inclusive = math.isclose(high, BANDS[-1][1])
    mask = (freqs >= low) & ((freqs <= high) if upper_inclusive else (freqs < high))
    return float(np.sum(psd[mask]))


def extract_features(window_uv: np.ndarray, config: DecoderConfig) -> np.ndarray:
    x = np.asarray(window_uv, dtype=np.float64)
    if x.shape != (config.window_samples,):
        raise ValueError(f"特征窗口必须为 {config.window_samples} 个采样点")
    if not np.all(np.isfinite(x)):
        raise ValueError("特征窗口包含非有限值")

    centered = x - np.mean(x)
    dx = np.diff(centered)
    ddx = np.diff(dx)
    var_x = float(np.var(centered))
    var_dx = float(np.var(dx))
    var_ddx = float(np.var(ddx))
    mobility = math.sqrt(var_dx / (var_x + EPS))
    complexity = math.sqrt(var_ddx / (var_dx + EPS)) / (mobility + EPS)
    robust_range = float(np.percentile(x, 95) - np.percentile(x, 5))
    zero_crossing = float(np.mean(centered[:-1] * centered[1:] < 0))

    nperseg = config.welch_samples
    freqs, psd = welch(
        centered,
        fs=config.fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
    )
    powers = np.array(
        [_band_power(freqs, psd, low, high) for low, high in BANDS], dtype=np.float64
    )
    total = float(np.sum(powers))
    absolute = np.log10(powers + EPS)
    relative = np.log10((powers + EPS) / (total + EPS))

    spectral_mask = (freqs >= BANDS[0][0]) & (freqs <= BANDS[-1][1])
    spectral_p = psd[spectral_mask]
    spectral_f = freqs[spectral_mask]
    probabilities = spectral_p / (np.sum(spectral_p) + EPS)
    centroid = float(np.sum(spectral_f * probabilities))
    entropy = float(-np.sum(probabilities * np.log(probabilities + EPS)) / np.log(len(probabilities)))
    alpha_mask = (freqs >= 8) & (freqs <= 13)
    alpha_peak = float(freqs[alpha_mask][np.argmax(psd[alpha_mask])])
    theta = float(np.sum(powers[2:4]))
    alpha = float(np.sum(powers[4:6]))
    beta = float(np.sum(powers[6:10]))

    features = np.concatenate(
        [
            absolute,
            relative,
            np.array(
                [
                    math.log10(var_x + EPS),
                    math.log10(float(np.mean(np.abs(dx))) + EPS),
                    math.log10(robust_range + EPS),
                    mobility,
                    complexity,
                    zero_crossing,
                    centroid,
                    entropy,
                    alpha_peak,
                    math.log10((theta + EPS) / (alpha + EPS)),
                    math.log10((beta + EPS) / (alpha + EPS)),
                ]
            ),
        ]
    )
    return features.astype(np.float32)


def extract_record_features(
    signal_uv: np.ndarray,
    valid: np.ndarray,
    labels: np.ndarray,
    config: DecoderConfig,
) -> dict:
    n_samples = len(signal_uv)
    endpoints = np.arange(config.window_samples - 1, n_samples, config.stride_samples)
    labeled = labels[endpoints] >= 0
    bad = ~valid
    bad_cumsum = np.concatenate([[0], np.cumsum(bad, dtype=np.int64)])
    starts = endpoints - config.window_samples + 1
    bad_counts = bad_cumsum[endpoints + 1] - bad_cumsum[starts]
    good = (bad_counts == 0) & labeled
    pure = np.zeros(len(endpoints), dtype=bool)
    for index in np.flatnonzero(good):
        start = starts[index]
        end = endpoints[index]
        pure[index] = bool(np.all(labels[start : end + 1] == labels[end]))

    selected = np.flatnonzero(good)
    x_features = np.empty((len(selected), len(feature_names())), dtype=np.float32)
    for output_index, index in enumerate(selected):
        start = starts[index]
        end = endpoints[index] + 1
        x_features[output_index] = extract_features(signal_uv[start:end], config)
    return {
        "X": x_features,
        "y": labels[endpoints[selected]].astype(np.int8),
        "pure": pure[selected],
        "endpoints": endpoints[selected],
    }


def discover_records(data_dir: str | Path) -> list[tuple[str, Path, Path]]:
    root = Path(data_dir)
    records = []
    for bdf_path in sorted(root.glob("*/*.bdf")):
        record_id = bdf_path.parent.name
        event_path = bdf_path.parent / f"{record_id}_events.csv"
        if not event_path.exists():
            raise FileNotFoundError(f"缺少事件文件: {event_path}")
        records.append((record_id, bdf_path, event_path))
    if not records:
        raise FileNotFoundError(f"未在 {root} 找到编号目录中的 BDF 文件")
    return records


def build_feature_dataset(data_dir: str | Path, config: DecoderConfig) -> tuple[dict, list[dict]]:
    xs, ys, pures, groups, all_endpoints = [], [], [], [], []
    summaries = []
    for record_id, bdf_path, event_path in discover_records(data_dir):
        record = load_bdf(bdf_path)
        if record["fs"] != config.fs:
            raise ValueError(f"{record_id} 采样率 {record['fs']} 与配置不一致")
        trials = pair_trials(read_events(event_path))
        labels = labels_from_trials(
            trials, record["start_time"], record["n_samples"], config.fs
        )
        features = extract_record_features(
            record["signal_uv"], record["valid"], labels, config
        )
        xs.append(features["X"])
        ys.append(features["y"])
        pures.append(features["pure"])
        groups.append(np.full(len(features["y"]), record_id))
        all_endpoints.append(features["endpoints"])
        relax_s = float(np.sum(labels == 0) / config.fs)
        task_s = float(np.sum(labels == 1) / config.fs)
        summaries.append(
            {
                "record": record_id,
                "samples": record["n_samples"],
                "duration_s": record["n_samples"] / config.fs,
                "channels": record["n_channels"],
                "sampling_rate_hz": config.fs,
                "trials": len(trials),
                "relax_s": relax_s,
                "task_s": task_s,
                "task_fraction": task_s / (relax_s + task_s),
                "bad_intervals": len(record["bad_intervals"]),
                "bad_s": float(np.sum(~record["valid"]) / config.fs),
                "usable_windows": len(features["y"]),
                "pure_windows": int(np.sum(features["pure"])),
                "mean_abs_uv": float(np.mean(np.abs(record["signal_uv"][record["valid"]]))),
            }
        )
        print(
            f"{record_id}: {record['n_samples']} samples, "
            f"{len(features['y'])} windows, {len(record['bad_intervals'])} bad intervals",
            flush=True,
        )
    dataset = {
        "X": np.concatenate(xs),
        "y": np.concatenate(ys),
        "pure": np.concatenate(pures),
        "groups": np.concatenate(groups),
        "endpoints": np.concatenate(all_endpoints),
    }
    return dataset, summaries


def model_scores(model, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(features))[:, 1]
    return expit(np.asarray(model.decision_function(features)).ravel())


class CausalStreamDecoder:
    """输入单位必须为微伏；输出分数越大越倾向 task。"""

    def __init__(self, model, threshold: float, config: DecoderConfig):
        self.model = model
        self.threshold = float(threshold)
        self.config = config
        self._samples = deque(maxlen=config.window_samples)
        self._valid = deque(maxlen=config.window_samples)
        self.n_received = 0

    def push(self, samples_uv, valid=None) -> list[dict]:
        chunk = np.asarray(samples_uv, dtype=np.float64).reshape(-1)
        if valid is None:
            quality = np.isfinite(chunk)
        else:
            quality = np.asarray(valid, dtype=bool).reshape(-1) & np.isfinite(chunk)
        if len(chunk) != len(quality):
            raise ValueError("valid 与 samples_uv 长度不一致")
        outputs = []
        for value, is_valid in zip(chunk, quality):
            sample_index = self.n_received
            self.n_received += 1
            self._samples.append(float(value))
            self._valid.append(bool(is_valid))
            due = (
                self.n_received >= self.config.window_samples
                and (sample_index - (self.config.window_samples - 1))
                % self.config.stride_samples
                == 0
            )
            if not due:
                continue
            window_valid = bool(np.all(self._valid))
            if window_valid:
                features = extract_features(np.asarray(self._samples), self.config)[None, :]
                score = float(model_scores(self.model, features)[0])
            else:
                score = 0.5
            outputs.append(
                {
                    "sample_index": sample_index,
                    "time_s": sample_index / self.config.fs,
                    "score_task": score,
                    "prediction": "task" if score >= self.threshold else "relax",
                    "valid": window_valid,
                }
            )
        return outputs
