"""数据读取：BDF 信号与 events.csv 试次。

复制自：
- src/eeg_decoder.py: parse_recording_start, load_bdf
- src/cogstate.py: Trial, _file_start, load_trials, bad_intervals, valid_window
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Trial:
    subject: str
    trial_id: int
    start: float
    end: float
    label: int


def _file_start(path):
    name = os.path.basename(path)
    m = re.search(r"_(\d{8})_(\d{6})\.(\d{3})_UTC", name)
    if not m:
        raise ValueError(f"cannot parse BDF time: {name}")
    dt = datetime.strptime(m.group(1) + m.group(2) + m.group(3), "%Y%m%d%H%M%S%f")
    return dt.replace(tzinfo=timezone(timedelta(hours=8))).timestamp()


def load_trials(csv_path, bdf_path):
    ev = pd.read_csv(csv_path)
    required = {"local_time", "event", "phase"}
    missing = required.difference(ev.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing columns {sorted(missing)}")
    allowed = {"relax", "task"}
    ts = pd.to_datetime(ev.local_time, utc=True, errors="raise").astype("int64").to_numpy() / 1e9
    out, pending = [], None
    offset = _file_start(bdf_path)
    for i, row in ev.iterrows():
        phase = str(row.phase).strip()
        if phase not in allowed:
            raise ValueError(f"{csv_path}: unsupported phase {phase!r} at row {i + 2}")
        event = str(row.event).strip()
        if event == "trial_start":
            if pending is not None:
                raise ValueError(f"{csv_path}: trial_start before trial_end at row {i + 2}")
            pending = (float(ts[i]), phase)
        elif event == "trial_end":
            if pending is None:
                raise ValueError(f"{csv_path}: trial_end without trial_start at row {i + 2}")
            start, start_phase = pending
            if phase != start_phase:
                raise ValueError(f"{csv_path}: phase mismatch at row {i + 2}")
            if ts[i] <= start:
                raise ValueError(f"{csv_path}: non-positive trial at row {i + 2}")
            out.append(Trial(os.path.basename(os.path.dirname(csv_path)), len(out), start - offset, float(ts[i]) - offset, int(start_phase == "task")))
            pending = None
    if pending is not None:
        raise ValueError(f"{csv_path}: trial_start has no trial_end")
    return out


def bad_intervals(raw):
    return [(float(a), float(a + d)) for a, d, desc in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description) if "BAD_ACQ_SKIP" in str(desc)]


def valid_window(start, end, bad, record_end=None):
    record_end = float("inf") if record_end is None else float(record_end)
    return start >= 0 and end <= record_end and end > start and not any(start < b and end > a for a, b in bad)


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


def load_bdf(path: str | Path) -> dict:
    import mne

    # MNE 1.4 仍调用 NumPy 2.x 已移除的 in1d 别名。
    if not hasattr(np, "in1d"):
        np.in1d = np.isin
    raw = mne.io.read_raw_bdf(path, preload=True, verbose="ERROR")
    fs = int(round(raw.info["sfreq"]))
    signal_uv = raw.get_data(picks=[0])[0].astype(np.float64) * 1e6
    valid = np.isfinite(signal_uv)
    bad = []
    for annotation in raw.annotations:
        if str(annotation["description"]).strip().upper() != "BAD_ACQ_SKIP":
            continue
        onset = float(annotation["onset"])
        duration = float(annotation["duration"])
        start = max(0, int(np.floor(onset * fs + 1e-9)))
        end = min(len(signal_uv), int(np.ceil((onset + duration) * fs - 1e-9)))
        if end > start:
            valid[start:end] = False
            bad.append((start, end))
    return {
        "signal_uv": signal_uv,
        "valid": valid,
        "fs": fs,
        "n_samples": len(signal_uv),
        "n_channels": len(raw.ch_names),
        "channel_names": list(raw.ch_names),
        "bad_intervals": bad,
        "start_time": parse_recording_start(path),
    }
