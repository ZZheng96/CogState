from __future__ import annotations
import glob, os, re
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt, welch

FS = 250.0
BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))

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


def _bandpowers(f, p):
    return np.asarray([np.trapezoid(p[(f >= lo) & (f < hi)], f[(f >= lo) & (f < hi)]) for lo, hi in BANDS], dtype=float)


def causal_features(x, sfreq=FS):
    x = np.asarray(x, dtype=float).ravel()
    if len(x) < int(sfreq) or not np.isfinite(x).all():
        return None
    sos = butter(4, [1 / (sfreq / 2), 45 / (sfreq / 2)], btype="band", output="sos")
    y = sosfilt(sos, x)
    f, p = welch(y, fs=sfreq, nperseg=min(len(y), int(2 * sfreq)), noverlap=0, detrend="constant")
    total = float(np.trapezoid(p[(f >= 1) & (f < 45)], f[(f >= 1) & (f < 45)]) + 1e-12)
    bp = _bandpowers(f, p)
    rel = bp / total
    feats = list(np.log(bp + 1e-12)) + list(rel)
    feats += [float(np.log((bp[i] + 1e-12) / (bp[j] + 1e-12))) for i, j in ((1, 2), (2, 3), (3, 4), (0, 3))]
    ps = p[(f >= 1) & (f < 45)] + 1e-12
    feats += [float(-np.sum((ps / ps.sum()) * np.log(ps / ps.sum()))), float(np.sum(f[(f >= 1) & (f < 45)] * ps) / ps.sum())]
    feats += [float(np.sqrt(np.mean(y ** 2))), float(np.ptp(y)), float(np.mean(np.diff(y) ** 2)), float(np.trapezoid(p[(f >= 30) & (f < 45)], f[(f >= 30) & (f < 45)]) / total)]
    segments = np.array_split(y, 3)
    for seg in segments:
        fs, pspec = welch(seg, fs=sfreq, nperseg=min(len(seg), int(2 * sfreq)), noverlap=0, detrend="constant")
        sbp = _bandpowers(fs, pspec)
        st = float(np.sum(sbp) + 1e-12)
        feats += list(sbp / st) + [float(np.sqrt(np.mean(seg ** 2))), float(np.ptp(seg)), float(np.mean(np.diff(seg) ** 2))]
    return np.asarray(feats, dtype=np.float32)


def make_dataset(root, window=6.0, stride=6.0, limit_subjects=None):
    import mne
    if window > 6.0:
        raise ValueError("official causal window must be <= 6 seconds")
    X, y, groups, meta = [], [], [], []
    paths = sorted(glob.glob(os.path.join(root, "*", "*_events.csv")))
    if limit_subjects:
        paths = [p for p in paths if os.path.basename(os.path.dirname(p)) in limit_subjects]
    for ci, csv in enumerate(paths, 1):
        sid = os.path.basename(os.path.dirname(csv)); bdf = glob.glob(os.path.join(os.path.dirname(csv), "*.bdf"))[0]
        print(f"[{ci}/{len(paths)}] subject {sid}", flush=True)
        raw = mne.io.read_raw_bdf(bdf, preload=True, verbose="ERROR")
        sfreq = float(raw.info["sfreq"]); record_end = raw.n_times / sfreq
        trials = load_trials(csv, bdf); bad = bad_intervals(raw)
        for tr in trials:
            t = tr.start + window
            while t <= tr.end + 1e-6:
                a, z = t - window, t
                if valid_window(a, z, bad, record_end):
                    s, e = int(round(a * sfreq)), int(round(z * sfreq)); x = raw.get_data(start=s, stop=e)[0]
                    if len(x) == e - s:
                        feat = causal_features(x, sfreq)
                        if feat is not None:
                            X.append(feat); y.append(tr.label); groups.append(sid); meta.append({"subject": sid, "trial_id": tr.trial_id, "time": t, "label": tr.label})
                t += stride
    return np.asarray(X), np.asarray(y), np.asarray(groups), meta


def metrics(y, p, groups=None):
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    return {"bACC": float(balanced_accuracy_score(y, p >= .5)), "AUC": float(roc_auc_score(y, p))}


class StreamingInference:
    def __init__(self, model, sfreq=FS, window=6.0):
        self.model = model; self.sfreq = sfreq; self.window = int(window * sfreq); self.buf = np.empty(0)
    def update(self, samples):
        self.buf = np.concatenate([self.buf, np.asarray(samples, dtype=float).ravel()])[-self.window:]
        if len(self.buf) < self.window:
            return None
        return float(self.model.predict_proba(causal_features(self.buf, self.sfreq)[None, :])[0, 1])


def save_bundle(model, path):
    import joblib
    joblib.dump({"model": model, "fs": FS, "window": 6.0, "bands": BANDS}, path)


def load_bundle(path):
    import joblib
    return joblib.load(path)
