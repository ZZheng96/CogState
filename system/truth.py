"""独立真值：只由 BDF（BAD_ACQ_SKIP 注释）与 events.csv 生成，与任何方案无关。

决策时刻 t（秒，相对 BDF 首样本）的标签：
  6 s 因果窗口 [t-6, t] 完全落在单个 trial 内且不与缺失段相交 -> 该 trial 的 phase（task=1, relax=0）；
  否则 -1（不参与评估）。
trial 与缺失段按被试缓存到 system/cache/truth/<source>/<sid>.json（只解析一次 BDF）。
"""
from __future__ import annotations

import json

import numpy as np

from . import CACHE, dataset

WIN_S = 6.0
STEP_S = 1.0


def meta(source: str, sid: str) -> dict:
    """{"trials": [(start, end, label)], "bad": [(a, b)], "duration": 秒}"""
    bdf, csv = dataset.paths(source, sid)
    f = CACHE / "truth" / source / f"{sid}.json"
    if f.exists():
        m = json.loads(f.read_text(encoding="utf-8"))
        if m.get("bdf") == bdf.name and m.get("csv") == csv.name:
            return m
    import mne
    from cogstate import load_trials
    raw = mne.io.read_raw_bdf(str(bdf), preload=False, verbose="ERROR")
    fs = float(raw.info["sfreq"])
    bad = [(float(o), float(o + d)) for o, d, s in zip(raw.annotations.onset, raw.annotations.duration,
                                                         raw.annotations.description)
           if str(s).strip().upper() == "BAD_ACQ_SKIP"]
    trials = [(float(t.start), float(t.end), int(t.label)) for t in load_trials(str(csv), str(bdf))]
    m = {"bdf": bdf.name, "csv": csv.name, "fs": fs, "duration": raw.n_times / fs, "trials": trials, "bad": bad}
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(m), encoding="utf-8")
    return m


def labels(source: str, sid: str, t: np.ndarray) -> np.ndarray:
    m = meta(source, sid)
    t = np.asarray(t, dtype=np.float64)
    y = np.full(len(t), -1, dtype=np.int8)
    for a, b, lab in m["trials"]:
        y[(t - WIN_S >= a - 1e-6) & (t <= b + 1e-6)] = lab
    for a, b in m["bad"]:
        y[(t - WIN_S < b) & (t > a)] = -1
    return y


def reference_grid(source: str, sid: str) -> tuple[np.ndarray, np.ndarray]:
    """1 s 参考网格上的全部可评估时刻（用于覆盖率）。"""
    m = meta(source, sid)
    t = np.arange(WIN_S, m["duration"] + 1e-9, STEP_S)
    y = labels(source, sid, t)
    return t[y >= 0], y[y >= 0]
