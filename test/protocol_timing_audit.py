from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_decoder import discover_records, pair_trials, parse_recording_start, read_events  # noqa: E402
from train import classification_metrics, make_folds  # noqa: E402

FS = 250
MAX_S = 2000
records = []
trial_rows = []
for record_id, bdf_path, event_path in discover_records(ROOT / "EEG_dataset"):
    start_time = parse_recording_start(bdf_path)
    trials = pair_trials(read_events(event_path))
    grid = np.full(MAX_S + 1, -1, dtype=np.int8)
    for trial_index, trial in enumerate(trials, start=1):
        start_s = (trial["start"] - start_time).total_seconds()
        end_s = (trial["end"] - start_time).total_seconds()
        target = int(trial["phase"] == "task")
        grid[max(0, int(np.ceil(start_s))):min(MAX_S + 1, int(np.ceil(end_s)))] = target
        trial_rows.append({
            "record": record_id,
            "trial": trial_index,
            "phase": trial["phase"],
            "start_s": start_s,
            "end_s": end_s,
            "duration_s": end_s - start_s,
        })
    endpoints = np.arange(0, MAX_S + 1, dtype=int)
    labeled = grid >= 0
    records.append({
        "record": record_id,
        "grid": grid,
        "seconds": endpoints[labeled],
        "y": grid[labeled],
        "phases": "".join("T" if trial["phase"] == "task" else "R" for trial in trials),
    })

trials_df = pd.DataFrame(trial_rows)
trials_df.to_csv(ROOT / "test" / "results" / "protocol_trials.csv", index=False)
print("Phase sequences:", trials_df.groupby("record")["phase"].apply(lambda x: "".join(x.str[0])).value_counts().to_dict())
print(trials_df.groupby(["trial", "phase"])[["start_s", "duration_s"]].agg(["mean", "std", "min", "max"]).round(3))

# 仅由训练记录的事件时间构建模板；测试记录只提供当前相对秒数。
groups = np.concatenate([np.full(len(item["y"]), item["record"]) for item in records])
y = np.concatenate([item["y"] for item in records])
seconds = np.concatenate([item["seconds"] for item in records])
folds = make_folds(groups, 6, 42)
scores = np.full(len(y), np.nan)
for test_records in folds:
    train = [item for item in records if item["record"] not in test_records]
    train_grid = np.stack([item["grid"] for item in train])
    valid = train_grid >= 0
    task = np.sum(train_grid == 1, axis=0)
    count = np.sum(valid, axis=0)
    template = np.divide(task, count, out=np.full(MAX_S + 1, 0.5), where=count > 0)
    test_mask = np.isin(groups, test_records)
    scores[test_mask] = template[seconds[test_mask]]

metrics = classification_metrics(y, scores)
print("Protocol-only grouped OOF:", metrics)
rows = []
for item in records:
    mask = groups == item["record"]
    rows.append({"record": item["record"], **classification_metrics(y[mask], scores[mask])})
record_metrics = pd.DataFrame(rows)
record_metrics.to_csv(ROOT / "test" / "results" / "protocol_template_record_metrics.csv", index=False)
pd.DataFrame({"record": groups, "second": seconds, "y": y, "score_task": scores}).to_csv(
    ROOT / "test" / "results" / "protocol_template_oof.csv", index=False
)
print(record_metrics[["bacc", "auc"]].describe().round(4))
