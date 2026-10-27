from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_decoder import (  # noqa: E402
    CausalStreamDecoder,
    DecoderConfig,
    extract_features,
    extract_record_features,
    feature_names,
    labels_from_trials,
    model_scores,
    pair_trials,
    parse_recording_start,
)
from train import make_folds  # noqa: E402


class SimpleModel:
    def predict_proba(self, features):
        score = 1 / (1 + np.exp(-features[:, feature_names().index("log_variance")]))
        return np.column_stack([1 - score, score])


def signal(length_s=12, fs=250):
    t = np.arange(length_s * fs) / fs
    return np.sin(2 * np.pi * 8 * t) + 0.3 * np.sin(2 * np.pi * 20 * t)


def test_window_limit():
    with pytest.raises(ValueError):
        DecoderConfig(window_s=6.004)
    assert DecoderConfig(window_s=6).window_samples == 1500


def test_filename_millisecond_timezone():
    start = parse_recording_start("001_20261017_170137.984_UTC+0800.bdf")
    assert start == datetime(2026, 10, 17, 17, 1, 37, 984000, tzinfo=timezone(timedelta(hours=8)))


def test_trial_pairing_and_sample_alignment():
    start = datetime(2026, 10, 17, 17, 1, 37, 984000, tzinfo=timezone(timedelta(hours=8)))
    events = [
        {"time": start + timedelta(seconds=1), "event": "trial_start", "phase": "relax"},
        {"time": start + timedelta(seconds=2), "event": "trial_end", "phase": "relax"},
        {"time": start + timedelta(seconds=2), "event": "trial_start", "phase": "task"},
        {"time": start + timedelta(seconds=3.5), "event": "trial_end", "phase": "task"},
    ]
    trials = pair_trials(events)
    labels = labels_from_trials(trials, start, 1000, fs=250)
    assert np.all(labels[250:500] == 0)
    assert np.all(labels[500:875] == 1)
    assert np.all(labels[:250] == -1)
    assert np.all(labels[875:] == -1)


def test_unpaired_trial_rejected():
    now = datetime.now(timezone.utc)
    with pytest.raises(ValueError):
        pair_trials([{"time": now, "event": "trial_start", "phase": "task"}])


def test_features_are_finite_and_named():
    config = DecoderConfig()
    values = extract_features(signal(4), config)
    assert values.shape == (len(feature_names()),)
    assert np.all(np.isfinite(values))


def test_stream_matches_offline_for_every_chunk_size():
    config = DecoderConfig()
    samples = signal()
    valid = np.ones(len(samples), dtype=bool)
    labels = np.ones(len(samples), dtype=np.int8)
    offline = extract_record_features(samples, valid, labels, config)
    expected = model_scores(SimpleModel(), offline["X"])
    baseline = None
    for chunk_size in [1, 7, 250, 999, len(samples)]:
        decoder = CausalStreamDecoder(SimpleModel(), threshold=0.5, config=config)
        outputs = []
        for start in range(0, len(samples), chunk_size):
            outputs.extend(decoder.push(samples[start : start + chunk_size]))
        scores = np.array([item["score_task"] for item in outputs])
        indices = np.array([item["sample_index"] for item in outputs])
        np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-7)
        np.testing.assert_array_equal(indices, offline["endpoints"])
        if baseline is None:
            baseline = outputs
        else:
            assert outputs == baseline


def test_bad_samples_invalidate_only_overlapping_windows():
    config = DecoderConfig()
    samples = signal(7)
    valid = np.ones(len(samples), dtype=bool)
    valid[1000:1010] = False
    decoder = CausalStreamDecoder(SimpleModel(), threshold=0.5, config=config)
    outputs = decoder.push(samples, valid)
    assert [item["valid"] for item in outputs] == [True, False, False, False]
    assert all(item["score_task"] == 0.5 for item in outputs if not item["valid"])


def test_future_changes_cannot_change_past_outputs():
    config = DecoderConfig()
    original = signal()
    modified = original.copy()
    split = 8 * config.fs
    modified[split:] *= 100
    outputs = []
    for samples in [original, modified]:
        decoder = CausalStreamDecoder(SimpleModel(), threshold=0.5, config=config)
        outputs.append(decoder.push(samples))
    past = split - 1
    past_original = [item for item in outputs[0] if item["sample_index"] <= past]
    past_modified = [item for item in outputs[1] if item["sample_index"] <= past]
    assert past_original == past_modified
    assert outputs[0][-1]["score_task"] != outputs[1][-1]["score_task"]


def test_group_folds_cover_records_without_overlap():
    groups = np.repeat(np.array([f"{i:03d}" for i in range(12)]), 4)
    folds = make_folds(groups, n_splits=5, seed=42)
    assert sorted(np.concatenate(folds).tolist()) == sorted(np.unique(groups).tolist())
    for i, left in enumerate(folds):
        for right in folds[i + 1 :]:
            assert set(left).isdisjoint(set(right))
