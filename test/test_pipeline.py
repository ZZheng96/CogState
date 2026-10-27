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
    causal_smooth_scores,
    extract_features,
    extract_record_features,
    feature_names,
    fuse_protocol_scores,
    labels_from_trials,
    learn_protocol_template,
    model_scores,
    pair_trials,
    parse_recording_start,
)
from train import cache_fingerprint, make_folds  # noqa: E402


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


def test_mean3_matches_offline_and_respects_receptive_field():
    config = DecoderConfig()
    samples = signal()
    labels = np.ones(len(samples), dtype=np.int8)
    valid = np.ones(len(samples), dtype=bool)
    offline = extract_record_features(samples, valid, labels, config)
    raw_scores = model_scores(SimpleModel(), offline["X"])
    expected = causal_smooth_scores(
        raw_scores,
        np.zeros(len(raw_scores), dtype=int),
        offline["endpoints"],
        config.stride_samples,
        "mean3",
    )
    decoder = CausalStreamDecoder(SimpleModel(), 0.5, config, postprocess="mean3")
    outputs = decoder.push(samples)
    np.testing.assert_allclose(
        [item["score_task"] for item in outputs], expected, rtol=0, atol=1e-12
    )
    with pytest.raises(ValueError):
        CausalStreamDecoder(SimpleModel(), 0.5, DecoderConfig(window_s=6), postprocess="mean3")


def test_mean3_uses_unlabeled_context_like_streaming():
    config = DecoderConfig()
    samples = signal()
    labels = np.full(len(samples), -1, dtype=np.int8)
    labels[5 * config.fs :] = 1
    valid = np.ones(len(samples), dtype=bool)
    offline = extract_record_features(samples, valid, labels, config)
    raw_scores = model_scores(SimpleModel(), offline["X"])
    smoothed = causal_smooth_scores(
        raw_scores,
        np.zeros(len(raw_scores), dtype=int),
        offline["endpoints"],
        config.stride_samples,
        "mean3",
    )
    decoder = CausalStreamDecoder(SimpleModel(), 0.5, config, postprocess="mean3")
    outputs = decoder.push(samples)
    labeled = offline["y"] >= 0
    actual = [item for item, keep in zip(outputs, labeled) if keep]
    np.testing.assert_allclose(
        [item["score_task"] for item in actual], smoothed[labeled], rtol=0, atol=1e-12
    )


def test_protocol_fusion_matches_streaming_and_preserves_uncovered_scores():
    config = DecoderConfig()
    samples = signal()
    valid = np.ones(len(samples), dtype=bool)
    labels = np.ones(len(samples), dtype=np.int8)
    offline = extract_record_features(samples, valid, labels, config)
    raw_scores = model_scores(SimpleModel(), offline["X"])
    smoothed = causal_smooth_scores(
        raw_scores,
        np.zeros(len(raw_scores), dtype=int),
        offline["endpoints"],
        config.stride_samples,
        "mean3",
    )
    probability = np.full(16, np.nan, dtype=np.float32)
    probability[4:12] = np.array([0, 0.5, 1, 0.25, 0, 1, 0.5, 1])
    protocol = {
        "probability": probability,
        "count": np.repeat([0, 1], [4, 12]).astype(np.int32),
    }
    expected = fuse_protocol_scores(
        smoothed, offline["endpoints"], config.fs, protocol, weight=0.1
    )
    outputs = []
    decoder = CausalStreamDecoder(
        SimpleModel(), 0.5, config, postprocess="mean3", protocol=protocol
    )
    for start in range(0, len(samples), 137):
        outputs.extend(decoder.push(samples[start : start + 137]))
    np.testing.assert_allclose(
        [item["score_task"] for item in outputs], expected, rtol=0, atol=1e-12
    )
    uncovered = {"probability": np.full(16, np.nan), "count": np.zeros(16, dtype=int)}
    np.testing.assert_array_equal(
        fuse_protocol_scores(smoothed, offline["endpoints"], config.fs, uncovered, 0.1),
        smoothed,
    )


@pytest.mark.parametrize("mode", ["delta", "raw_delta"])
def test_baseline_adaptation_matches_offline(mode):
    config = DecoderConfig()
    samples = signal(65)
    valid = np.ones(len(samples), dtype=bool)
    labels = np.ones(len(samples), dtype=np.int8)
    baseline_vectors = []
    for endpoint in range(config.window_samples - 1, 60 * config.fs, config.stride_samples):
        start = endpoint - config.window_samples + 1
        baseline_vectors.append(extract_features(samples[start : endpoint + 1], config))
    baseline = np.median(np.asarray(baseline_vectors), axis=0).astype(np.float32)
    offline = extract_record_features(samples, valid, labels, config)
    selected = offline["endpoints"] >= 60 * config.fs
    raw = offline["X"][selected]
    if mode == "delta":
        model_input = raw - baseline
    else:
        model_input = np.concatenate([raw, raw - baseline], axis=1)
    expected = model_scores(SimpleModel(), model_input)
    decoder = CausalStreamDecoder(SimpleModel(), 0.5, config, feature_mode=mode)
    outputs = []
    for start in range(0, len(samples), 137):
        outputs.extend(decoder.push(samples[start : start + 137]))
    actual = [item for item in outputs if item["sample_index"] >= 60 * config.fs]
    np.testing.assert_allclose(
        [item["score_task"] for item in actual], expected, rtol=0, atol=1e-7
    )


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


def test_protocol_template_excludes_held_out_events(tmp_path):
    start = datetime(2026, 10, 17, 17, 1, 37, 984000, tzinfo=timezone(timedelta(hours=8)))
    for record_id, phases in {"001": ("relax", "task"), "002": ("task", "relax")}.items():
        record = tmp_path / record_id
        record.mkdir()
        (record / f"{record_id}_20261017_170137.984_UTC+0800.bdf").write_bytes(b"")
        lines = ["local_time,event,phase,key"]
        for trial_index, phase in enumerate(phases):
            trial_start = start + timedelta(seconds=60 + 60 * trial_index)
            trial_end = trial_start + timedelta(seconds=60)
            lines.append(f"{trial_start.isoformat()},trial_start,{phase},")
            lines.append(f"{trial_end.isoformat()},trial_end,{phase},")
        (record / f"{record_id}_events.csv").write_text("\n".join(lines), encoding="utf-8")
    train_only = learn_protocol_template(tmp_path, {"001"}, max_seconds=200)
    assert train_only["probability"][60] == 0
    assert train_only["probability"][120] == 1
    both = learn_protocol_template(tmp_path, max_seconds=200)
    assert both["probability"][60] == 0.5
    assert both["probability"][120] == 0.5


def test_cache_fingerprint_changes_with_input_data(tmp_path):
    record = tmp_path / "001"
    record.mkdir()
    bdf = record / "001_test.bdf"
    events = record / "001_events.csv"
    bdf.write_bytes(b"abc")
    events.write_text("local_time,event,phase,key\n", encoding="utf-8")
    first = cache_fingerprint(tmp_path)
    assert first == cache_fingerprint(tmp_path)
    bdf.write_bytes(b"abcd")
    assert first != cache_fingerprint(tmp_path)


def test_group_folds_cover_records_without_overlap():
    groups = np.repeat(np.array([f"{i:03d}" for i in range(12)]), 4)
    folds = make_folds(groups, n_splits=5, seed=42)
    assert sorted(np.concatenate(folds).tolist()) == sorted(np.unique(groups).tolist())
    for i, left in enumerate(folds):
        for right in folds[i + 1 :]:
            assert set(left).isdisjoint(set(right))
