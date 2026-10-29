"""Focused regression tests for the Kalman deep-learning streaming contract."""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kalman_deep import (  # noqa: E402
    KalmanDeepModel,
    kalman_filter_window,
    load_kalman_deep_checkpoint,
)


def main():
    rng = np.random.default_rng(0)
    x = rng.normal(size=1500).astype(np.float32)
    checkpoint_names = ["no_kalman", "kalman", "raw_cnn", "raw_kalman_cnn"]
    for name in checkpoint_names:
        path = ROOT / "test" / f"kalman_deep_final_{name}.pt"
        ck, net, mean, scale = load_kalman_deep_checkpoint(path, device="cpu")
        model = KalmanDeepModel.from_checkpoint(path, device="cpu")
        assert model.use_kalman is bool(ck["use_kalman"])
        one = model.update(x)
        model.reset()
        start = 0
        many = None
        for n in [1, 17, 101, 333, 1048]:
            many = model.update(x[start:start + n])
            start += n
        assert one is not None and many is not None and np.isclose(one, many, atol=1e-6)
        signal = kalman_filter_window(x) if ck["use_kalman"] else x
        if ck["model"].startswith("cnn"):
            inp = signal.reshape(250, 6).mean(1)[None]
        else:
            from cogstate import causal_features
            inp = causal_features(signal)[None]
        offline = float(model.predict_proba(inp)[0, 0])
        assert np.isclose(one, offline, atol=1e-6)
        model.reset()
        assert model.update(np.zeros(1499, dtype=np.float32)) is None
        assert model.update(np.zeros(1, dtype=np.float32)) is not None
        model.reset()
        assert model.update(np.zeros(10, dtype=np.float32)) is None
        assert model.update(np.array([np.nan], dtype=np.float32)) is None
        assert model.update(x) is not None
        try:
            KalmanDeepModel(net, mean, scale, sfreq=249, device="cpu")
        except ValueError:
            pass
        else:
            raise AssertionError("invalid sampling rate accepted")

    future = np.full(200, 99.0, dtype=np.float32)
    assert np.allclose(kalman_filter_window(x), kalman_filter_window(np.concatenate([x, future]))[:1500])
    for bad in [(-1, 1e-8, 1e-8), (1e-10, 0, 1e-8), (1e-10, 1e-8, -1)]:
        try:
            kalman_filter_window(x, *bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid parameters accepted: {bad}")
    print("kalman_deep_four_checkpoints_offline_streaming_chunk_reset_bounds: PASS")


if __name__ == "__main__":
    main()
