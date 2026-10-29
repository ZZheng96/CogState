import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from preprocessing import (PreprocessingConfig, QualityThresholds, causal_filter,
                           preprocess_window, quality_metrics, resample_offline_polyphase,
                           StreamingPreprocessor)

rng = np.random.default_rng(3)
x = rng.normal(size=1500)
cfg = PreprocessingConfig(mains_hz=50.0, quality_mode="features")

# Prefix and chunking must not use future samples.
y = causal_filter(x, cfg)
assert np.allclose(y[:800], causal_filter(np.concatenate([x[:800], rng.normal(size=1000)]), cfg)[:800])
stream = StreamingPreprocessor(cfg)
parts = [stream.update(x[:300]), stream.update(x[300:901]), stream.update(x[901:])]
assert parts[-1] is not None
assert np.allclose(parts[-1]["signal"], preprocess_window(x, cfg)[0])

# Quality thresholds are unsupervised and fitted on training rows only.
rows = [quality_metrics(rng.normal(size=1500), config=cfg) for _ in range(5)]
th = QualityThresholds.fit(rows)
assert th.flag(rows[0])[0] in (True, False)
assert preprocess_window(np.full(1500, np.nan), cfg)[1]["nan_fraction"] == 1.0

# Fixed amplitude rejection is explicit; outside-window edits cannot affect result.
bad_cfg = PreprocessingConfig(quality_mode="reject", amplitude_limit=2.0)
assert preprocess_window(np.full(1500, 100.0), bad_cfg)[0] is None
assert np.allclose(causal_filter(x, cfg), causal_filter(np.concatenate([x, np.ones(500)]), cfg)[:1500])

# Polyphase output has expected shape and suppresses an out-of-band tone.
t = np.arange(1500) / 250.0
mixed = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*100*t)
r = resample_offline_polyphase(mixed, 4, 5)
assert r.shape == (1200,)
assert np.std(r) < np.std(mixed)
print("preprocessing_causality_quality_resampling: PASS")
