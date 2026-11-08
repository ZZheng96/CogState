"""验证 system/algorithms 复制版与原 src/test 代码数值一致（特征 atol=0、流式分数逐帧相等）。"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for p in ("src", "test/03_deep", "test/08_diagnostics", "test/01_features", "test/17_compliant"):
    sys.path.insert(0, str(ROOT / p))

import streaming as OLD  # noqa: E402
import infer as OLD_INFER  # noqa: E402
import eeg_decoder as OLD_DEC  # noqa: E402
from system.algorithms import features as NF, e17 as NE, postprocess as NP, io as NIO  # noqa: E402
from system import dataset  # noqa: E402

sid = sys.argv[1] if len(sys.argv) > 1 else "018"
bdf, csv = dataset.paths("builtin", sid)
a, b = OLD_DEC.load_bdf(bdf), NIO.load_bdf(bdf)
assert np.array_equal(a["signal_uv"], b["signal_uv"]) and np.array_equal(a["valid"], b["valid"])
x = a["signal_uv"] * 1e-6
x[~a["valid"]] = np.nan
print("load_bdf identical", flush=True)

# 特征：前 200 个 6 s 有效窗
ok = 0
for k in range(0, len(x) - 1500, 250):
    w = x[k:k + 1500]
    if not np.isfinite(w).all():
        continue
    f0, f1 = OLD.window_features(w, 250.), NF.window_features(w, 250.)
    assert np.allclose(f0, f1, atol=0, rtol=0) and np.array_equal(f0, f1), k
    ok += 1
    if ok >= 200:
        break
print(f"window_features identical on {ok} windows", flush=True)

# 流式推理：前 ~1300 s
n = 1300 * 250
d0 = OLD.StreamingInference(ROOT / "test/17_compliant/model_bundle.pt")
d1 = NE.StreamingInference(NE.BUNDLE)
p0, p1 = OLD_INFER.PostProcessor(1200, 300), NP.PostProcessor(1200, 300)
r0, r1 = [], []
for i in range(0, n, 250):
    for t, s, _ in d0.update(x[i:i + 250]):
        r0.append((t, s, p0.update(s)))
    for t, s, _ in d1.update(x[i:i + 250]):
        r1.append((t, s, p1.update(s)))
r0, r1 = np.array(r0), np.array(r1)
print(f"stream frames old={len(r0)} new={len(r1)} max|diff|={np.abs(r0 - r1).max() if len(r0) else 0}")
assert r0.shape == r1.shape and np.array_equal(r0, r1)
print("StreamingInference + PostProcessor identical")
