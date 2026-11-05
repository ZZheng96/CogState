"""核验 src/infer.py 的 dt1200_hmm 后处理：复现 e44.json、前缀不变性、分块流式一致性。

运行：.venv\\Scripts\\python.exe test/44_late_drift/verify_infer_dt.py [--subject 018]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
import e44  # noqa: E402
import infer  # noqa: E402
import streaming as S  # noqa: E402

_orig_wf, _cache = S.window_features, {}


def _cached_wf(x, fs):
    k = np.ascontiguousarray(x, dtype=np.float64).tobytes()
    if k not in _cache:
        _cache[k] = _orig_wf(x, fs)
    return _cache[k]


S.window_features = _cached_wf


def reproduce():
    subs, D = e44.L.load()
    data = np.load(e44.TESTS / "42_e17_loso_hmm" / "e42_scores.npz")
    scores = {q: data[q] for q in subs}
    M = e44.post_variants(e44.pad(subs, scores))["dt1200_hmm"].cpu().numpy()
    ref = {q: M[i, :len(scores[q])] for i, q in enumerate(subs)}
    mine = {}
    for q in subs:
        pp = infer.PostProcessor(detrend_halflife=1200)
        mine[q] = np.array([pp.update(float(s)) for s in scores[q]])
    diff = max(float(np.abs(mine[q] - ref[q]).max()) for q in subs)
    # 同一 e44 ema_mean/hmm 函数改用 float64 输入，排除 GPU float32 舍入，检验逐行实现一致
    import math
    import torch
    T = max(len(scores[q]) for q in subs)
    S64 = torch.zeros(len(subs), T, device=e44.DEV, dtype=torch.float64)
    for i, q in enumerate(subs):
        S64[i, :len(scores[q])] = torch.tensor(scores[q], device=e44.DEV, dtype=torch.float64)
    M64 = e44.hmm(S64 - e44.ema_mean(S64, 1200),
                  torch.full_like(S64, -math.log(300))).cpu().numpy()
    diff64 = max(float(np.abs(mine[q] - M64[i, :len(scores[q])]).max()) for i, q in enumerate(subs))
    sign = float(np.mean(np.concatenate([(mine[q] > 0) == (ref[q] > 0) for q in subs])))
    grp = e44.groups()
    full, late = e44.eval_rows(D, subs, mine)
    f, l = e44.decompose(subs, full, grp)["all"], e44.decompose(subs, late, grp)["all"]
    j = json.loads(e44.RESULT.read_text(encoding="utf-8"))["e42|dt1200_hmm"]
    pick = lambda r: {k: round(r[k], 4) for k in ("bacc", "auc", "f1")}  # noqa: E731
    return dict(max_abs_vs_e44=diff, max_abs_vs_e44_float64=diff64, sign_agree=sign, full=pick(f), late45=pick(l),
                e44_full=pick(j["full"]["all"]), e44_late45=pick(j["late45"]["all"]))


def stream(bundle, sig, fs, chunk):
    dec = S.StreamingInference(bundle, fs=fs, step=1.0)
    post = infer.PostProcessor(detrend_halflife=1200)
    return infer.stream_outputs(dec, post, sig, fs, chunk)


def streaming_checks(sid, prefix_s):
    import torch
    bundle = torch.load(S.BUNDLE, map_location="cpu", weights_only=False)
    bdf = next((ROOT / "EEG_dataset" / sid).glob("*.bdf"))
    rec = infer.load_bdf(bdf)
    fs = rec["fs"]
    sig = np.asarray(rec["signal_uv"], dtype=float) * 1e-6
    sig[~rec["valid"]] = np.nan
    ref = stream(bundle, sig, fs, len(sig))
    s_ref = np.array([r["score_task"] for r in ref])
    out = {"n_frames": len(ref)}
    pre = stream(bundle, sig[:int(prefix_s * fs)], fs, 250)
    k = len(pre)
    out["prefix"] = dict(N_s=prefix_s, n=k, identical=pre == ref[:k],
                         max_abs=float(np.abs(np.array([r["score_task"] for r in pre]) - s_ref[:k]).max()))
    print("[prefix]", out["prefix"], flush=True)
    out["chunks"] = {}
    for name, c in (("250pt", 250), ("137pt", 137), ("0.1s", 25), ("0.37s", 92), ("1s", 250), ("5s", 1250)):
        r = stream(bundle, sig, fs, c)
        same = len(r) == len(ref) and all(p["time_s"] == q["time_s"] for p, q in zip(r, ref))
        out["chunks"][name] = dict(n=len(r), same_times=same, identical=r == ref,
                                   max_abs=float(np.abs(np.array([p["score_task"] for p in r]) - s_ref).max())
                                   if same else None)
        print("[chunk]", name, out["chunks"][name], flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="018")
    ap.add_argument("--prefix", type=float, default=1500.0)
    ap.add_argument("--skip-stream", action="store_true")
    a = ap.parse_args()
    out = {"reproduce": reproduce()}
    print("[reproduce]", out["reproduce"], flush=True)
    if a.skip_stream:
        return
    out["streaming"] = streaming_checks(a.subject.zfill(3), a.prefix)
    (HERE / "verify_infer_dt.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
