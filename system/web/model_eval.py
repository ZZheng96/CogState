"""把 src/infer.py 的当前方案（E17 多尺度 LR+MLP + 因果 EMA 去趋势 + 前向 HMM）接入查看器。

对每个被试按 infer.py 的口径从记录首样本起流式推理（每 1 s 一个决策，窗口 = 过去 6 s），
缓存逐帧结果到 system/web/cache/pred_<sid>.npz，并计算 bACC / AUC / F1。

评估帧：6 s 窗口 [t-6, t] 完全落在单个 trial 内且不跨缺失段，标签 = 该 trial 的 phase。
注意：model_bundle.pt 用全部 57 人训练，这里的指标是训练集内（in-sample）指标，
跨被试泛化成绩见 test/17_compliant/报告_E17_合规方案.md。

用法：
  python system/web/model_eval.py --all          # 预计算全部被试
  python system/web/model_eval.py --subject 018
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
CACHE = Path(__file__).resolve().parent / "cache"
BUNDLE_PATH = ROOT / "test" / "17_compliant" / "model_bundle.pt"
WIN_S = 6.0
for p in (str(SRC), str(ROOT / "test" / "17_compliant")):
    if p not in sys.path:
        sys.path.insert(0, p)

if not hasattr(np, "in1d"):
    np.in1d = np.isin  # type: ignore[attr-defined]

SCHEME = {
    "name": "E17 多尺度 LR+MLP + 因果 EMA 去趋势(hl=1200) + 前向 HMM(d=300)",
    "entry": "src/infer.py",
    "model": "test/17_compliant/model_bundle.pt",
    "window_s": WIN_S,
    "step_s": 1.0,
    "threshold": 0.0,
    "note": "模型用全部 57 人训练，此处为训练集内指标；跨被试 5 折成绩 bACC .762 / AUC .823（见 E17 报告）",
}

_lock = threading.Lock()
_decoders: dict[str, object] = {}


def _engine(bundle_path: Path = BUNDLE_PATH):
    key = str(bundle_path)
    if key not in _decoders:
        import torch
        from streaming import StreamingInference
        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        _decoders.clear()  # 同时只保留一个模型在显存中
        _decoders[key] = StreamingInference(bundle, fs=250, step=1.0)
    return _decoders[key]


def _trials(sid: str):
    from cogstate import load_trials
    d = ROOT / "EEG_dataset" / sid
    csv = next(d.glob("*_events.csv"))
    bdf = next(d.glob("*.bdf"))
    return bdf, load_trials(str(csv), str(bdf))


def _labels(sid: str, t: np.ndarray) -> np.ndarray:
    """-1 = 不参与评估。唯一实现在 system.truth。"""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from system import truth
    return truth.labels("builtin", sid, t)


def compute(sid: str, bundle_path: Path = BUNDLE_PATH, out: Path | None = None) -> dict:
    from eeg_decoder import load_bdf
    from infer import PostProcessor, stream_outputs
    bdf, trials = _trials(sid)
    rec = load_bdf(bdf)
    fs = rec["fs"]
    x = np.asarray(rec["signal_uv"], dtype=float) * 1e-6
    x[~rec["valid"]] = np.nan
    with _lock:
        dec = _engine(bundle_path)
        dec.reset()
        raw_scores = []
        orig = dec.update

        def tap(chunk):
            out = orig(chunk)
            raw_scores.extend(s for _, s, _ in out)
            return out

        dec.update = tap
        try:
            outs = stream_outputs(dec, PostProcessor(detrend_halflife=1200, dwell=300, temperature=1.0), x, fs, 250)
        finally:
            dec.update = orig
    t = np.array([o["time_s"] for o in outs], dtype=np.float64)
    post = np.array([o["score_task"] for o in outs], dtype=np.float32)
    raw = np.array(raw_scores, dtype=np.float32)
    y = _labels(sid, t)
    out = out or CACHE / f"pred_{sid}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, t=t, raw=raw, post=post, y=y)
    tmp.replace(out)
    return {"t": t, "raw": raw, "post": post, "y": y}


def load(sid: str, compute_missing: bool = True) -> dict | None:
    f = CACHE / f"pred_{sid}.npz"
    if f.exists():
        z = np.load(f)
        return {k: z[k] for k in ("t", "raw", "post", "y")}
    return compute(sid) if compute_missing else None


def metrics(y: np.ndarray, s: np.ndarray) -> dict | None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from system.evaluate import metrics as _m
    return _m(y, s)


def subject_metrics(sid: str, compute_missing: bool = True) -> dict | None:
    d = load(sid, compute_missing)
    if d is None:
        return None
    return {"subject": sid, "raw": metrics(d["y"], d["raw"]), "post": metrics(d["y"], d["post"])}


def summary(sids: list[str]) -> dict:
    rows = [r for r in (subject_metrics(s, compute_missing=False) for s in sids) if r]
    out = {"scheme": SCHEME, "n_done": len(rows), "n_total": len(sids), "subjects": rows}
    for key in ("raw", "post"):
        ms = [r[key] for r in rows if r[key]]
        if ms:
            out[key] = {k: float(np.mean([m[k] for m in ms])) for k in ("bacc", "auc", "f1")}
            out[key]["n_ge_09"] = int(sum(m["bacc"] >= 0.9 for m in ms))
    return out


def at(sid: str, end: float) -> dict | None:
    """判断时刻 end 处（取 <= end 的最近决策帧）的方案输出。"""
    d = load(sid, compute_missing=False)
    if d is None:
        return None
    i = int(np.searchsorted(d["t"], end + 1e-6)) - 1
    if i < 0:
        return {"ready": True, "frame": None}
    return {
        "ready": True,
        "frame": {
            "t": float(d["t"][i]),
            "raw": float(d["raw"][i]),
            "post": float(d["post"][i]),
            "pred": "task" if d["post"][i] > 0 else "relax",
            "label": int(d["y"][i]),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    sids = sorted(p.name for p in (ROOT / "EEG_dataset").iterdir() if p.is_dir() and list(p.glob("*.bdf")))
    if a.subject:
        sids = [a.subject]
    for sid in sids:
        if a.all and (CACHE / f"pred_{sid}.npz").exists():
            continue
        r = subject_metrics(sid) if not a.all else (compute(sid) and subject_metrics(sid, False))
        print(sid, json.dumps({k: r[k] for k in ("raw", "post")}), flush=True)
    s = summary(sids)
    print(json.dumps({k: s.get(k) for k in ("n_done", "raw", "post")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
