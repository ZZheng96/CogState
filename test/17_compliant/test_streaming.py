"""流式推理一致性测试（被试 018，全部 trial）。

1. 流式 878 维原始特征 / 5×878 标准化特征 / 分数 vs 离线 sliding1 npz + loso.run_stats_t（帧时间对齐）
2. 前缀不变性：只喂前 N 秒 vs 喂全部，前 N 秒内输出逐位一致
3. 块长 0.1/0.37/1/5 s 输出一致

运行：.venv\\Scripts\\python.exe test/17_compliant/test_streaming.py [--subject 018] [--prefix 600]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

import loso as L
import streaming as S


_orig_wf, _cache = S.window_features, {}


def _cached_wf(x, fs):
    """按窗口样本字节缓存特征：只有样本完全相同才命中，缓冲/步长/统计逻辑仍被完整检验。"""
    k = np.ascontiguousarray(x, dtype=np.float64).tobytes()
    if k not in _cache:
        _cache[k] = _orig_wf(x, fs)
    return _cache[k]


S.window_features = _cached_wf


def run(bundle, x, fs, trials, chunk_s, max_s=None):
    eng = S.StreamingInference(bundle, keep_debug=True)
    n = int(round(chunk_s * fs))
    t0 = time.perf_counter()
    res = S.stream_subject(eng, x, fs, trials, lambda: n, max_s=max_s)
    print(f"  chunk={chunk_s}s max_s={max_s} frames={len(res)} {time.perf_counter() - t0:.0f}s", flush=True)
    return res, eng.debug


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="018")
    ap.add_argument("--prefix", type=float, default=600.0)
    a = ap.parse_args()
    sid = a.subject.zfill(3)
    bundle = torch.load(S.BUNDLE, map_location="cpu", weights_only=False)
    x, fs, trials = S.load_subject(sid)
    out = {}

    print("[run] streaming", flush=True)
    runs = {c: run(bundle, x, fs, trials, c) for c in (1.0, 0.1, 0.37, 5.0)}
    ref, dbg = runs[1.0]

    # 1. 离线对比
    Z = np.load(S.SLID / f"sub-{sid}.npz")
    o = np.argsort(Z["t"], kind="stable")
    F_off, t_off, y_off = np.nan_to_num(Z["F"][o].astype(np.float32)), Z["t"][o].astype(float), Z["y"][o]
    X = torch.tensor(F_off, device=L.dev)
    Zoff = torch.cat([((X - m) / sd).clamp(-10, 10) for m, sd in (L.run_stats_t(X, w) for w in bundle["scales"])], 1)
    sc_off = S._Scorer(bundle).to(L.dev)(Zoff).cpu().numpy()
    Zoff = Zoff.cpu().numpy()
    t_st = np.array([r[0] for r in ref])
    match = len(t_st) == len(t_off) and np.abs(t_st - t_off).max() < 0.01
    F_st = np.stack([d[0] for d in dbg])
    Z_st = np.stack([d[1] for d in dbg])
    s_st = np.array([r[1] for r in ref])
    rel = np.abs(F_st - F_off) / (np.abs(F_off) + 1e-6)
    out["offline"] = dict(
        n_stream=len(t_st), n_offline=len(t_off), time_aligned=bool(match),
        max_abs_dt=float(np.abs(t_st - t_off).max()) if len(t_st) == len(t_off) else None,
        label_equal=bool(np.array_equal([r[3] for r in ref], y_off)),
        raw_feat_max_abs=float(np.abs(F_st - F_off).max()), raw_feat_max_rel=float(rel.max()),
        norm_feat_max_abs=float(np.abs(Z_st - Zoff).max()),
        score_max_abs=float(np.abs(s_st - sc_off).max()),
        state_agree=float(((s_st > 0) == (sc_off > 0)).mean()))
    print("[offline]", out["offline"], flush=True)

    # 2. 前缀不变性
    pre, _ = run(bundle, x, fs, trials, 1.0, max_s=a.prefix)
    k = len(pre)
    out["prefix"] = dict(N_s=a.prefix, n_prefix=k,
                         identical=bool(k > 0 and pre == ref[:k]),
                         score_max_abs=float(np.abs(np.array([r[1] for r in pre]) - s_st[:k]).max()))
    print("[prefix]", out["prefix"], flush=True)

    # 3. 块长一致
    out["chunks"] = {}
    for c, (r, _) in runs.items():
        same_t = len(r) == len(ref) and all(abs(p[0] - q[0]) < 1e-9 for p, q in zip(r, ref))
        d = float(np.abs(np.array([p[1] for p in r]) - s_st).max()) if same_t else None
        out["chunks"][str(c)] = dict(n=len(r), same_times=same_t, score_max_abs=d,
                                     identical=bool(r == ref))
    print("[chunks]", out["chunks"], flush=True)
    out["stream_metrics"] = S.metrics([r[3] for r in ref], s_st)
    print("[metrics]", out["stream_metrics"])
    (S.HERE / "test_streaming.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
