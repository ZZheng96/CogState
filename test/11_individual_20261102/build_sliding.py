"""E12：把「预先切分的 6 s 窗」换成「连续滑窗」——从 BDF 重新按 2 s 步长提特征。

规则第（1）条要求每次判断的窗口与有效感受野 ≤6 s，但没说决策必须每 6 s 一次。
本脚本用 6 s 窗、步长 2 s 连续滑窗，让「个体记忆」可以按 2 s 粒度更新，
而不是只能每 6 s 更新一次。特征函数与训练集完全一致（858 维高维 + 20 维肌电），
每个窗口的预处理仍在窗内独立完成（与 database 构建时一致）。

产物：src/../test/11_individual_20261102/sliding/sub-XXX.npz
    F (n,878) float32, y (n,) int8, t (n,) float32（窗口末端秒），blk (n,) int16（trial 序号）

运行：py -3.11 test/11_individual_20261102/build_sliding.py [--stride 2] [--subjects 036,054]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import scipy.signal  # noqa: F401  必须早于 torch shim，否则 scipy 的 array-api 探测失败

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "sliding"
WIN = 6.0


def _install_torch_shim():
    """highdim_deep_ablation 只为 CNN 导入 torch；本机无 torch，用最小 shim 只取特征函数。"""
    if "torch" in sys.modules:
        return
    tt, nn = types.ModuleType("torch"), types.ModuleType("torch.nn")

    class _M:  # 仅作为 nn.Module 的占位基类
        pass

    for a in ("Module", "Sequential", "Conv1d", "BatchNorm1d", "ReLU",
              "AdaptiveAvgPool1d", "Linear", "Dropout", "BCEWithLogitsLoss"):
        setattr(nn, a, _M)
    tt.nn, tt.Tensor, tt.float32 = nn, _M, "float32"
    tt.no_grad = lambda: None
    sys.modules["torch"], sys.modules["torch.nn"] = tt, nn


def _paths():
    for d in ("src", "test/03_deep", "test/08_diagnostics", "test/01_features"):
        p = str(ROOT / d)
        if p not in sys.path:
            sys.path.append(p)


def build_one(args):
    sid, stride, outdir = args
    _install_torch_shim()
    _paths()
    import mne
    from cogstate import FS, bad_intervals, load_trials, valid_window
    from highdim_deep_ablation import highdim_features
    from emg_v2 import features_v2

    csv = next((ROOT / "EEG_dataset").glob(f"{sid}/*_events.csv"))
    bdf = next(csv.parent.glob("*.bdf"))
    raw = mne.io.read_raw_bdf(str(bdf), preload=True, verbose="ERROR")
    fs, rec_end = float(raw.info["sfreq"]), raw.n_times / float(raw.info["sfreq"])
    assert abs(fs - FS) < 1e-6, (sid, fs)
    bad = bad_intervals(raw)
    trials = load_trials(str(csv), str(bdf))
    F, Y, T, B = [], [], [], []
    t0 = time.perf_counter()
    for bi, tr in enumerate(trials):
        end = tr.start + WIN
        while end <= tr.end + 1e-6:
            if valid_window(end - WIN, end, bad, rec_end):
                s, e = int(round((end - WIN) * fs)), int(round(end * fs))
                x = raw.get_data(start=s, stop=e)[0]
                if len(x) == e - s and np.isfinite(x).all():
                    F.append(np.concatenate([highdim_features(x.astype(float), fs),
                                             features_v2(x.astype(float))]))
                    Y.append(tr.label)
                    T.append(end)
                    B.append(bi)
            end += stride
    raw.close()
    F = np.asarray(F, dtype=np.float32)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"sub-{sid}.npz", F=F, y=np.asarray(Y, dtype=np.int8),
                        t=np.asarray(T, dtype=np.float32), blk=np.asarray(B, dtype=np.int16))
    return sid, F.shape[0], int(sum(1 for _ in trials)), round(time.perf_counter() - t0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=float, default=2.0)
    ap.add_argument("--subjects", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    _install_torch_shim()
    _paths()
    if a.subjects:
        subs = [s.strip().zfill(3) for s in a.subjects.split(",")]
    else:
        subs = sorted(p.name for p in (ROOT / "EEG_dataset").iterdir() if p.is_dir())
    print(f"[build] {len(subs)} subjects stride={a.stride}s workers={a.workers} out={a.out}", flush=True)
    t0 = time.perf_counter()
    with Pool(a.workers) as pool:
        for i, (sid, n, ntr, sec) in enumerate(pool.imap_unordered(
                build_one, [(s, a.stride, a.out) for s in subs]), 1):
            print(f"[{i}/{len(subs)}] {sid} rows={n} trials={ntr} {sec}s "
                  f"elapsed={time.perf_counter() - t0:.0f}s", flush=True)
    meta = {"stride_s": a.stride, "window_s": WIN, "subjects": subs,
            "notes": "6 s 窗、指定步长滑窗；特征函数与 database/realorder 完全一致"}
    (Path(a.out) / "manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                               encoding="utf-8")
    print(f"[done] {time.perf_counter() - t0:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
