"""训练入口：从原始带标签记录（BDF + <id>_events.csv）训练模型与门控。

用法：
  python code/train.py --data <目录> --out <模型目录> [--subjects 001 002 ...] [--device auto|cpu|cuda]
                       [--no-gate | --fit-gate]

流程：
  1. 特征：6 s 窗、1 s 步长，窗口须完整落在单个 trial 内且不含缺失段 -> <out>/features/sub-<id>.npz（已存在则复用）
  2. train_bundle -> <out>/model_bundle.pt
  3. 门控（默认）：复制随包 OOF 门控 model/deploy_gate.json（作用于 std 归一化的 lr/mlp 分量，可迁移）；
     --no-gate：不用门控；
     --fit-gate：每条训练记录整段以 1 s 块流式通过新模型（不用门控），记录 lr/mlp 分量，
     计算 9 维因果门控特征（窗口 120 帧），在 label>=0 的帧上拟合
     LogisticRegression(C=0.1, balanced, liblinear) -> <out>/deploy_gate.json
之后：python code/infer.py --model <out> ...
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import (GATE, StreamingInference, build_features, find_records, fit_gate, gate_features,  # noqa: E402
                  labels_at, read_record, read_trials, stream_record, train_bundle)


def subject_of(bdf):
    return Path(bdf).name.split("_")[0]


def gate_components(bundle_path, bdf, events, device=None):
    """整段流式（chunk=1 s）通过 bundle（无门控），返回 (X 门控特征, y 帧标签)。"""
    eng = StreamingInference(bundle_path, device=device, gate=False, keep_components=True)
    x, fs, bad = read_record(bdf)
    res = stream_record(eng, x, fs, 1.0)
    comp = np.asarray(eng.components, dtype=float)
    y = labels_at([r[0] for r in res], read_trials(events, bdf), bad)
    return gate_features(comp[:, 0], comp[:, 1]), y


def main():
    ap = argparse.ArgumentParser(description="从原始带标签记录训练认知状态模型")
    ap.add_argument("--data", required=True, help="记录目录（递归查找 *.bdf，需同目录 <id>_events.csv）")
    ap.add_argument("--out", required=True, help="输出模型目录（model_bundle.pt + deploy_gate.json）")
    ap.add_argument("--subjects", nargs="*", help="可选，仅用这些被试编号（文件名首段）")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--no-gate", action="store_true", help="不用门控（推理时 score = lr+mlp -> EMA + HMM）")
    g.add_argument("--fit-gate", action="store_true", help="在训练记录上样本内重新拟合门控（默认复制随包门控参数）")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = [(b, e) for b, e in find_records(a.data) if e is not None]
    if a.subjects:
        keep = set(a.subjects)
        recs = [(b, e) for b, e in recs if subject_of(b) in keep]
    if not recs:
        sys.exit("未找到带事件文件的记录")
    print(f"[data] {len(recs)} records: {' '.join(subject_of(b) for b, _ in recs)}", flush=True)

    t0 = time.time()
    files = {}
    for b, e in recs:
        sid = subject_of(b)
        files[sid] = build_features(b, e, out / "features" / f"sub-{sid}.npz")
        print(f"[feat] {sid} frames={len(np.load(files[sid])['y'])}  {time.time() - t0:.0f}s", flush=True)

    bundle_path = out / "model_bundle.pt"
    train_bundle(bundle_path, files=files, device=a.device)
    gate_path = out / "deploy_gate.json"
    if a.no_gate:
        if gate_path.exists():
            gate_path.unlink()
        print("[gate] skipped (--no-gate)")
        return
    if not a.fit_gate:
        shutil.copyfile(GATE, gate_path)
        print(f"[gate] 复制随包门控参数 {GATE} -> {gate_path}")
        print(f"[done] {time.time() - t0:.0f}s  infer: python code/infer.py --model {out} --input <bdf|dir>")
        return

    Xs, ys = [], []
    for b, e in recs:
        X, y = gate_components(bundle_path, b, e, a.device)
        Xs.append(X[y >= 0]); ys.append(y[y >= 0])
        print(f"[gate] {subject_of(b)} frames={int((y >= 0).sum())}  {time.time() - t0:.0f}s", flush=True)
    g = fit_gate(np.concatenate(Xs), np.concatenate(ys), f"in-sample, {len(recs)} subjects")
    gate_path.write_text(json.dumps(g, indent=2), encoding="utf-8")
    print(f"[gate] coef={np.round(g['coef'], 4).tolist()} intercept={g['intercept']:.4f} -> {gate_path}")
    print(f"[done] {time.time() - t0:.0f}s  infer: python code/infer.py --model {out} --input <bdf|dir>")


if __name__ == "__main__":
    main()
