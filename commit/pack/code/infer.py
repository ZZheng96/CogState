"""本地文件推理入口：单个 BDF 或目录（递归查找 *.bdf）。

用法：
  python code/infer.py --input <record.bdf | 目录> [--events <events.csv>] [--output <result.csv | 目录>]
                       [--chunk 1.0] [--device auto|cpu|cuda] [--model model/ 或 model_bundle.pt] [--gate deploy_gate.json]

目录模式下每个 <id>_*.bdf 自动配对同目录 <id>_events.csv（存在时）。
推理始终从首个样本起整段流式进行，与是否提供事件文件无关；事件仅在推理后用于附加 label 列
（-1 = 窗口未完整落在单个 trial 内或触及缺失段）并在 label>=0 的帧上计算 summary。
输出列：t_end_s, local_time（北京时间 = 文件名起点 + t_end_s）, score, state[, label]。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import (StreamingInference, find_records, labels_at, local_time, metrics, read_record,  # noqa: E402
                  read_trials, record_start, resolve_model, stream_record)


def load_engine(model=None, gate=None, device=None):
    model, gate = resolve_model(model, gate)
    return StreamingInference(model, device=device, gate=gate), model


def run_file(eng, bdf, events=None, output=None, chunk=1.0, verbose=True):
    """推理单条记录，返回 [(t_end, score, state, label)]，可选写 CSV。"""
    x, fs, bad = read_record(bdf)
    res = stream_record(eng, x, fs, chunk)      # 推理与是否提供 events 无关
    if events:   # 仅事后附加标签：-1 = 窗口不完整落在单个 trial 内或触及缺失段
        y = labels_at([r[0] for r in res], read_trials(events, bdf), bad)
        res = [(t, sc, st, int(lb)) for (t, sc, st), lb in zip(res, y)]
    else:
        res = [(t, sc, st, None) for t, sc, st in res]
    start = record_start(bdf)
    head = "t_end_s,local_time,score,state" + (",label" if events else "")
    rows = [f"{t:.3f},{local_time(start, t)},{sc:.6f},{st}" + (f",{lb}" if events else "") for t, sc, st, lb in res]
    if verbose:
        print(f"[input]  {Path(bdf).name}  fs={fs:g} Hz  duration={len(x) / fs:.1f} s  events={'yes' if events else 'no'}")
        print(f"[output] frames={len(res)}")
        print("  " + head)
        for r in rows[:3]:
            print("  " + r)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text("\n".join([head] + rows) + "\n", encoding="utf-8")
        if verbose:
            print(f"[saved]  {output}")
    return res


def summarize(results, path=None):
    """results：{记录名: [(t, score, state, label)]}；每条记录一行 + pooled + mean。"""
    lines = ["record,frames,bacc,auc,f1"]
    per, ys, ss = [], [], []
    for name, res in results.items():
        y = np.array([r[3] for r in res if r[3] is not None and r[3] >= 0])
        s = np.array([r[1] for r in res if r[3] is not None and r[3] >= 0])
        if len(y) == 0:
            continue
        m = metrics(y, s)
        per.append((len(y), m))
        ys.append(y); ss.append(s)
        lines.append(f"{name},{len(y)},{m['bacc']:.4f},{m['auc']:.4f},{m['f1']:.4f}")
    if not per:
        return None
    m = metrics(np.concatenate(ys), np.concatenate(ss))
    lines.append(f"POOLED,{sum(n for n, _ in per)},{m['bacc']:.4f},{m['auc']:.4f},{m['f1']:.4f}")
    mean = {k: np.nanmean([q[k] for _, q in per]) for k in ("bacc", "auc", "f1")}
    lines.append(f"MEAN,{np.mean([n for n, _ in per]):.1f},{mean['bacc']:.4f},{mean['auc']:.4f},{mean['f1']:.4f}")
    print("[summary]")
    for ln in lines:
        print("  " + ln)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[saved]  {path}")
    return lines


def run(input, events=None, output=None, chunk=1.0, device=None, model=None, gate=None):
    eng, model = load_engine(model, gate, device)
    print(f"[model]  {model}  gate={'on' if eng.gated else 'off'}  device={eng.dev}  chunk={chunk:g} s")
    inp = Path(input)
    if inp.is_dir():
        out_dir = Path(output) if output else Path("results")
        recs = find_records(inp)
        print(f"[records] {len(recs)} BDF under {inp}")
        results = {}
        for bdf, ev in recs:
            res = run_file(eng, bdf, ev, out_dir / f"{bdf.stem}_result.csv", chunk)
            if ev is not None:
                results[bdf.stem] = res
        summarize(results, out_dir / "summary.csv")
        return results
    out = Path(output) if output else None
    if out is not None and out.suffix.lower() != ".csv":   # 视为目录
        out = out / f"{inp.stem}_result.csv"
    res = run_file(eng, inp, events, out, chunk)
    if events:
        summarize({inp.stem: res}, out.with_name(out.stem + "_summary.csv") if out else None)
    return res


def main():
    ap = argparse.ArgumentParser(description="单通道 EEG 认知状态推理（本地文件/目录）")
    ap.add_argument("--input", required=True, help="单个 BDF 或目录（递归查找 *.bdf）")
    ap.add_argument("--events", help="单文件模式可选事件 CSV；目录模式自动配对 <id>_events.csv")
    ap.add_argument("--output", help="结果 CSV 路径或目录；目录模式默认 ./results")
    ap.add_argument("--chunk", type=float, default=1.0, help="每次送入的数据块长度（秒，>0），默认 1.0")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="计算设备，默认 auto")
    ap.add_argument("--model", help="模型目录或 model_bundle.pt，默认 model/model_bundle.pt")
    ap.add_argument("--gate", help="门控 JSON，默认与模型同目录的 deploy_gate.json（不存在则不用门控）")
    a = ap.parse_args()
    if a.chunk <= 0:
        ap.error("--chunk 必须大于 0")
    run(a.input, a.events, a.output, a.chunk, a.device, a.model, a.gate)


if __name__ == "__main__":
    main()
