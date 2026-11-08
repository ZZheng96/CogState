"""E95 推理口径对比：A 整段流式（新版 infer.run_file） vs B 按 trial 分段 + mark_invalid（旧版 01559f6 streaming.run）。

A：stream_record 从首样本整段推理，label=labels_at（-1 剔除）。
B：每被试 reset 一次，仅喂 trial 区间 [start,end]，每段开头 mark_invalid(t_next=s/fs)，label=trial 标签。
C：B 的帧中剔除 labels_at=-1 的帧。
逐被试帧结果缓存到 frames/sub_<id>.npz；汇总写 results.json / per_subject.csv。

用法：python run_compare.py [--workers 6] [--subs 001,002]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CODE = ROOT / "commit" / "pack" / "code"
MODEL = ROOT / "commit" / "pack" / "model" / "model_bundle.pt"
GATE = MODEL.with_name("deploy_gate.json")
DATA = ROOT / "EEG_dataset"
OUT = HERE / "frames"
CHUNK = 1.0
sys.path.insert(0, str(CODE))

_ENG = None


def _engine():
    global _ENG
    if _ENG is None:
        from core import StreamingInference
        _ENG = StreamingInference(MODEL, device="cuda", gate=GATE)
    return _ENG


def run_subject(bdf, ev):
    from core import labels_at, read_record, read_trials, stream_record
    sid = Path(bdf).name.split("_")[0]
    out = OUT / f"sub_{sid}.npz"
    if out.exists():
        return sid, str(out), 0.0
    t0 = time.time()
    eng = _engine()
    x, fs, bad = read_record(bdf)
    trials = read_trials(ev, bdf)
    # A：整段流式
    ra = stream_record(eng, x, fs, CHUNK)
    ta = np.array([r[0] for r in ra]); sa = np.array([r[1] for r in ra])
    ya = labels_at(ta, trials, bad)
    # B：旧版 run() 的 trial 分段逻辑（旧版每次 run 新建引擎 = reset）
    eng.reset()
    n = max(1, int(round(CHUNK * fs)))
    rb = []
    for a, b, lab in trials:
        s = int(round(a * fs)); e = min(int(np.floor((b + 1e-6) * fs)), len(x))
        eng.mark_invalid(t_next=s / fs)
        for i in range(s, e, n):
            rb.extend((t, sc, lab) for t, sc, _ in eng.update(x[i:min(i + n, e)]))
    tb = np.array([r[0] for r in rb]); sb = np.array([r[1] for r in rb]); yb = np.array([r[2] for r in rb])
    yb_new = labels_at(tb, trials, bad)   # C 用：新版口径下该帧的标签
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, ta=ta, sa=sa, ya=ya, tb=tb, sb=sb, yb=yb, yb_new=yb_new,
                        dur=len(x) / fs, n_trials=len(trials), n_bad=len(bad))
    return sid, str(out), time.time() - t0


def summarize():
    from core import metrics
    rows, pool = [], {k: ([], []) for k in "ABC"}
    for f in sorted(OUT.glob("sub_*.npz")):
        d = np.load(f)
        sid = f.stem[4:]
        sets = {"A": (d["ya"][d["ya"] >= 0], d["sa"][d["ya"] >= 0]),
                "B": (d["yb"], d["sb"]),
                "C": (d["yb"][d["yb_new"] >= 0], d["sb"][d["yb_new"] >= 0])}
        row = dict(sub=sid, frames_total_A=int(len(d["ya"])))
        for k, (y, s) in sets.items():
            m = metrics(y.astype(int), s)
            row.update({f"{k}_n": int(len(y)), f"{k}_bacc": m["bacc"], f"{k}_auc": m["auc"], f"{k}_f1": m["f1"]})
            pool[k][0].append(y); pool[k][1].append(s)
        rows.append(row)
    agg = {}
    for k in "ABC":
        pm = metrics(np.concatenate(pool[k][0]).astype(int), np.concatenate(pool[k][1]))
        agg[k] = dict(mean={m: float(np.nanmean([r[f"{k}_{m}"] for r in rows])) for m in ("bacc", "auc", "f1")},
                      pooled=pm, frames=int(sum(len(y) for y in pool[k][0])))
    keys = list(rows[0])
    (HERE / "per_subject.csv").write_text(
        "\n".join([",".join(keys)] + [",".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in keys)
                                      for r in rows]) + "\n", encoding="utf-8")
    (HERE / "results.json").write_text(json.dumps(dict(aggregate=agg, per_subject=rows), indent=1, ensure_ascii=False),
                                       encoding="utf-8")
    for k in "ABC":
        print(k, agg[k])
    return agg, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--subs", default="")
    a = ap.parse_args()
    from core import find_records
    recs = [(str(b), str(e)) for b, e in find_records(DATA) if e is not None]
    if a.subs:
        keep = set(a.subs.split(","))
        recs = [r for r in recs if Path(r[0]).name.split("_")[0] in keep]
    print(f"[records] {len(recs)} with events", flush=True)
    if a.workers <= 1:  # 单进程复用同一引擎：多进程各自加载模型会耗尽显存
        for b, e in recs:
            sid, p, dt = run_subject(b, e)
            print(f"[done] {sid} {dt:.0f}s", flush=True)
    else:
        with ProcessPoolExecutor(a.workers) as ex:
            futs = [ex.submit(run_subject, b, e) for b, e in recs]
            for f in as_completed(futs):
                sid, p, dt = f.result()
                print(f"[done] {sid} {dt:.0f}s", flush=True)
    summarize()


if __name__ == "__main__":
    main()
