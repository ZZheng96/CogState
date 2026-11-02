"""重建"真实采集顺序"数据集：从 BDF 重新切 6 秒窗，并把 database/windows 里已算好的
858 维 EEG 特征 + 20 维肌电特征按原始波形哈希精确对齐回来。

产物（同目录）：
    realorder.npz
        F     (n, 878) float32   database 的 f||e，逐窗精确对齐
        y     (n,)     int8      0=relax 1=task
        g     (n,)     <U3       被试编号
        t     (n,)     float32   窗口末端相对该 BDF 首样本的秒数（真实顺序）
        blk   (n,)     int8      块序号 0..5（同被试内 trial 顺序）
        pos   (n,)     int32     被试内窗序号（按真实时间排序后的位置）
        subject_counts / block_lengths 记录在 manifest 里

为什么必须重建：database/windows 里的窗顺序被固定种子打乱，且不保存时间戳/块号；
只靠它无法做任何"按真实顺序的因果"实验。波形哈希在 x 完全一致时命中率应为 100%。

运行：py -3.11 test/10_projects/protocol_20261102/build_realorder.py
"""
from __future__ import annotations

import glob
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

import mne  # noqa: E402

from cogstate import FS, bad_intervals, load_trials, valid_window  # noqa: E402

WIN = 6.0


def h(x: np.ndarray) -> bytes:
    return hashlib.md5(np.ascontiguousarray(x, dtype=np.float32).tobytes()).digest()


def build_index():
    """hash -> (f, e)。同一波形（如 008 的饱和段）可能出现多次甚至跨标签，但特征只由波形决定，
    因此按波形去重是安全的；跨标签重复单独计数以便在报告里披露。"""
    idx, n_dup, n_cross = {}, 0, 0
    n = 0
    for name in ("relax", "task"):
        for p in sorted((ROOT / "database" / "windows" / name).glob("sub-*.npz")):
            d = np.load(p)
            for i in range(len(d["x"])):
                k = h(d["x"][i])
                if k in idx:
                    n_dup += 1
                    if idx[k][2] != int(d["label"][i]):
                        n_cross += 1
                else:
                    idx[k] = (d["f"][i], d["e"][i], int(d["label"][i]))
                n += 1
    return idx, n, n_dup, n_cross


def main():
    t0 = time.perf_counter()
    idx, n_db, n_dup, n_cross = build_index()
    print(f"[index] database windows={n_db} unique_hashes={len(idx)} duplicates={n_dup} "
          f"cross_label_duplicates={n_cross} elapsed={time.perf_counter() - t0:.1f}s", flush=True)

    F, Y, G, T, B, P = [], [], [], [], [], []
    missing, subject_meta = 0, {}
    paths = sorted((ROOT / "EEG_dataset").glob("*/*_events.csv"))
    for ci, csv in enumerate(paths, 1):
        sid = csv.parent.name
        bdf = next(csv.parent.glob("*.bdf"))
        raw = mne.io.read_raw_bdf(str(bdf), preload=True, verbose="ERROR")
        fs, rec_end = float(raw.info["sfreq"]), raw.n_times / float(raw.info["sfreq"])
        assert abs(fs - FS) < 1e-6, (sid, fs)
        bad = bad_intervals(raw)
        trials = load_trials(str(csv), str(bdf))
        blk_len = []
        rows = 0
        for tr in trials:
            blk_len.append(round(tr.end - tr.start, 3))
            end = tr.start + WIN
            while end <= tr.end + 1e-6:
                if valid_window(end - WIN, end, bad, rec_end):
                    s, e = int(round((end - WIN) * fs)), int(round(end * fs))
                    x = raw.get_data(start=s, stop=e)[0]
                    if len(x) == e - s and np.isfinite(x).all():
                        hit = idx.get(h(x))
                        if hit is None:
                            missing += 1
                        else:
                            f, em, _lab = hit
                            F.append(np.concatenate([f, em]))
                            Y.append(tr.label)
                            G.append(sid)
                            T.append(end)
                            B.append(tr.trial_id)
                            P.append(rows)
                            rows += 1
                end += WIN
        subject_meta[sid] = {"rows": rows, "n_trials": len(trials),
                             "trial_len_s": blk_len,
                             "labels": [tr.label for tr in trials]}
        raw.close()
        print(f"[{ci}/{len(paths)}] sub-{sid} rows={rows} trials={len(trials)} "
              f"miss={missing} elapsed={time.perf_counter() - t0:.1f}s", flush=True)

    F = np.asarray(F, dtype=np.float32)
    arr = dict(F=F, y=np.asarray(Y, dtype=np.int8), g=np.asarray(G),
               t=np.asarray(T, dtype=np.float32), blk=np.asarray(B, dtype=np.int8),
               pos=np.asarray(P, dtype=np.int32))
    np.savez_compressed(HERE / "realorder.npz", **arr)
    manifest = {"fs": FS, "window_s": WIN, "stride_s": WIN, "n_windows": int(len(Y)),
                "n_subjects": int(len(subject_meta)), "unmatched_windows": int(missing),
                "subjects": subject_meta,
                "note": "F = database 的 858 维 EEG 特征在前、20 维肌电特征在后，按原始波形 md5 精确对齐；"
                        "t 为窗口末端相对 BDF 首样本秒数；blk 为被试内 trial 序号；pos 为被试内真实顺序位置"}
    (HERE / "realorder_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[done] rows={len(Y)} subjects={len(subject_meta)} unmatched={missing} "
          f"F={F.shape} elapsed={time.perf_counter() - t0:.1f}s", flush=True)
    if missing:
        print("[warn] 存在未命中窗口：特征不可用，后续实验需排除或重算", flush=True)


if __name__ == "__main__":
    main()
