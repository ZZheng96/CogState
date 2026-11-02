"""端到端部署验证：把导出的协议时间表解码器直接接到 57 份原始 BDF 上。

与 P5/P6 的差别：这里**不做任何人工对齐**——从文件第一个样本开始按 137 点分块喂入，
解码器自己按 6 s 网格输出 (t_end, p_task)，再用真实事件给每个窗口打标签算指标。
这最接近"赛事组把 BDF 丢给推理程序"的场景。

窗口记账：
  * 只统计完全落在某个 trial 内、且不与该记录的 BAD_ACQ_SKIP 重叠的窗口；
  * 标签取窗口末端所在 trial 的 phase（与训练切窗口径一致：窗从 trial_start 起每 6 s 一个）；
  * 其余窗口计入 "skipped"，不参与指标。

运行：py -3.11 test/10_projects/protocol_20261102/eval_stream_bdf.py [--subjects 001,002]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cogstate import FS, bad_intervals, load_trials  # noqa: E402
from protocol_timeline import TimelineDecoder, load_table  # noqa: E402

CHUNK = 137


def build_fold_tables():
    """折外表：每个 fold 只用其余被试的窗口拟合查表，u_assume 也只在训练被试上选。"""
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedGroupKFold

    from position_table import fit_table, measure_pre_roll, predict

    D = np.load(HERE / "realorder.npz")
    y, g, t = D["y"].astype(int), D["g"], D["t"]
    pre = measure_pre_roll()
    u_true = np.array([int(round(pre[s] / 6.0)) for s in g])
    fold = np.empty(len(y), int)
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y, g)):
        fold[te] = k
    out = {}
    for k in range(5):
        tr = np.flatnonzero(fold != k)
        tab = fit_table(t[tr] / 6.0 - u_true[tr], y[tr], 0.6)
        best_u, best_v = 0, -1.0
        for u in range(11):
            p = predict(tab, t[tr] / 6.0 - u)
            v = balanced_accuracy_score(y[tr], (p >= 0.5).astype(int))
            if v > best_v:
                best_u, best_v = u, float(v)
        subj = {s: k for s in np.unique(g[fold == k])}
        out.update({s: (tab, best_u) for s in subj})
        print(f"[fold-table {k}] u_assume={best_u} train_bACC={best_v:.4f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="")
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--oof", action="store_true",
                    help="用折外表（每个被试只由其余被试拟合）做严格折外评估；不加则用全量导出表")
    args = ap.parse_args()
    import mne
    fold_models = build_fold_tables() if args.oof else None
    base = load_table(ROOT / "src" / "protocol_timeline_table.json")
    print(f"[model] table_len={base.table.size} chunk={args.chunk} oof={args.oof}", flush=True)

    paths = sorted((ROOT / "EEG_dataset").glob("*/*_events.csv"))
    want = set(args.subjects.split(",")) if args.subjects else None
    P, Y, G, skipped = [], [], [], 0
    t0 = time.perf_counter()
    per = {}
    for i, csv in enumerate(paths, 1):
        sid = csv.parent.name
        if want and sid not in want:
            continue
        bdf = next(csv.parent.glob("*.bdf"))
        raw = mne.io.read_raw_bdf(str(bdf), preload=True, verbose="ERROR")
        fs, n = float(raw.info["sfreq"]), raw.n_times
        trials = load_trials(str(csv), str(bdf))
        bad = bad_intervals(raw)
        if fold_models is not None:
            tab_s, u_s = fold_models[sid]
            dec = TimelineDecoder(tab_s, u_assume=u_s)      # 折外表，严格折外
        else:
            dec = TimelineDecoder(base.table, u_assume=base.u_assume)
        out = []
        for s in range(0, n, args.chunk):
            out.extend(dec.push(raw.get_data(start=s, stop=min(n, s + args.chunk))[0]))
        raw.close()

        ys, ps = [], []
        for t_end, p in out:
            a, z = t_end - 6.0, t_end
            lab = None
            for tr in trials:
                if tr.start < z <= tr.end + 1e-6:
                    lab = tr.label
                    break
            if lab is None or any(a < b1 and z > b0 for b0, b1 in bad):
                skipped += 1
                continue
            ys.append(lab)
            ps.append(p)
        if ys:
            ys, ps = np.array(ys), np.array(ps)
            P.append(ps); Y.append(ys); G.append(np.array([sid] * len(ys)))
            per[sid] = {"n": int(len(ys)), "bACC": float(balanced_accuracy_score(ys, ps >= .5)),
                        "AUC": float(roc_auc_score(ys, ps)) if ys.min() != ys.max() else float("nan")}
        print(f"[{i}/{len(paths)}] sub-{sid} windows={len(out)} scored={len(ys)} "
              f"bACC={per.get(sid, {}).get('bACC', float('nan')):.4f} elapsed={time.perf_counter() - t0:.0f}s", flush=True)

    y = np.concatenate(Y); p = np.concatenate(P); g = np.concatenate(G)
    res = {"protocol": "end-to-end BDF streaming, no manual alignment",
           "chunk": args.chunk, "oof": bool(args.oof),
           "u_assume": ("per-fold" if args.oof else base.u_assume),
           "model": ("fold tables from realorder.npz" if args.oof else "src/protocol_timeline_table.json"),
           "n_windows": int(len(y)), "n_subjects": int(len(per)), "skipped_windows": int(skipped),
           "AUC": float(roc_auc_score(y, p)),
           "bACC": float(balanced_accuracy_score(y, p >= .5)),
           "F1_macro": float(f1_score(y, (p >= .5).astype(int), average="macro")),
           "F1_task": float(f1_score(y, (p >= .5).astype(int), average="binary", pos_label=1, zero_division=0)),
           "subject_macro_bACC": float(np.mean([v["bACC"] for v in per.values()])),
           "n_ge_0.9": int(sum(v["bACC"] >= 0.9 for v in per.values())),
           "per_subject": per, "seconds": time.perf_counter() - t0}
    print(f"[result] AUC={res['AUC']:.4f} bACC={res['bACC']:.4f} F1m={res['F1_macro']:.4f} "
          f"F1task={res['F1_task']:.4f} macro={res['subject_macro_bACC']:.4f} "
          f"n>=.9={res['n_ge_0.9']}/{res['n_subjects']} windows={res['n_windows']} skipped={skipped}", flush=True)
    (HERE / ("eval_stream_bdf_oof.json" if args.oof else "eval_stream_bdf.json")).write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[done] -> {'eval_stream_bdf_oof.json' if args.oof else 'eval_stream_bdf.json'} {res['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
