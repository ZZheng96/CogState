"""端到端 u_assume 敏感性：57 份 BDF 只读一遍，再用导出的表对 u=0..12 重算指标。

为什么需要：P6 的端到端结果是 u_assume=9 的单一取值。若该取值只是碰巧，结论就不稳。
这里给出完整的 bACC/AUC 曲线，用来判断"假定前导 54–60 s"这一策略的稳健性。

运行：py -3.11 test/10_projects/protocol_20261102/sweep_u.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cogstate import bad_intervals, load_trials  # noqa: E402
from protocol_timeline import load_table  # noqa: E402

CHUNK = 137
US = range(0, 13)


def main():
    import mne
    tab = load_table(ROOT / "src" / "protocol_timeline_table.json").table
    rows = []          # (t_end, label)
    t0 = time.perf_counter()
    for i, csv in enumerate(sorted((ROOT / "EEG_dataset").glob("*/*_events.csv")), 1):
        sid = csv.parent.name
        bdf = next(csv.parent.glob("*.bdf"))
        raw = mne.io.read_raw_bdf(str(bdf), preload=True, verbose="ERROR")
        fs, n = float(raw.info["sfreq"]), raw.n_times
        trials, bad = load_trials(str(csv), str(bdf)), bad_intervals(raw)
        raw.close()
        # 6 s 网格窗口末端（与 TimelineDecoder 的推进方式一致：每 1500 点一个窗）
        out = [k * 1500 / fs for k in range(1, n // 1500 + 1)]
        for t_end in out:
            a = t_end - 6.0
            lab = next((tr.label for tr in trials if tr.start < t_end <= tr.end + 1e-6), None)
            if lab is None or any(a < b1 and t_end > b0 for b0, b1 in bad):
                continue
            rows.append((t_end, lab, sid))
        print(f"[{i}/57] sub-{sid} scored={sum(1 for r in rows if r[2] == sid)} "
              f"elapsed={time.perf_counter() - t0:.0f}s", flush=True)

    t_end = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    g = np.array([r[2] for r in rows])
    res = {"protocol": "end-to-end BDF, u sweep", "chunk": CHUNK, "n_windows": int(len(y)),
           "n_subjects": int(len(np.unique(g))), "u": {}}
    for u in US:
        idx = np.clip(np.round(t_end / 6.0).astype(int) - u, 0, tab.size - 1)
        p = tab[idx]
        yh = (p >= 0.5).astype(int)
        per = [balanced_accuracy_score(y[g == s], yh[g == s]) for s in np.unique(g)]
        res["u"][str(u)] = {"AUC": float(roc_auc_score(y, p)),
                            "bACC": float(balanced_accuracy_score(y, yh)),
                            "F1_macro": float(f1_score(y, yh, average="macro")),
                            "subject_macro_bACC": float(np.mean(per)),
                            "n_ge_0.9": int(np.sum(np.array(per) >= 0.9))}
        r = res["u"][str(u)]
        print(f"[u={u:2d}] AUC={r['AUC']:.4f} bACC={r['bACC']:.4f} F1m={r['F1_macro']:.4f} "
              f"macro={r['subject_macro_bACC']:.4f} n>=.9={r['n_ge_0.9']}", flush=True)
    res["seconds"] = time.perf_counter() - t0
    (HERE / "sweep_u.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[done] -> sweep_u.json {res['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
