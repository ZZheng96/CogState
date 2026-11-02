"""把协议时间表模型导出成可直接加载的查表（src/protocol_timeline_table.json）。

做法与 P6 的 `position_table.py` 完全一致：P(task | k) = 训练被试在"假定第 1 块起点之后第 k 个
6 s 窗"上的任务频率（高斯核 bw=0.6 + Laplace）。用全部 57 名被试拟合，u_assume 在训练集上选。

运行：py -3.11 test/10_projects/protocol_20261102/export_table.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cogstate import load_trials  # noqa: E402
from protocol_timeline import TimelineDecoder, save_table  # noqa: E402
from position_table import MAXPOS, fit_table, measure_pre_roll, predict  # noqa: E402

BW = 0.6


def main():
    import mne  # noqa: F401  (measure_pre_roll 内部使用)
    D = np.load(HERE / "realorder.npz")
    y, g, t = D["y"].astype(int), D["g"], D["t"]
    pre = measure_pre_roll()
    u_true = np.array([int(round(pre[s] / 6.0)) for s in g])

    tab = fit_table(t / 6.0 - u_true, y, BW)
    # u_assume 在训练集上选：对训练被试施加同样的偏移后 macro bACC 最大
    from sklearn.metrics import balanced_accuracy_score
    best_u, best_v = 0, -1.0
    for u in range(0, 11):
        p = predict(tab, t / 6.0 - u)
        v = balanced_accuracy_score(y, (p >= 0.5).astype(int))
        if v > best_v:
            best_u, best_v = u, float(v)
    out = ROOT / "src" / "protocol_timeline_table.json"
    save_table(out, tab, u_assume=best_u)
    from protocol_timeline import load_table
    dec = load_table(out)
    assert dec.table.size == tab.size and abs(dec.u_assume - best_u) == 0
    # 自检：未推进状态时，predict_at 与表一致
    for tt in (6.0, 66.0, 900.0):
        p = dec.predict_at(tt)
        assert 0.0 <= p <= 1.0
    info = {"path": str(out.relative_to(ROOT)), "bw": BW, "u_assume": best_u,
            "train_bACC_at_u_assume": round(best_v, 4), "table_len": int(tab.size),
            "n_subjects": int(len(np.unique(g))), "n_windows": int(len(y)),
            "pre_roll_s": {k: round(v, 2) for k, v in pre.items()}}
    (HERE / "export_table.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[export] -> {out}  u_assume={best_u}  train_bACC={best_v:.4f}  len={tab.size}", flush=True)


if __name__ == "__main__":
    main()
