"""E36：核验 E33/E34/E35 汇总数、复现一致性与外层标签泄漏（扰动检验）。

“重跑 vs 备份”需先把原 e34.json/e34_ev.npz/e35.json 复制到 backup/ 再重跑 e34.py/e35.py；
核验完成且一致后备份已删除（结果见 verify_log.txt）。

运行：.venv\Scripts\python.exe test/36_verify_e33_35/verify.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
T = HERE.parent
sys.path.insert(0, str(T / "33_response_routing"))
import e33  # noqa: E402


def summ(v: np.ndarray) -> str:
    v = np.asarray(v, dtype=float)
    return (f"bACC {np.nanmean(v[:, 0]):.6f} AUC {np.nanmean(v[:, 1]):.6f} F1 {np.nanmean(v[:, 2]):.6f} "
            f"n90 {int(np.nansum(v[:, 0] >= .9))} min {np.nanmin(v[:, 0]):.6f}")


def per_rows(per: dict, keys=("bacc", "auc", "f1")) -> np.ndarray:
    return np.array([[p[k] for k in keys] if isinstance(p, dict) else p for p in per.values()])


def check_e35():
    print("== E35 ==")
    r = json.loads((T / "35_subject_oracle" / "e35.json").read_text(encoding="utf-8"))
    print("json 逐人重算:", summ(per_rows(r["per_subject"])))
    b = json.loads((HERE / "backup" / "e35.json").read_text(encoding="utf-8"))
    print("重跑 vs 备份 完全一致:", r == b)


def check_e34():
    print("== E34 ==")
    r = json.loads((T / "34_transductive_threshold" / "e34.json").read_text(encoding="utf-8"))
    for k in ("fixed_best_in_sample_selection", "nested_loso"):
        print(k, "json 逐人重算:", summ(per_rows(r[k]["per_subject"])))
    print("test_oracle 存档:", r["test_oracle"])
    b = json.loads((HERE / "backup" / "e34.json").read_text(encoding="utf-8"))
    print("重跑 json vs 备份 完全一致:", r == b)
    ev, ev0 = np.load(T / "34_transductive_threshold" / "e34_ev.npz")["EV"], np.load(HERE / "backup" / "e34_ev.npz")["EV"]
    print("重跑 EV vs 备份 最大差:", float(np.nanmax(np.abs(ev - ev0))), "NaN 位置一致:", bool((np.isnan(ev) == np.isnan(ev0)).all()))
    methods = [p["method"] for p in r["picks"].values()]
    print("LOSO 所选阈值方法计数:", {m: methods.count(m) for m in set(methods)})


def check_e33():
    print("== E33 ==")
    d = json.loads((T / "33_response_routing" / "e33_results.json").read_text(encoding="utf-8"))
    for ds in ("E23", "E28"):
        res = d[ds]["results"]
        for k in ("omnibus_nested", "soft_all_raw", "fixed_pool_anchor", "global_loso", "oracle_test_upper"):
            if k in res:
                print(ds, k, "json 逐人重算:", summ(per_rows(res[k]["per_subject"])))
    # E28：从 traces 重建 omnibus，并用 e28_ev.npz 原始指标复核逐人分数。
    src = e33.load_source("E28")
    r = d["E28"]
    specs = e33.model_specs()
    tr = r["traces"]
    choices = []
    for i in range(57):
        cand = [(tr[s["name"]]["inner_scores"][i][h], s["name"], h) for s in specs for h in range(len(s["grid"]))]
        _, name, h = max(cand, key=lambda z: z[0])
        choices.append(tr[name]["outer_choices"][i][h])
    vals = src["metrics"][np.arange(57), choices]
    print("E28 omnibus 由 traces+EV 重建:", summ(vals))
    oracle = np.nanmax(src["metrics"][..., 0], 1)
    print("omnibus 选中即 oracle 最优的人数:", int((np.abs(vals[:, 0] - oracle) < 1e-9).sum()), "/57")

    # 扰动检验：把外层目标的测试 bACC 行替换为随机数，外层选择应完全不变。
    f, _ = e33.get_cal("E28", src)
    y0 = torch.as_tensor(src["metrics"][..., 0], device=e33.DEV, dtype=torch.float32)
    cal = torch.as_tensor(f[:, 0], device=e33.DEV, dtype=torch.float32)
    avail = torch.isfinite(y0)
    gen = torch.Generator(device=e33.DEV).manual_seed(0)
    changed, total = 0, 0
    targets = [0, 10, 25, 33, 56]  # 33 = 034 所在位置附近，另含普通被试
    for spec in specs:
        x = torch.as_tensor(e33.feature_view(f, spec["view"]), device=e33.DEV, dtype=torch.float32)
        fn = {"knn": e33.run_knn, "ridge": e33.run_ridge, "pcr": e33.run_pcr}[spec["kind"]]
        for o in targets:
            y = y0.clone()
            y[o] = torch.where(avail[o], torch.rand(y.shape[1], device=e33.DEV, generator=gen), y[o])
            si, co = fn(spec, x, y, cal, avail, o)
            si0, co0 = fn(spec, x, y0, cal, avail, o)
            total += 1
            changed += int(co != co0 or not np.allclose(si, si0))
            changed += int(co0 != tr[spec["name"]]["outer_choices"][o])
    print(f"扰动外层目标测试标签后 内层得分/外层选择变化或与存档不符: {changed}/{total} (应为0)")


if __name__ == "__main__":
    check_e35()
    check_e34()
    check_e33()
