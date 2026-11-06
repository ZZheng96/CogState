"""E54 诊断：低组后段"方向退化"到底是什么。

问题：E43/E44 发现低组 29 人块4–5 的 AUC 掉到 .56，4 人（004/009/010/050）整体 AUC<.5。
本脚本只做诊断，不训练新模型，回答三个问题：
  Q1 方向退化是"类间均值相向收敛"还是"类内方差爆炸"？
  Q2 退化是否与无标签可观测的量（分数方差、翻转率、特征漂移）相关？
  Q3 若允许用前 600 帧的无标签信息做一次在线校准（不含标签），能否预测后段方向？

运行：.venv\\Scripts\\python.exe test/54_direction_diag/e54.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent
sys.path.insert(0, str(TESTS / "17_compliant"))
import loso as L  # noqa: E402

DEV = "cuda"
SCORES = TESTS / "53_e17_post_loso" / "e53_scores.npz"
OUT = HERE / "e54.json"


def groups():
    with open(TESTS / "43_group_analysis" / "group_members.csv", encoding="utf-8-sig") as f:
        return {r["subject"]: r["group"] for r in csv.DictReader(f)}


def dprime(y, s):
    """d' = (mean_task - mean_relax) / pooled_sd，衡量分离度（与阈值无关）。"""
    a, b = s[y == 1], s[y == 0]
    if len(a) < 5 or len(b) < 5:
        return float("nan")
    sd = np.sqrt(0.5 * (a.var() + b.var())) + 1e-9
    return float((a.mean() - b.mean()) / sd)


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    if len(set(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def main():
    subs, D = L.load()
    grp = groups()
    data = np.load(SCORES)
    rows = {}
    for q in subs:
        s, y, b = data[q], D[q][1], D[q][2]
        r = {}
        for name, m in (("b01", b <= 1), ("b23", (b >= 2) & (b <= 3)), ("b45", b >= 4)):
            r[f"d_{name}"] = dprime(y[m], s[m])
            r[f"auc_{name}"] = auc(y[m], s[m])
            r[f"sd_task_{name}"] = float(s[m][y[m] == 1].std())
            r[f"sd_relax_{name}"] = float(s[m][y[m] == 0].std())
            r[f"mean_task_{name}"] = float(s[m][y[m] == 1].mean())
            r[f"mean_relax_{name}"] = float(s[m][y[m] == 0].mean())
        # 无标签可观测量：分数方差、符号翻转率、分数漂移
        r["sd_all"] = float(s.std())
        sg = (s > 0).astype(int)
        r["flip_rate"] = float(np.mean(sg[1:] != sg[:-1]))
        r["drift"] = float(s[-200:].mean() - s[:200].mean())
        r["group"] = grp[q]
        rows[q] = r

    def agg(qs, keys):
        return {k: float(np.nanmean([rows[q][k] for q in qs])) for k in keys}

    keys = ["d_b01", "d_b23", "d_b45", "auc_b01", "auc_b23", "auc_b45",
            "sd_task_b45", "sd_relax_b45", "mean_task_b45", "mean_relax_b45",
            "sd_all", "flip_rate", "drift"]
    high = [q for q in subs if grp[q] == "high"]
    low = [q for q in subs if grp[q] == "low"]
    rev = ["004", "009", "010", "050"]
    summary = {"high": agg(high, keys), "low": agg(low, keys), "rev4": agg(rev, keys), "all": agg(subs, keys)}
    for k, v in summary.items():
        print(k, json.dumps({a: round(b, 3) for a, b in v.items()}, ensure_ascii=False), flush=True)

    # Q2：无标签量与后段 d' 的相关（全体 + 低组）
    from scipy.stats import spearmanr
    corr = {}
    for name, qs in (("all", subs), ("low", low)):
        for feat in ("sd_all", "flip_rate", "drift", "sd_task_b45", "sd_relax_b45"):
            x = [rows[q][feat] for q in qs]
            yv = [rows[q]["d_b45"] for q in qs]
            rho, p = spearmanr(x, yv)
            corr[f"{name}|{feat}"] = {"rho": float(rho), "p": float(p)}
    print("corr", json.dumps({k: round(v["rho"], 3) for k, v in corr.items()}, ensure_ascii=False), flush=True)

    # Q3：前段 d' 能否预测后段 d'（若可以，说明退化是"人"的属性而非随机）
    ok = [q for q in subs if np.isfinite(rows[q]["d_b01"]) and np.isfinite(rows[q]["d_b45"])]
    ok_low = [q for q in ok if grp[q] == "low"]
    rho_all, p_all = spearmanr([rows[q]["d_b01"] for q in ok], [rows[q]["d_b45"] for q in ok])
    rho_low, p_low = spearmanr([rows[q]["d_b01"] for q in ok_low], [rows[q]["d_b45"] for q in ok_low])
    pred = {"all": {"n": len(ok), "rho": float(rho_all), "p": float(p_all)},
            "low": {"n": len(ok_low), "rho": float(rho_low), "p": float(p_low)}}
    print("d_b01 -> d_b45", pred, flush=True)

    OUT.write_text(json.dumps({"summary": summary, "corr": corr, "front_to_late": pred,
                               "per_subject": rows}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
