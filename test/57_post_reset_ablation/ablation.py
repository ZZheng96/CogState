from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TESTS = ROOT / "test"
sys.path.insert(0, str(TESTS / "17_compliant"))
import loso  # noqa: E402

SCORES = TESTS / "53_e17_post_loso" / "e53_scores.npz"
OUT_JSON = HERE / "results.json"
OUT_CSV = HERE / "results.csv"
OUT_MD = HERE / "报告_57_后处理重置消融.md"


def metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import f1_score, roc_auc_score
    p = score > 0
    return {"bacc": float(0.5 * (p[y == 1].mean() + (1 - p[y == 0]).mean())),
            "auc": float(roc_auc_score(y, score)), "f1": float(f1_score(y, p))}


class Detrend:
    def __init__(self, hl=1200):
        self.a = 1 - 0.5 ** (1 / hl)
        self.mean = None

    def update(self, x):
        if self.mean is None:
            self.mean = float(x)
        else:
            self.mean += self.a * (float(x) - self.mean)
        return float(x) - self.mean


class HMM:
    def __init__(self, dwell=300):
        self.stay = math.log1p(-1 / dwell)
        self.switch = -math.log(dwell)
        self.lp = np.array([-math.log(2), -math.log(2)], dtype=float)
        self.started = False

    def update(self, x):
        em = np.array([-np.logaddexp(0, x), -np.logaddexp(0, -x)])
        if self.started:
            pred = np.array([np.logaddexp(self.lp[0] + self.stay, self.lp[1] + self.switch),
                             np.logaddexp(self.lp[0] + self.switch, self.lp[1] + self.stay)])
        else:
            pred = self.lp
        u = pred + em
        self.lp = u - np.logaddexp(u[0], u[1])
        self.started = True
        return float(self.lp[1] - self.lp[0])


def process(raw, blocks, mode):
    """All state changes occur before current frame; block IDs are evaluation metadata only."""
    out = np.empty(len(raw), dtype=float)
    dt = Detrend(1200)
    hmm = HMM(300)
    prev_block = None
    for i, (x, b) in enumerate(zip(raw, blocks)):
        new_block = prev_block is not None and b != prev_block
        reset_both = mode in {"reset_each", "reset_block4"} and ((mode == "reset_each") or b == 4)
        reset_h = mode in {"hmm_reset_each", "reset_each", "reset_block4"} and ((mode == "hmm_reset_each") or mode == "reset_each" or b == 4)
        reset_d = mode in {"detrend_reset_each", "reset_each", "reset_block4"} and ((mode == "detrend_reset_each") or mode == "reset_each" or b == 4)
        if new_block and reset_both:
            dt, hmm = Detrend(1200), HMM(300)
        elif new_block and reset_d:
            dt = Detrend(1200)
        elif new_block and reset_h:
            hmm = HMM(300)
        if mode == "raw":
            out[i] = x
        elif mode == "dt1200":
            out[i] = dt.update(x)
        else:
            z = dt.update(x)
            out[i] = hmm.update(z)
        prev_block = b
    return out


def group_map():
    p = TESTS / "43_group_analysis" / "group_members.csv"
    with p.open(encoding="utf-8-sig") as f:
        return {r["subject"]: r["group"] for r in csv.DictReader(f)}


def aggregate(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in ("bacc", "auc", "f1")}


def main():
    subs, D = loso.load()
    data = np.load(SCORES)
    groups = group_map()
    modes = {
        "raw": "1 raw原始分数", "dt1200": "2 dt1200无HMM", "dt1200_hmm": "3 dt1200+HMM d300基线",
        "reset_each": "4 每个block开始重置dt/HMM", "reset_block4": "5 仅块4开始重置dt/HMM",
        "hmm_reset_each": "6 仅HMM重置", "detrend_reset_each": "7 仅detrend重置",
    }
    all_results, csv_rows = {}, []
    for mode, label in modes.items():
        per = {}
        for q in subs:
            y, b = D[q][1], D[q][2]
            s = process(data[q].astype(float), b, mode)
            masks = {"full": np.ones(len(y), bool), "first600": np.arange(len(y)) < 600, "block45": b >= 4}
            per[q] = {name: metrics(y[m], s[m]) for name, m in masks.items()}
        scopes = {"all": subs, "high": [q for q in subs if groups[q] == "high"], "low": [q for q in subs if groups[q] == "low"]}
        agg = {scope: {part: aggregate([per[q][part] for q in qs]) for part in ("full", "first600", "block45")} for scope, qs in scopes.items()}
        all_results[mode] = {"label": label, "aggregate": agg, "per_subject": per}
        for scope, parts in agg.items():
            for part, m in parts.items():
                csv_rows.append({"mode": mode, "label": label, "scope": scope, "part": part, **m})
    OUT_JSON.write_text(json.dumps({"source": str(SCORES), "modes": all_results}, ensure_ascii=False, indent=2), encoding="utf-8")
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mode", "label", "scope", "part", "bacc", "auc", "f1"])
        w.writeheader(); w.writerows(csv_rows)
    base = all_results["dt1200_hmm"]["aggregate"]
    lines = ["# 57 后处理重置消融（E53 LOSO 分数）", "", f"数据源：`{SCORES}`。所有后处理按时间顺序逐帧递推，block 仅在边界到达时触发状态重置；未重新训练。", "", "## 当前基线", "", "| 范围 | 片段 | bACC | AUC | F1 |", "|---|---|---:|---:|---:|"]
    for scope in ("all", "high", "low"):
        for part, title in (("full", "全段"), ("first600", "前600"), ("block45", "块4-5")):
            m = base[scope][part]; lines.append(f"| {scope} | {title} | {m['bacc']:.4f} | {m['auc']:.4f} | {m['f1']:.4f} |")
    lines += ["", "## 全体方案比较", "", "| 方案 | 全段 bACC/AUC/F1 | 前600 bACC/AUC/F1 | 块4-5 bACC/AUC/F1 | 块4-5 bACC变化 | 全段bACC变化 |", "|---|---|---|---|---:|---:|"]
    for mode, label in modes.items():
        a = all_results[mode]["aggregate"]["all"]
        fmt = lambda m: f"{m['bacc']:.4f}/{m['auc']:.4f}/{m['f1']:.4f}"
        lines.append(f"| {label} | {fmt(a['full'])} | {fmt(a['first600'])} | {fmt(a['block45'])} | {a['block45']['bacc']-base['all']['block45']['bacc']:+.4f} | {a['full']['bacc']-base['all']['full']['bacc']:+.4f} |")
    lines += ["", "## 结论", "", "- “改善块4-5且全段不明显下降”按块4-5 bACC 上升、全段 bACC 下降不超过 0.005 判定。", "- 逐方案是否满足："]
    for mode, label in modes.items():
        a = all_results[mode]["aggregate"]["all"]
        ok = a["block45"]["bacc"] > base["all"]["block45"]["bacc"] and a["full"]["bacc"] >= base["all"]["full"]["bacc"] - 0.005
        lines.append(f"  - **{label}**：{'是' if ok else '否'}。")
    lines += ["", "## 文件", "", f"- JSON：`{OUT_JSON}`", f"- CSV：`{OUT_CSV}`", f"- 本报告：`{OUT_MD}`"]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for mode, label in modes.items():
        a = all_results[mode]["aggregate"]["all"]
        print(f"{mode:20s} full {a['full']['bacc']:.4f}/{a['full']['auc']:.4f}/{a['full']['f1']:.4f} "
              f"first600 {a['first600']['bacc']:.4f}/{a['first600']['auc']:.4f}/{a['first600']['f1']:.4f} "
              f"block45 {a['block45']['bacc']:.4f}/{a['block45']['auc']:.4f}/{a['block45']['f1']:.4f}")


if __name__ == "__main__":
    main()
