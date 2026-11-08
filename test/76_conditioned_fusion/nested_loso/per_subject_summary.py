"""汇总嵌套 LOSO 逐被试指标（只读结果 JSON，不重训）。"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULT = HERE / "nested_loso_results.json"
METHODS = ("w0.50", "conditioned", "conditioned_shrink", "mlp")
PARTS = ("full", "early", "late")
METRICS = ("auc", "f1", "bacc")


def main():
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    rows = result["folds"]
    assert len(rows) == 57

    def g(row, m, part, k):
        return row["metrics"][m][part][k]

    lines = ["# 嵌套 LOSO 逐被试表现（conditioned_shrink vs 固定 w=.5）", "",
             "57 人严格外层 LOSO，每人完全留出；指标为该被试自身评价段上的 AUC/F1/bACC。", "",
             "## 全段逐被试", "",
             "|被试|shrink AUC|w.5 AUC|ΔAUC|shrink F1|w.5 F1|shrink bACC|w.5 bACC|ΔbACC|",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        a, b = g(row, "conditioned_shrink", "full", "auc"), g(row, "w0.50", "full", "auc")
        f, fb = g(row, "conditioned_shrink", "full", "f1"), g(row, "w0.50", "full", "f1")
        c, cb = g(row, "conditioned_shrink", "full", "bacc"), g(row, "w0.50", "full", "bacc")
        lines.append(f"|{row['target']}|{a:.4f}|{b:.4f}|{a-b:+.4f}|{f:.4f}|{fb:.4f}|{c:.4f}|{cb:.4f}|{c-cb:+.4f}|")

    lines += ["", "## 分布统计", "", "|方法|阶段|指标|均值|中位数|最小|最大|", "|---|---|---|---:|---:|---:|---:|"]
    for m in METHODS:
        for part in PARTS:
            for k in METRICS:
                vals = [g(r, m, part, k) for r in rows if g(r, m, part, k) is not None]
                lines.append(f"|{m}|{part}|{k}|{np.mean(vals):.6f}|{np.median(vals):.6f}|{min(vals):.4f}|{max(vals):.4f}|")

    lines += ["", "## 逐被试改善/退步（shrink 相对 w=.5）", "", "|阶段|指标|改善|退步|持平|", "|---|---|---:|---:|---:|"]
    for part in PARTS:
        for k in METRICS:
            imp = reg = eq = 0
            for r in rows:
                a, b = g(r, "conditioned_shrink", part, k), g(r, "w0.50", part, k)
                if a is None or b is None:
                    continue
                imp += a > b
                reg += a < b
                eq += a == b
            lines.append(f"|{part}|{k}|{imp}|{reg}|{eq}|")

    lines += ["", "## 达标人数（bACC≥0.9 / ≥0.8）", "", "|方法|阶段|bACC≥0.9|bACC≥0.8|", "|---|---|---:|---:|"]
    for m in METHODS:
        for part in PARTS:
            vals = [g(r, m, part, "bacc") for r in rows if g(r, m, part, "bacc") is not None]
            lines.append(f"|{m}|{part}|{sum(v >= .9 for v in vals)}|{sum(v >= .8 for v in vals)}|")

    (HERE / "报告_逐被试表现.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:8]))
    print("written", HERE / "报告_逐被试表现.md")


if __name__ == "__main__":
    main()
