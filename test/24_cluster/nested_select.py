"""E24 嵌套选择：分组方案（特征集×算法×k，top1 组内选法）本身也由 LOSO 选，消除"扫 k 挑最好"的选择偏差。
对留出 i：每个方案在 j≠i 上用 others(lab, j, excl={i}) 的 top1 评估 → 取平均 bACC 最高的方案 → 用于 i。
输出 nested_select.json。需先运行 prof_cluster.py（prof_groups.json）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clib import HERE, load, agg  # noqa: E402

R = load()
nq = R["nq"]
TB = R["ev"][:, 0].reshape(nq, -1)
TA, TF = R["ev"][:, 1].reshape(nq, -1), R["ev"][:, 2].reshape(nq, -1)
G = {k: np.array(v) for k, v in json.load(open(HERE / "prof_groups.json")).items()}


def top1(lab, j, excl):
    o = [x for x in np.where(lab == lab[j])[0] if x != j and x not in excl]
    if not o:
        o = [x for x in range(nq) if x != j and x not in excl]
    mu = np.nanmean(TB[o], 0)
    return int(np.argmax(np.where(np.isnan(TB[j]), -np.inf, mu)))


def main():
    res = {}
    for tag, names in (("all", list(G)), ("ward", [n for n in G if "_ward_" in n]),
                       ("ax3", [n for n in G if n.startswith("ax3")]), ("ax5", [n for n in G if n.startswith("ax5")]),
                       ("cal", [n for n in G if n.startswith("cal")])):
        vals, pick = [], []
        for i in range(nq):
            sc = [np.nanmean([TB[j, top1(G[n], j, (i,))] for j in range(nq) if j != i]) for n in names]
            n = names[int(np.argmax(sc))]
            c = top1(G[n], i, ())
            vals.append((TB[i, c], TA[i, c], TF[i, c]))
            pick.append(n)
        a = agg(vals)
        a["picked"] = {n: pick.count(n) for n in sorted(set(pick))}
        res[tag] = a
        print(f"nested[{tag}] bACC {a['bacc']:.3f} AUC {a['auc']:.3f} F1 {a['f1']:.3f} n90 {a['n90']} {a['picked']}",
              flush=True)
    json.dump(res, open(HERE / "nested_select.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
