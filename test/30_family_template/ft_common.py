"""E30 公共：家族划分、族内原配方解析（cfg 下标 + 阈值下标）、族成员。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "29_closest_pair"))
from e29lib import GROUPS, R, THR2, TH, base, cfg_name, dev, ev2b, metrics, sd, subs  # noqa: E402,F401

FAM = {"A": [0, 2, 3, 6, 7, 9, 10, 11, 13, 16], "B": [1, 4, 8, 12, 14, 17], "C": [5, 15]}
F = sys.argv[1] if len(sys.argv) > 1 else "A"
G = json.load(open(HERE.parent / "26_merge2" / "merge2.json", encoding="utf-8"))["groups"]
NAME2C = {}
for c in range(ev2b.shape[1]):
    n = cfg_name(c, base)
    NAME2C[n] = c
    if "+" in n:
        a, b = n.split("+")
        NAME2C[f"{b}+{a}"] = c
TNAME = [t[0] for t in THR2]


def recipe_of(g):
    lab = g["recipe"]
    if lab.startswith("专属"):  # excl.json 只有测试指标、无分数序列，无法在校准块上评分 → 不进候选
        return None
    body, thr = [x.strip() for x in lab.rsplit("|", 1)]
    body = body[2:-1] if body.startswith("z[") else body
    cs = [NAME2C[p.strip()] for p in body.split(";")]
    return lab, cs, TNAME.index(thr)


gi = FAM[F]
REC, RG = [], []
for i in gi:
    r = recipe_of(G[i])
    if r:
        REC.append(r)
        RG.append(i)
U = [j for i in gi for j in GROUPS[i]]
home = {j: (RG.index(i) if i in RG else -1) for i in gi for j in GROUPS[i]}
