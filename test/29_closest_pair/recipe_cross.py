"""E29 快速版：18 组配方交叉套用（不搜索），看哪两组的方法最接近。
cross[g, h] = 组 g 的配方用在组 h 成员上的组内最小 bACC。
两组合并的直接可行度 = max(用 g 的配方覆盖 g∪h 的最小值, 用 h 的配方覆盖 g∪h 的最小值)。
运行：python -u test/29_closest_pair/recipe_cross.py → recipe_cross.json / recipe_cross.log
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
T = HERE.parent
sys.path.insert(0, str(T / "26_merge2"))
sys.path.insert(0, str(T / "25_merge"))
from e26lib import metrics, subj_data, zpair  # noqa: E402
from clib import load  # noqa: E402
from merge import build_pool  # noqa: E402

TH = .9
log = open(HERE / "recipe_cross.log", "w", encoding="utf-8")


def pr(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.write(s + "\n")


R = load()
subs, nq = R["subs"], R["nq"]
ev2 = np.load(T / "26_merge2" / "ev2.npz")["ev2"]
M0, _, _, _ = build_pool(R)
Mx = np.concatenate([M0, ev2.transpose(0, 3, 1, 2).reshape(nq, 3, -1)], 2)
del M0
G = json.load(open(T / "26_merge2" / "merge2.json", encoding="utf-8"))["groups"]
groups = [[subs.index(q) for q in g["members"]] for g in G]
SD = {}


def rec_eval(rk, j):
    """配方 rk 用在被试 j 上 → (bACC, AUC, F1)。"""
    if rk[0] == "s":
        return Mx[j, :, rk[1]]
    a, b, t = rk[1]
    if j not in SD:
        SD[j] = subj_data(R, subs[j])
    C, cal_t, tst, yy = SD[j]
    return metrics(zpair(C, cal_t, [a], [b]), tst, yy)[0, t].cpu().numpy()


ng = len(G)
V = np.array([[rec_eval(G[g]["rk"], j) for j in range(nq)] for g in range(ng)])  # (配方, 被试, 3)
B = np.nan_to_num(V[..., 0], nan=0.)
cross = np.array([[B[g, groups[h]].min() for h in range(ng)] for g in range(ng)])
pairs = []
for a in range(ng):
    for b in range(a + 1, ng):
        u = groups[a] + groups[b]
        va, vb = B[a, u].min(), B[b, u].min()
        k = a if va >= vb else b
        pairs.append(dict(a=a, b=b, v=float(max(va, vb)), use=k, va=float(va), vb=float(vb),
                          mean=float(B[k, u].mean())))
pairs.sort(key=lambda d: -d["v"])

pr("各组配方：")
for i, g in enumerate(G):
    pr(f"  G{i:<2} n={len(g['members'])} {g['recipe']}")
pr("\n配方交叉矩阵（行=配方来源组，列=被套用组，值=该组成员最小 bACC）：")
pr("      " + " ".join(f"G{h:<4}" for h in range(ng)))
for g in range(ng):
    pr(f"G{g:<4} " + " ".join(f"{cross[g, h]:.3f}" for h in range(ng)))
pr("\n直接套用最接近的 10 对（不做任何搜索）：")
for d in pairs[:10]:
    pr(f"  {d['v']:.3f}  G{d['a']}+G{d['b']}  用 G{d['use']} 配方（G{d['a']}配方 {d['va']:.3f} / G{d['b']}配方 {d['vb']:.3f}）"
       f"  组均值 {d['mean']:.3f}")
top = pairs[0]
u = groups[top["a"]] + groups[top["b"]]
pr(f"\n最接近一对 G{top['a']}+G{top['b']} 用 G{top['use']} 配方「{G[top['use']]['recipe']}」逐人：")
for j in u:
    x = V[top["use"], j]
    pr(f"  {subs[j]}: bACC {x[0]:.3f} AUC {x[1]:.3f} F1 {x[2]:.3f}{'  <.9' if x[0] < TH else ''}")
json.dump(dict(cross=cross.tolist(), pairs=pairs, recipes=[g["recipe"] for g in G],
               members=[g["members"] for g in G]), open(HERE / "recipe_cross.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
pr("done")
