"""E29 挑出瓶颈人：两组合并不可行时，去掉瓶颈成员 m 后看 (A∪B)\\{m} 是否可行，再把 m 塞进其他组 C。
都可行 → 组数 −1。
A 阶段：G4+G12 去 001 → G4+035，001 依次试 17 个其他组。
B 阶段：recipe_cross 最接近的前 NP 对，同样操作（瓶颈 = 合并最优配方下 bACC 最低者，最多试 2 人）。
运行：python -u test/29_closest_pair/pluck.py → pluck.log / pluck.json
"""
from __future__ import annotations

import json
import time

from e29lib import GROUPS, HERE, TH, subs, try_merge

NP = 12
t0 = time.time()
log = open(HERE / "pluck.log", "w", encoding="utf-8")
out = []


def pr(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.write(s + "\n")
    log.flush()


def show(tag, U, r):
    pr(f"  {tag}: min {r['v']:.3f} {r['kind']} {r['recipe']}  [{time.time() - t0:.0f}s]")
    pr("     " + " ".join(f"{subs[j]}={x[0]:.3f}" for j, x in zip(U, r["per"])))


def place(m, skip):
    """把被试 m 试塞进其他组，返回可行列表。"""
    ok = []
    for c, g in enumerate(GROUPS):
        if c in skip:
            continue
        U = g + [m]
        r = try_merge(U)
        pr(f"    {subs[m]} → G{c}({len(g)}人): min {r['v']:.3f} {r['kind']}  [{time.time() - t0:.0f}s]")
        if r["v"] >= TH:
            ok.append((c, r))
            show(f"    ✔ G{c}+{subs[m]}", U, r)
    return ok


def attempt(a, b):
    U = GROUPS[a] + GROUPS[b]
    r = try_merge(U)
    show(f"G{a}+G{b} 全体", U, r)
    rec = dict(a=a, b=b, full=r["v"], tries=[])
    if r["v"] >= TH:
        rec["result"] = "直接可行"
        return rec
    order = sorted(range(len(U)), key=lambda i: r["per"][i][0])[:2]
    for i in order:
        m = U[i]
        V = [j for j in U if j != m]
        r2 = try_merge(V)
        show(f"G{a}+G{b} 去 {subs[m]}", V, r2)
        t = dict(drop=subs[m], rest=r2["v"], rest_recipe=r2["recipe"], places=[])
        if r2["v"] >= TH:
            for c, r3 in place(m, {a, b}):
                t["places"].append(dict(group=c, v=r3["v"], recipe=r3["recipe"]))
        rec["tries"].append(t)
        if t["places"]:
            rec["result"] = f"可行：去 {subs[m]}，{subs[m]} 并入 G{t['places'][0]['group']}"
            pr(f"  ★ {rec['result']}")
            return rec
    rec["result"] = "不可行"
    return rec


pairs = json.load(open(HERE / "recipe_cross.json", encoding="utf-8"))["pairs"][:NP]
for i, d in enumerate(pairs):
    pr(f"\n=== [{i + 1}/{len(pairs)}] G{d['a']}+G{d['b']}（交叉 {d['v']:.3f}）===")
    out.append(attempt(d["a"], d["b"]))
    json.dump(out, open(HERE / "pluck.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
pr("\n汇总：")
for o in out:
    pr(f"  G{o['a']}+G{o['b']}: {o['result']}")
pr(f"done [{time.time() - t0:.0f}s]")
