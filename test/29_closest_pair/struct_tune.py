"""E29 结构版：按配方部件（特征族+超参）找方法最接近的组对，再只在组内做方法细节微调。
1) 结构相似度（不算指标）：配方拆成 base 部件；相同部件 1 分，同族且超参只差一步 0.5 分；
   sim = 匹配分 / max(部件数)。专属组用该人 ev2 单人最优 cfg 的部件代替。
2) 组内微调：候选部件 = 两组配方部件 ∪ 其一步超参邻居（同族），
   候选 cfg = 单部件 + 两部件和（须在 1653 cfg 内），搜 单cfg / z 双cfg / z 三cfg × 85 阈值，
   只用两组成员计算，目标组内最小 bACC ≥ .9。
运行：python -u test/29_closest_pair/struct_tune.py [前N对=6] → struct_tune.log / struct_tune.json
"""
from __future__ import annotations

import itertools
import json
import re
import sys
import time

import numpy as np

from e29lib import (GROUPS, HERE, NT2, THR2, TH, _full, _search, _zrows, base, cfg_name, ev2b, subs, zpair)

NTOP = int(sys.argv[1]) if len(sys.argv) > 1 else 6
t0 = time.time()
log = open(HERE / "struct_tune.log", "w", encoding="utf-8")


def pr(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.write(s + "\n")
    log.flush()


G = json.load(open(HERE.parent / "26_merge2" / "merge2.json", encoding="utf-8"))["groups"]
NCFG = ev2b.shape[1]
NAME2C = {cfg_name(c, base): c for c in range(NCFG)}
for c in range(NCFG):  # a+b 与 b+a 同一 cfg
    n = cfg_name(c, base)
    if "+" in n:
        a, b = n.split("+")
        NAME2C[f"{b}+{a}"] = c

# ---------- 部件解析 ----------
PAT = [  # (族, 正则, 超参取值表)
    ("k", r"k(\d+)_w([\d.]+)$", [[400, 800, 1600, 4390], [0.6, 0.85]]),
    ("own", r"own(\d+)_C([\d.]+)$", [[20, 50, 200, 800, 4390], [0.001, 0.01, 0.1]]),
    ("sim", r"sim(\d+)_C([\d.]+)$", [[800, 4390], [0.001, 0.01]]),
    ("b23", r"b23([rn])(\d+)_C([\d.]+)$", [["r", "n"], [100, 400], [0.1]]),
    ("g", r"g(dwt|emg|hilbert|oth|stft)_k(\d+)$", [["dwt", "emg", "hilbert", "oth", "stft"], [50, 200]]),
    ("idxg", r"idxg(dwt|emg|hilbert|oth|stft)10$", [["dwt", "emg", "hilbert", "oth", "stft"]]),
    ("idx", r"idx(\d+)$", [[1, 3, 10, 30, 100]]),
    ("f", r"f(\d)$", [[0, 1, 2, 3, 4]]),
    ("ens", r"ens8$", []),
]


def parse(p):
    for fam, rx, _ in PAT:
        m = re.match(rx, p)
        if m:
            return fam, m.groups()
    raise ValueError(p)


def neighbors(p):
    """同族、恰一个超参移一步（数值）或换一个取值（类别）的部件。"""
    fam, hp = parse(p)
    vals = dict((f, v) for f, _, v in PAT)[fam]
    out = []
    for i, (h, vs) in enumerate(zip(hp, vals)):
        sv = [str(v) for v in vs]
        if h not in sv:
            continue
        k = sv.index(h)
        num = not isinstance(vs[0], str)
        steps = [k - 1, k + 1] if num else [x for x in range(len(vs)) if x != k]
        for s in steps:
            if 0 <= s < len(vs):
                nh = list(hp)
                nh[i] = sv[s]
                q = render(fam, nh)
                if q in base:
                    out.append(q)
    return out


def render(fam, hp):
    return {"k": "k{}_w{}", "own": "own{}_C{}", "sim": "sim{}_C{}", "b23": "b23{}{}_C{}",
            "g": "g{}_k{}", "idxg": "idxg{}10", "idx": "idx{}", "f": "f{}", "ens": "ens8"}[fam].format(*hp)


def parts_of(recipe, members):
    m = re.match(r"专属配方\((\d+)\)", recipe)
    if m:  # 专属：该人 ev2 单人最优 cfg
        j = subs.index(members[0])
        c = int(ev2b[j].max(1).argmax())
        recipe = cfg_name(c, base)
    body = recipe.split("|")[0].strip()
    body = body[2:-1] if body.startswith("z[") else body
    return [[p.strip() for p in term.split("+")] for term in body.split(";")], recipe


REC = [parts_of(g["recipe"], g["members"]) for g in G]
PS = [sorted({p for t in r for p in t}) for r, _ in REC]


def sim(a, b):
    A, B = PS[a], PS[b]
    sc = sum(1. if p in B else (.5 if any(q in B for q in neighbors(p)) else 0.) for p in A)
    sc2 = sum(1. if p in A else (.5 if any(q in A for q in neighbors(p)) else 0.) for p in B)
    return min(sc, sc2) / max(len(A), len(B))


ng = len(G)
pr("各组部件：")
for i in range(ng):
    pr(f"  G{i:<2} n={len(G[i]['members'])} {PS[i]}   ← {REC[i][1]}")
pairs = sorted(((sim(a, b), a, b) for a in range(ng) for b in range(a + 1, ng)), reverse=True)
pr("\n结构最接近的组对（不算指标）：")
for s, a, b in pairs[:15]:
    pr(f"  {s:.2f}  G{a}+G{b}  共有 {sorted(set(PS[a]) & set(PS[b]))}")


# ---------- 组内微调 ----------
def cand_cfgs(a, b):
    P = set(PS[a]) | set(PS[b])
    P |= {q for p in list(P) for q in neighbors(p)}
    P = sorted(P)
    cs = {NAME2C[p] for p in P if p in NAME2C}
    cs |= {NAME2C[f"{p}+{q}"] for p, q in itertools.combinations(P, 2) if f"{p}+{q}" in NAME2C}
    return P, sorted(cs)


def tune(U, cs):
    best = dict(v=-1.)
    # 单 cfg（直接查 ev2b，只取组成员）
    sub = ev2b[U][:, cs]  # (n, ncs, 85)
    mn = sub.min(0)
    k = np.unravel_index(np.argmax(np.where(mn >= TH, sub.mean(0), mn - 1)), mn.shape)
    best = dict(v=float(mn[k]), kind="单cfg", recipe=f"{cfg_name(cs[k[0]], base)} | {THR2[k[1]][0]}")
    ia, ib = map(list, zip(*itertools.combinations(cs, 2)))
    fn = lambda C, ct: zpair(C, ct, ia, ib)  # noqa: E731
    v, kk = _search(U, fn, .75)
    if v > best["v"]:
        p, t = divmod(kk, NT2)
        best = dict(v=v, kind="z双", recipe=f"z[{cfg_name(ia[p], base)} ; {cfg_name(ib[p], base)}] | {THR2[t][0]}",
                    per=_full(U, fn, p, t))
    if best["v"] < TH and len(cs) <= 70:
        ta, tb, tc = map(list, zip(*itertools.combinations(cs, 3)))
        for s in range(0, len(ta), 20000):  # 分块防显存溢出
            A, Bq, Cq = ta[s:s + 20000], tb[s:s + 20000], tc[s:s + 20000]
            fn3 = lambda C, ct: _zrows(C[A] + C[Bq] + C[Cq], ct)  # noqa: E731
            v, kk = _search(U, fn3, .75)
            if v > best["v"]:
                p, t = divmod(kk, NT2)
                best = dict(v=v, kind="z三", recipe="z[" + " ; ".join(cfg_name(c, base) for c in (A[p], Bq[p], Cq[p]))
                            + f"] | {THR2[t][0]}", per=_full(U, fn3, p, t))
    return best


res = []
done = 0
for s, a, b in pairs:
    if done >= NTOP:
        break
    if s <= 0:
        break
    U = GROUPS[a] + GROUPS[b]
    P, cs = cand_cfgs(a, b)
    pr(f"\n[{done + 1}/{NTOP}] G{a}+G{b} 结构相似 {s:.2f}，成员 {[subs[j] for j in U]}")
    pr(f"  候选部件 {len(P)}：{P}\n  候选 cfg {len(cs)}")
    r = tune(U, cs)
    pr(f"  → 组内最小 bACC {r['v']:.3f}  {r['kind']}  {r['recipe']}  {'可行' if r['v'] >= TH else '不可行'}"
       f"  [{time.time() - t0:.0f}s]")
    if "per" in r:
        for j, x in zip(U, r["per"]):
            pr(f"    {subs[j]}: bACC {x[0]:.3f} AUC {x[1]:.3f} F1 {x[2]:.3f}{'  <.9' if x[0] < TH else ''}")
    res.append(dict(a=a, b=b, sim=s, members=[subs[j] for j in U], parts=P, ncfg=len(cs),
                    v=r["v"], kind=r["kind"], recipe=r["recipe"],
                    per=np.asarray(r["per"]).tolist() if "per" in r else None))
    json.dump(res, open(HERE / "struct_tune.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    done += 1
pr("done")
