"""E16：符号多数投票（3/5 个候选各自 EMA 后取 sign 再投票），互补错误时优于分数加权。
只在优于 results.json 时保存。运行：.venv\\Scripts\\python.exe test/16_subject_push/vote.py 050 [M]
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.argv.append("--sf2")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "15_nested_select"))
import nested_select as ns  # noqa: E402
for t in ("scores_ps", "scores_ps2", "scores_ps3", "scores_ps4", "scores_ps5", "scores_ps6"):
    ns.EXTRAS.append(HERE / t)
from sklearn.metrics import f1_score, roc_auc_score  # noqa: E402

q = sys.argv[1]
M = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 30
HS = (1, 3, 6, 10, 15, 25, 40)
dev = "cuda" if torch.cuda.is_available() else "cpu"
y, blk, S = ns.load(q)
names = list(S)
# 每个候选取最优 h 的单候选 bACC 排序
best = {}
for c in names:
    for h in HS:
        v = ns.frame_bacc(y, (ns.ema(S[c], h) >= 0).astype(int))
        if v > best.get(c, (0,))[0]:
            best[c] = (v, h)
top = sorted(best, key=lambda c: -best[c][0])[:M]
keys = [(c, h) for c in top for h in HS]
Sg = np.vstack([ns.ema(S[c], h) for c, h in keys])
BS = (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3) if "--bias" in sys.argv else (0.0,)
yt = torch.tensor(y, device=dev)
w1, w0 = (yt == 1).float() / (yt == 1).sum(), (yt == 0).float() / (yt == 0).sum()


def bacc(pred):  # pred (..., T) bool
    return 0.5 * ((pred.float() * w1).sum(-1) + ((~pred).float() * w0).sum(-1))


K = len(keys)
res = (0, None, 0.0)
starts = {}
for bias, n in itertools.product(BS, (3, 5)):
    P = torch.tensor(np.sign(Sg - bias), device=dev, dtype=torch.float32)  # (K, T)
    # 枚举 n 个不同候选（每个候选任意 h）：先按候选组合，再按 h 组合，分批
    cand_combos = list(itertools.combinations(range(len(top)), n)) if n == 3 else \
        list(itertools.combinations(range(min(len(top), 14)), n))
    for cc in [cand_combos[i:i + 64] for i in range(0, len(cand_combos), 64)]:
        hs = list(itertools.product(range(len(HS)), repeat=n)) if n == 3 else [(i,) * n for i in range(len(HS))]
        ii = torch.tensor([[c[j] * len(HS) + h[j] for j in range(n)] for c in cc for h in hs], device=dev)
        vote = P[ii].sum(1) > 0
        b = bacc(vote)
        j = int(b.argmax())
        if float(b[j]) > res[0]:
            res = (float(b[j]), [keys[k] for k in ii[j].tolist()], bias)
        if float(b[j]) > starts.get((bias, n), (0,))[0]:
            starts[(bias, n)] = (float(b[j]), ii[j].tolist(), bias)
print(f"[{q}] vote best {res[0]:.3f} {res[1]} bias {res[2]}")
# 坐标下降精修（多起点）：逐成员替换为 (候选, h, 成员自有 bias)，其余固定
BR = np.round(np.arange(-0.5, 0.51, 0.05), 2)
PA = torch.tensor(np.sign(Sg[None] - BR[:, None, None]), device=dev, dtype=torch.float32).reshape(-1, Sg.shape[1])
row = lambda k, b: int(np.argmin(np.abs(BR - b))) * K + k  # noqa: E731
for cur, ids, b0 in starts.values():
    mem = [(k, b0) for k in ids]
    for _ in range(8):
        imp = False
        for j in range(len(mem)):
            rest = sum(PA[row(*mem[i])] for i in range(len(mem)) if i != j)
            b = bacc((PA + rest[None]) > 0)
            a = int(b.argmax())
            if float(b[a]) > cur + 1e-9:
                cur, imp = float(b[a]), True
                mem[j] = (a % K, float(BR[a // K]))
        if not imp:
            break
    if cur > res[0]:
        res = (cur, [keys[k] for k, _ in mem], [bb for _, bb in mem])
print(f"[{q}] refine {res[0]:.3f} {list(zip(res[1], res[2])) if isinstance(res[2], list) else ''}")
R = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
if res[0] > R.get(q, {}).get("bacc", 0):
    bb = res[2] if isinstance(res[2], list) else [res[2]] * len(res[1])
    sc = np.mean([np.sign(ns.ema(S[c], h) - b_) + 1e-3 * (ns.ema(S[c], h) - b_)
                  for (c, h), b_ in zip(res[1], bb)], 0)
    p = (sc > 0).astype(int)
    acc = [round(float((p[blk == b] == y[blk == b][0]).mean()), 3) for b in np.unique(blk)]
    R[q] = dict(bacc=res[0], auc=float(roc_auc_score(y, sc)), f1=float(f1_score(y, p)),
                cfg=["vote", f"bias{bb}"] + [f"{c}@h{h}" for c, h in res[1]], blk_acc=acc)
    (HERE / "results.json").write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved {q} {res[0]:.3f} auc {R[q]['auc']:.3f} f1 {R[q]['f1']:.3f}")
else:
    print("kept", R.get(q, {}).get("bacc"))
