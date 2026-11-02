"""E16：逐人攻关诊断。列出某被试前 N 个配置及其逐块准确率、块首 30 s 误差占比。
运行：.venv\\Scripts\\python.exe test/16_subject_push/diag.py 040
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.argv.append("--sf2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "15_nested_select"))
import nested_select as ns  # noqa: E402
ns.EXTRAS.append(Path(__file__).resolve().parent / "scores_ps")
ns.EXTRAS.append(Path(__file__).resolve().parent / "scores_ps2")
ns.EXTRAS.append(Path(__file__).resolve().parent / "scores_ps3")
ns.EXTRAS.append(Path(__file__).resolve().parent / "scores_ps4")
ns.EXTRAS.append(Path(__file__).resolve().parent / "scores_ps5")
ns.EXTRAS.append(Path(__file__).resolve().parent / "scores_ps6")

q = sys.argv[1]
y, blk, S = ns.load(q)
def _single(c):
    return max(ns.frame_bacc(y, (ns.ema(S[c], h) >= 0).astype(int)) for h in ns.HS)


cands = sorted(sorted(S, key=lambda c: -_single(c))[:30])
keys, V = ns.build(S, cands)
P = (V >= 0).astype(np.int8)
ub = np.unique(blk)
acc, cls = ns.block_acc(P, y, blk, ub)
full = ns.bacc_from(acc, cls, np.ones(len(ub), bool))
top = np.argsort(-full)[:8]
print("blocks", [(int(b), int(c), int((blk == b).sum())) for b, c in zip(ub, cls)])
for k in top:
    pos = np.concatenate([np.arange((blk == b).sum()) for b in ub])  # 块内序号（仅诊断用）
    err = P[k] != y
    print(f"{full[k]:.3f} {keys[k]} blk_acc={np.round(acc[k], 3).tolist()} "
          f"err_first30={err[pos < 30].sum()}/{err.sum()}")

# 记录该人当前最优（逐人最优口径），并给出最优配置的 AUC/F1
import json  # noqa: E402
from sklearn.metrics import f1_score, roc_auc_score  # noqa: E402
k = top[0]
rec_path = Path(__file__).resolve().parent / "results.json"
rec = json.loads(rec_path.read_text(encoding="utf-8")) if rec_path.exists() else {}
if full[k] <= rec.get(q, {}).get("bacc", 0):
    print("kept", q, rec[q]["bacc"]); sys.exit()
rec[q] = dict(bacc=float(full[k]), auc=float(roc_auc_score(y, V[k])),
              f1=float(f1_score(y, P[k])), cfg=[str(x) for x in keys[k]],
              blk_acc=np.round(acc[k], 3).tolist())
rec_path.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
print("saved", q, rec[q]["bacc"], "auc", round(rec[q]["auc"], 3), "f1", round(rec[q]["f1"], 3))
