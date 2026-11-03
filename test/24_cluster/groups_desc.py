"""E24 组描述：对给定分组（prof_groups.json 中的名字）列出成员、画像轴均值、组内胜出配置（全组 in-sample）
及组内 LOSO top1 的平均 bACC。用法：python groups_desc.py cal_ward_k8 [更多名字]  → groups_desc.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clib import HERE, AX5, load  # noqa: E402

R = load()
nq, nt = R["nq"], R["nt"]
TB = R["ev"][:, 0].reshape(nq, -1)
CANDS, THRN, keys = R["meta"]["cands"], R["meta"]["thr"], R["keys"]
G = json.load(open(HERE / "prof_groups.json"))
per = {}
import csv  # noqa: E402
for r in csv.DictReader(open(HERE / "prof_cluster_per_subject.csv", encoding="utf-8")):
    per[r["subject"]] = r
ie8 = CANDS.index("ens8") * nt
out = {}
for nm in sys.argv[1:]:
    lab = np.array(G[nm])
    d = []
    for g in np.unique(lab):
        mem = np.where(lab == g)[0]
        mu = np.nanmean(TB[mem], 0)
        c = int(np.nanargmax(mu))
        ax = {k: round(float(np.nanmean(R["P"][mem, keys.index(k)])), 3) for k in AX5}
        loso = float(np.mean([float(per[R["subs"][i]][f"{nm}_top1:bacc"]) for i in mem]))
        d.append(dict(group=int(g), n=len(mem), members=[R["subs"][i] for i in mem], axes=ax,
                      win=f"{CANDS[c // nt]}|{THRN[c % nt]}", win_insample_bacc=round(float(mu[c]), 3),
                      ens8_t0=round(float(np.nanmean(TB[mem, ie8])), 3), loso_top1=round(loso, 3),
                      oracle=round(float(np.nanmean(np.nanmax(TB[mem], 1))), 3)))
        print(nm, d[-1], flush=True)
    out[nm] = d
json.dump(out, open(HERE / "groups_desc.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
