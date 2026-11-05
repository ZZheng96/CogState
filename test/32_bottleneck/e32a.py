"""E32a：剔除瓶颈人后，其余被试能否用【单一配置】整体达标（oracle，复用 E31b 缓存 e31b_ev.npz）。
列 = 1152 配置 × 85 阈值 + 18 原组配方。
1) 每列 n≥.9 计数 → 单配置最大覆盖及未覆盖者；
2) 剔除指定瓶颈集后，余下人的 max-min；
3) 贪心：每轮取余下人 max-min 最优列，剔除该列下最差者，直到可行 → 需剔除多少人。
运行：.venv\\Scripts\\python.exe test/32_bottleneck/e32a.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
T = HERE.parent
for p in ("17_compliant", "18_calib", "19_xblock", "23_synthesis"):
    sys.path.insert(0, str(T / p))
import loso as L  # noqa: E402

spec = importlib.util.spec_from_file_location("e26lib", T / "26_merge2" / "e26lib.py")
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)

WGRID, ALPHAS = (300, 600, 1200), (0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7)
TN = ("inf", "0.2", "0.5")
COMPS = ["goth_k200", "gemg_k200", "ghilbert_k200", "gstft_k200", "idx10", "idxgdwt10", "own200_C0.1",
         "b23n100_C0.1", "b23n400_C0.1", "k4390_w0.6", "k1600_w0.85", "sim800_C0.01", "sim4390_C0.01",
         "sc99999_k200", "jw0.95_k1600_C0.1"]
PN = [f"b|pool|w{w}|a{a:g}|t{t}" for w in WGRID for t in TN for a in ALPHAS]
NAMES = PN + [f"z[{p};{c}]" for p in PN for c in COMPS]
NC, NT = len(NAMES), len(E.THR2)
r2 = json.load(open(T / "26_merge2" / "round2.json", encoding="utf-8"))["groups"]


def cname(k):
    if k >= NC * NT:
        return f"原组配方G{k - NC * NT + 1}:{r2[k - NC * NT]['recipe']}"
    c, t = divmod(k, NT)
    return f"{NAMES[c]}@{E.THR2[t][0]}"


def main():
    subs, _ = L.load()
    z = np.load(T / "31_integrate" / "e31b_ev.npz")
    B = np.concatenate([z["EV"].reshape(len(subs), NC * NT, 3), z["ORIG"]], 1)
    Bt = torch.tensor(np.nan_to_num(B[..., 0], nan=-1.0), device="cuda")
    out = {}

    def stat(idx, k):
        v = B[idx, k]
        return dict(cfg=cname(k), min=float(v[:, 0].min()), bacc=float(v[:, 0].mean()),
                    auc=float(np.nanmean(v[:, 1])), f1=float(v[:, 2].mean()), n=len(idx),
                    n90=int((v[:, 0] >= .9).sum()))

    # 1) 单配置最大覆盖
    cnt = (Bt >= .9).sum(0)
    k = int(torch.where(cnt == cnt.max(), Bt.mean(0), torch.tensor(-9., device="cuda")).argmax())
    miss = [subs[i] for i in range(len(subs)) if B[i, k, 0] < .9]
    out["max_cover"] = dict(stat(list(range(len(subs))), k), missing=miss)
    print("单配置最大覆盖", out["max_cover"])
    covered = [i for i in range(len(subs)) if B[i, k, 0] >= .9]
    out["max_cover_rest"] = stat(covered, k)
    print("  覆盖者内", out["max_cover_rest"])

    # 2) 剔除指定瓶颈集
    for name, bad in (("b4", ["047", "001", "021", "043"]), ("b6", ["047", "001", "021", "043", "053", "054"])):
        idx = [i for i, q in enumerate(subs) if q not in bad]
        mn = Bt[idx].min(0).values
        k = int(torch.where(mn >= mn.max() - 1e-7, Bt[idx].mean(0), torch.tensor(-9., device="cuda")).argmax())
        out[name] = dict(stat(idx, k), excluded=bad,
                         fails=sorted([(subs[i], round(float(B[i, k, 0]), 4)) for i in idx if B[i, k, 0] < .9],
                                      key=lambda x: x[1]))
        print(name, out[name])

    # 3) 贪心剔除
    idx, removed = list(range(len(subs))), []
    while True:
        mn = Bt[idx].min(0).values
        k = int(torch.where(mn >= mn.max() - 1e-7, Bt[idx].mean(0), torch.tensor(-9., device="cuda")).argmax())
        if mn[k] >= .9:
            break
        w = idx[int(Bt[idx, k].argmin())]
        removed.append((subs[w], round(float(B[w, k, 0]), 4), round(float(mn[k]), 4)))
        idx.remove(w)
    out["greedy"] = dict(stat(idx, k), removed=removed)
    print("贪心", len(removed), out["greedy"])
    json.dump(out, open(HERE / "e32a.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
