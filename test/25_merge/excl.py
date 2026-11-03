"""E25 专属配方：把 E22 中 cands 池未覆盖（池 oracle<0.9）的 7 人的最优配置按 round2/3/5/6/8 原定义重建，
并在全部 56 名 6 块被试上评估同一配方（034 缺块 0–2，own/b23/sim 等不可用 → NaN）。
只建配方用到的组件。输出 excl.json（每配方 × 每人 bACC/AUC/F1，及本人复现值与 best.json 对比）。
运行：.venv\\Scripts\\python.exe test/25_merge/excl.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "19_xblock"))
sys.path.insert(0, str(HERE.parent / "22_subject"))
from xblock import L, WS, dev, dval, z, fit_wlr  # noqa: E402
from xblock2 import KS, WOS  # noqa: E402
from round8 import groups, fine_groups, causal_q, NF  # noqa: E402

# 配方：(结构, 阈值)。结构：str=组件；("z", a, b, ...)=z(和)；("tail", T, a)=z(T + a/3)
P = lambda *a: ("z",) + a  # noqa: E731
RECIPES = {
    "022": (("tail", P(P("ghilbert_k200", "fghilbert_envelope_3_k20"),
                       P("fghilbert_frequency_3_k65", "fgstft_log_power_k125"),
                       P("fghilbert_envelope_3_k20", "idxoth3")), "f4"), ("q", 300, .55)),
    "042": (("tail", P(P("b23n10_C0.1", "wcal100_C0.1"), P("b23n50_C0.001", "fbstft_1.0_relative_k325"),
                       P("fbhilbert_frequency_0_k65", "fbstft_1.0_relative_k325")), "idxemg3"), ("q", 300, .525)),
    "052": (P(P("k4390_w0.6", "b23n100_C0.01"), P("k4390_w0.85", "sim4390_C0.01"), P("k400_w0.6", "k4390_w0.85")),
            ("q", 300, .35)),
    "030": (P(P("sim5w0.5_4390_C0.01", "idxstft30"), P("sim5w0.5_4390_C0.01", "idxstft10"),
              P("ens8", "sim5w0.5_4390_C0.001")), ("q", 300, .55)),
    "041": (P(P("b23r400_C0.1", "goth_k200"), P("sim200_C0.01", "f0"), "goth_k200"), ("m", 300, None)),
    "053": (P(P("k4390_w0.6", "goth_k200"), "k4390_w0.6", P("k1600_w0.6", "k4390_w0.6")), ("m", 300, None)),
    "054": (P("sc99999_k200", "jw0.95_k1600_C0.1"), ("m", 0, None)),
}


def leaves(t, acc):
    if isinstance(t, str):
        acc.add(t)
    else:
        for a in t[1:]:
            leaves(a, acc)
    return acc


NEED = set().union(*(leaves(r[0], set()) for r in RECIPES.values()))


def build(q, i, subs, D, F, dall, SC, G, FG, Xall, yall, sid):
    y, blk = D[q][1], D[q][2]
    cal = blk <= 3
    cal_t = torch.tensor(cal, device=dev)
    X = F[q]
    p1, p2 = blk <= 1, (blk >= 2) & (blk <= 3)
    d1, d2 = dval(X[p1], y[p1]), dval(X[p2], y[p2])
    cons = torch.where(torch.sign(d1) == torch.sign(d2), torch.minimum(d1.abs(), d2.abs()), torch.zeros_like(d1))
    order = torch.argsort(cons, descending=True)
    o2 = torch.argsort(d2.abs(), descending=True)
    jk = [f"k{k}_w{wo}" for wo in WOS for k in KS]
    S = {k: torch.tensor(SC[f"{q}|{k}"], device=dev) for k in jk}
    S["ens8"] = z(torch.stack([S[k] for k in jk]).mean(0), cal_t)
    for n in NEED:
        if n.startswith("b23"):
            t, rest = n[3], n[4:]
            k, C = int(rest.split("_C")[0]), float(rest.split("_C")[1])
            cc = (order if t == "r" else o2)[:k]
            S[n] = z(L.fit_lr(X[p2][:, cc], y[p2], C)(X[:, cc]), cal_t)
        elif n.startswith("wcal"):
            k, C = int(n[4:].split("_C")[0]), float(n.split("_C")[1])
            sw2 = np.where(p2[cal], 2.0, 1.0)
            sw2 = sw2 / sw2.sum()
            S[n] = z(fit_wlr(X[cal][:, o2[:k]], y[cal], sw2, C)(X[:, o2[:k]]), cal_t)
        elif n.startswith(("fg", "fb")):
            g, k = n[2:].rsplit("_k", 1)
            cols = FG[g]
            if n.startswith("fg"):
                c = cols[torch.argsort(cons[cols], descending=True)][:int(k)]
                S[n] = z(L.fit_lr(X[cal][:, c], y[cal], 1e-2)(X[:, c]), cal_t)
            else:
                c = cols[torch.argsort(d2[cols].abs(), descending=True)][:int(k)]
                S[n] = z(L.fit_lr(X[p2][:, c], y[p2], 1e-2)(X[:, c]), cal_t)
        elif n.startswith("g") and "_k" in n:
            g, k = n[1:].split("_k")
            cols = G[g]
            c = cols[torch.argsort(cons[cols], descending=True)][:int(k)]
            S[n] = z(L.fit_lr(X[cal][:, c], y[cal], 1e-2)(X[:, c]), cal_t)
        elif n.startswith("sc"):  # round2：单一 running 尺度 w 内一致性前 k 维，本人校准 LR
            w, k = n[2:].split("_k")
            j = WS.index(int(w))
            oj = torch.argsort(cons[j * NF:(j + 1) * NF], descending=True) + j * NF
            c = oj[:int(k)]
            S[n] = z(L.fit_lr(X[cal][:, c], y[cal], 1e-2)(X[:, c]), cal_t)
        elif n.startswith("jw"):  # round2：其他全部 6 块被试 + 本人校准联合 LR
            wo = float(n[2:].split("_k")[0])
            k = int(n.split("_k")[1].split("_C")[0])
            C = float(n.split("_C")[1])
            oth = sid != i
            Xo, yo = Xall[torch.tensor(oth, device=dev)], yall[oth]
            sw = np.concatenate([np.full(len(yo), (1 - wo) / len(yo)), np.full(cal.sum(), wo / cal.sum())])
            c = order[:k]
            S[n] = z(fit_wlr(torch.cat([Xo[:, c], X[cal][:, c]]), np.concatenate([yo, y[cal]]), sw, C)(X[:, c]),
                     cal_t)
            del Xo
    dq = dval(X[cal], y[cal])
    cos = torch.nn.functional.cosine_similarity(dall, dq[None], 1)
    cos[i] = -9
    rk = torch.argsort(cos, descending=True).tolist()
    for n in [n for n in NEED if n.startswith("sim")]:
        if "w" in n.split("_")[0]:
            nn, rest = n[3:].split("w", 1)
            nn, wo = int(nn), float(rest.split("_")[0])
            k, C = int(n.split("_")[1]), float(n.split("_C")[1])
        else:
            nn, wo, k, C = 10, .5, int(n[3:].split("_C")[0]), float(n.split("_C")[1])
        sim = [subs[j] for j in rk[:nn]]
        Xo = torch.cat([F[o] for o in sim])
        yo = np.concatenate([D[o][1] for o in sim])
        sw = np.concatenate([np.full(len(yo), (1 - wo) / len(yo)), np.full(cal.sum(), wo / cal.sum())])
        c = order[:k]
        S[n] = z(fit_wlr(torch.cat([Xo[:, c], X[cal][:, c]]), np.concatenate([yo, y[cal]]), sw, C)(X[:, c]), cal_t)
        del Xo
    sg = torch.sign(d1 + d2)
    Zc = (X - X[cal].mean(0)) / (X[cal].std(0) + 1e-6) * sg
    for n in NEED:
        if n.startswith("f") and n[1:].isdigit():
            S[n] = z(Zc[:, order[int(n[1:])]], cal_t)
        elif n.startswith("idx"):
            g = n[3:].rstrip("0123456789")
            m = int(n[3 + len(g):])
            cols = G[g]
            og = cols[torch.argsort(cons[cols], descending=True)]
            S[n] = z(Zc[:, og[:m]].mean(1), cal_t)
    return S, cal_t


def ev_tree(t, S, cal_t):
    if isinstance(t, str):
        return S[t]
    if t[0] == "tail":
        return z(ev_tree(t[1], S, cal_t) + ev_tree(t[2], S, cal_t) / 3, cal_t)
    return z(sum(ev_tree(a, S, cal_t) for a in t[1:]), cal_t)


def main():
    best = json.load(open(HERE.parent / "22_subject" / "best.json"))
    G = {k: torch.tensor(v, device=dev) for k, v in groups().items()}
    FG = {k: torch.tensor(v, device=dev) for k, v in fine_groups().items()}
    subs, D = L.load()
    subs = [q for q in subs if len(np.unique(D[q][2])) == 6]
    F = None
    for w in WS:
        Fw = L.feats(D, subs, w)
        F = Fw if F is None else {q: torch.cat([F[q], Fw[q]], 1) for q in subs}
    dall = torch.stack([dval(F[q], D[q][1]) for q in subs])
    SC = np.load(HERE.parent / "19_xblock" / "xblock2_scores.npz")
    Xall = torch.cat([F[q] for q in subs])
    yall = np.concatenate([D[q][1] for q in subs])
    sid = np.concatenate([[i] * len(D[q][1]) for i, q in enumerate(subs)])
    out = {r: {} for r in RECIPES}
    for i, q in enumerate(subs):
        S, cal_t = build(q, i, subs, D, F, dall, SC, G, FG, Xall, yall, sid)
        y, blk = D[q][1], D[q][2]
        tst = blk >= 4
        for r, (tree, (kind, w, qq)) in RECIPES.items():
            s = ev_tree(tree, S, cal_t)
            if kind == "q":
                th = causal_q(s, w, qq)[tst]
            else:
                th = L.causal_median(s, w)[tst] if w else torch.tensor(0.0, device=dev)
            out[r][q] = L.subj_metrics(y[tst], s[tst], th)
        print(q, " ".join(f"{r}:{out[r][q]['bacc']:.3f}" for r in RECIPES), flush=True)
        del S
        torch.cuda.empty_cache()
    rep = {r: dict(owner_repro=out[r][r]["bacc"], owner_best_json=best[r]["bacc"],
                   cfg=best[r]["cfg"]) for r in RECIPES}
    print(json.dumps(rep, indent=1))
    json.dump(dict(per=out, repro=rep), open(HERE / "excl.json", "w"), indent=1)


if __name__ == "__main__":
    main()
