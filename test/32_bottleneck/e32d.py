"""E32d：嵌套验证门控可行性。
对 G2∪G3 14 人，留出一对 relax+task 块（块 0/2 为 relax、1/3 为 task，单块只有一类，故成对留出：
(2,3) 为主、另 (0,1)(0,3)(1,2) 轮换），其余 2 块训练，重训 RA/RB 的 7 个组件，
在留出块上算 RA@m300 / RB@m1200 的样本外 AUC、bACC，按判据为每人选配方，检查是否 G2→RA、G3→RB。
组件重训严格复刻 test/23_synthesis/common.py（cands_regular）与 test/19_xblock/xblock2.py：
  idxgdwt10 / ghilbert_k200 / b23r400_C0.1：只用本人训练块；
  sim*：相似 10 人（全量数据）+ 本人训练块；相似度用本人训练块效应量；
  k4390_w0.6：其他 55 人（全量）+ 本人训练块，本人权重 .6。
一致性排序：原为块 0–1 vs 2–3；此处训练仅 2 块，改为各块前半 vs 后半。b23 = 训练的那一对块。z 参考 = 训练块。
复刻已验证：块 0–3 训练时 7 组件与缓存相关 = 1.0000（首轮运行，见报告）。测试段结果用原缓存（块 0–3 训练）。
运行：.venv\\Scripts\\python.exe test/32_bottleneck/e32d.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
T = HERE.parent
for p in ("19_xblock", "22_subject", "23_synthesis", "24_cluster", "26_merge2"):
    sys.path.insert(0, str(T / p))
from xblock import L, WS, dev, dval, fit_wlr  # noqa: E402
from round6 import groups  # noqa: E402
from clib import load  # noqa: E402
import e26lib as E  # noqa: E402

G2 = "016 013 006 058 014 037 036 008".split()
G3 = "018 048 056 021 038 059".split()
MEM = G2 + G3
COMP = ["idxgdwt10", "sim4390_C0.001", "k4390_w0.6", "ghilbert_k200", "sim4390_C0.01", "b23r400_C0.1", "sim800_C0.001"]
THN = [t[0] for t in E.THR2]
IA, IB = THN.index("m300"), THN.index("m1200")
FOLDS = ["23", "01", "03", "12"]  # 留出的 relax+task 块对（每块单一类别，必须成对留出）
LOG = open(HERE / "e32d.log", "w", encoding="utf-8")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def zm(s, m):
    return (s - s[m].mean()) / (s[m].std() + 1e-6)


def recipes(C, m):
    RA = zm(zm(C[0] + C[1], m) + zm(C[2] + C[3], m), m)
    RB = zm(zm(C[4] + C[5], m) + C[6], m)
    return torch.stack([RA, RB])


def rauc(s, y):
    r = s.argsort().argsort().float() + 1
    npos, nneg = y.sum().float(), (~y).sum().float()
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def comps(q, i, subs6, D, F, dall, G, trn_b):
    """trn_b: 训练块集合 → 7 组件分数 (7,T)。"""
    y, blk = D[q][1], D[q][2]
    trn = np.isin(blk, trn_b)
    trn_t = torch.tensor(trn, device=dev)
    X = F[q]
    if sorted(trn_b) == [0, 1, 2, 3]:
        p1, p2 = blk <= 1, (blk >= 2) & (blk <= 3)
    else:  # 训练只剩 1 relax + 1 task 块：每块按时间前/后半拆成两半，保留"跨时间一致性"
        p1 = np.zeros(len(blk), bool)
        for b in trn_b:
            ix = np.where(blk == b)[0]
            p1[ix[:len(ix) // 2]] = True
        p2 = trn & ~p1
    d1, d2 = dval(X[p1], y[p1]), dval(X[p2], y[p2])
    cons = torch.where(torch.sign(d1) == torch.sign(d2), torch.minimum(d1.abs(), d2.abs()), torch.zeros_like(d1))
    order = torch.argsort(cons, descending=True)
    S = {}
    sg = torch.sign(d1 + d2)
    Zc = (X - X[trn].mean(0)) / (X[trn].std(0) + 1e-6) * sg
    og = G["dwt"][torch.argsort(cons[G["dwt"]], descending=True)]
    S["idxgdwt10"] = zm(Zc[:, og[:10]].mean(1), trn_t)
    del Zc
    oh = G["hilbert"][torch.argsort(cons[G["hilbert"]], descending=True)][:200]
    S["ghilbert_k200"] = zm(L.fit_lr(X[trn][:, oh], y[trn], 1e-2)(X[:, oh]), trn_t)
    pb = trn & (blk >= 2) & (blk <= 3) if sorted(trn_b) == [0, 1, 2, 3] else trn  # b23 = 最近一对训练块
    c = order[:400]
    S["b23r400_C0.1"] = zm(L.fit_lr(X[pb][:, c], y[pb], .1)(X[:, c]), trn_t)
    # sim：相似度用本人训练块
    dq = dval(X[trn], y[trn])
    cos = torch.nn.functional.cosine_similarity(dall, dq[None], 1)
    cos[i] = -9
    sim = [subs6[j] for j in torch.argsort(cos, descending=True)[:10].tolist()]
    Xo = torch.cat([F[o] for o in sim])
    yo = np.concatenate([D[o][1] for o in sim])
    yj = np.concatenate([yo, y[trn]])
    sw = np.concatenate([np.full(len(yo), .5 / len(yo)), np.full(trn.sum(), .5 / trn.sum())])
    for k, Cc in ((4390, 1e-3), (4390, 1e-2), (800, 1e-3)):
        c = order[:k]
        S[f"sim{k}_C{Cc}"] = zm(fit_wlr(torch.cat([Xo[:, c], X[trn][:, c]]), yj, sw, Cc)(X[:, c]), trn_t)
    del Xo
    # joint：其他全部人 + 本人训练块，wo=.6
    oth = [o for o in subs6 if o != q]
    Xo = torch.cat([F[o] for o in oth] + [X[trn]])
    yo = np.concatenate([D[o][1] for o in oth])
    yj = np.concatenate([yo, y[trn]])
    sw = np.concatenate([np.full(len(yo), .4 / len(yo)), np.full(trn.sum(), .6 / trn.sum())])
    c = order[:4390]
    S["k4390_w0.6"] = zm(fit_wlr(Xo[:, c], yj, sw, 1e-2)(X[:, c]), trn_t)
    del Xo
    torch.cuda.empty_cache()
    return torch.stack([S[n] for n in COMP])


def main():
    R = load()
    base = R["meta"]["base"]
    ci = [base.index(c) for c in COMP]
    subs, D = L.load()
    subs6 = [q for q in subs if len(np.unique(D[q][2])) == 6]
    F = None
    for w in WS:
        Fw = L.feats(D, subs6, w)
        F = Fw if F is None else {q: torch.cat([F[q], Fw[q]], 1) for q in subs6}
    G = {k: torch.tensor(v, device=dev) for k, v in groups().items()}
    dall = torch.stack([dval(F[q], D[q][1]) for q in subs6])
    out = {"per": {}}
    for q in MEM:
        y, blk = D[q][1], D[q][2]
        assert np.array_equal(y, R["D"][q][1]) and np.array_equal(blk, R["D"][q][2])
        i = subs6.index(q)
        yt = torch.tensor(y, device=dev).bool()
        # 测试段（原缓存，块 0–3 训练）
        cal = E.calref(q, R["D"])
        C0 = torch.tensor(R["SC"][q], device=dev, dtype=torch.float32)[ci]
        M0 = recipes(C0, cal)
        tst = torch.tensor(np.where(blk >= 4)[0], device=dev)
        mt = E.metrics(M0, tst, yt[tst])  # (2,85,3)
        rec = dict(test=dict(RA=[float(x) for x in mt[0, IA]], RB=[float(x) for x in mt[1, IB]]), oos={})
        for f in FOLDS:
            hb = [int(c) for c in f]
            trn_b = [b for b in range(4) if b not in hb]
            Cf = comps(q, i, subs6, D, F, dall, G, trn_b)
            trn = torch.tensor(np.isin(blk, trn_b), device=dev)
            M = recipes(Cf, trn)
            ho = torch.tensor(np.where(np.isin(blk, hb))[0], device=dev)
            yh = yt[ho]
            mh = E.metrics(M, ho, yh)
            rec["oos"][f] = dict(
                RA=dict(rauc=rauc(M[0, ho], yh), bacc=float(mh[0, IA, 0]), auc=float(mh[0, IA, 1]), f1=float(mh[0, IA, 2])),
                RB=dict(rauc=rauc(M[1, ho], yh), bacc=float(mh[1, IB, 0]), auc=float(mh[1, IB, 1]), f1=float(mh[1, IB, 2])))
            o = rec["oos"][f]
            log(q, f"blk{f}", f"RA rauc {o['RA']['rauc']:.3f} bACC {o['RA']['bacc']:.3f} | "
                f"RB rauc {o['RB']['rauc']:.3f} bACC {o['RB']['bacc']:.3f}")
        out["per"][q] = rec
        json.dump(out, open(HERE / "e32d.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # 判据（>0 → RA）
    def crit(rec, name):
        o = rec["oos"]
        dA = {f: o[f]["RA"]["rauc"] - o[f]["RB"]["rauc"] for f in o}
        dB = {f: o[f]["RA"]["bacc"] - o[f]["RB"]["bacc"] for f in o}
        return dict(dAUC_23=dA["23"], dbACC_23=dB["23"], sum_23=dA["23"] + dB["23"],
                    dAUC_mean=np.mean(list(dA.values())), dbACC_mean=np.mean(list(dB.values())),
                    sum_mean=np.mean(list(dA.values())) + np.mean(list(dB.values())),
                    vote4=sum(np.sign(dA[f] + dB[f]) for f in o))[name]

    names = ["dAUC_23", "dbACC_23", "sum_23", "dAUC_mean", "dbACC_mean", "sum_mean", "vote4"]
    gate = json.load(open(HERE / "e32c.json", encoding="utf-8"))["gate"]
    out["crit"] = {}
    for nm in names:
        v = {q: float(crit(out["per"][q], nm)) for q in MEM}
        pick = {q: "RA" if v[q] > 0 else "RB" for q in MEM}
        ok = {q: pick[q] == ("RA" if q in G2 else "RB") for q in MEM}
        te = [out["per"][q]["test"][pick[q]] for q in MEM]
        a2, a3 = np.array([v[q] for q in G2]), np.array([v[q] for q in G3])
        res = dict(val={q: round(v[q], 4) for q in MEM}, pick=pick, n_ok=int(sum(ok.values())),
                   margin=float(a2.min() - a3.max()),
                   min_bacc=float(min(t[0] for t in te)), bacc=float(np.mean([t[0] for t in te])),
                   auc=float(np.mean([t[1] for t in te])), f1=float(np.mean([t[2] for t in te])))
        out["crit"][nm] = res
        log(f"{nm:11s} 正确 {res['n_ok']}/14 margin {res['margin']:+.4f} 合并 min {res['min_bacc']:.4f} "
            f"bACC {res['bacc']:.4f} AUC {res['auc']:.4f} F1 {res['f1']:.4f}")
    te_or = [out["per"][q]["test"]["RA" if q in G2 else "RB"] for q in MEM]
    out["oracle"] = dict(min_bacc=min(t[0] for t in te_or), bacc=float(np.mean([t[0] for t in te_or])),
                         auc=float(np.mean([t[1] for t in te_or])), f1=float(np.mean([t[2] for t in te_or])))
    out["gate_check"] = {q: (round(out["per"][q]["test"]["RA"][0] - gate["RA_m300"][q], 5),
                             round(out["per"][q]["test"]["RB"][0] - gate["RB_m1200"][q], 5)) for q in MEM}
    log("oracle", out["oracle"])
    log("gate_check(应≈0)", out["gate_check"])
    json.dump(out, open(HERE / "e32d.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
