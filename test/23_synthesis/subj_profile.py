"""B. 被试画像（文件名避开标准库 profile 模块）（只用校准块 0–3；034 只有块 3 残段可用，标 special）+ 画像与结果的 Spearman 相关。
输出 profile.csv、profile_corr.csv。需先运行 common.py（cands_eval.npz 用于逐族优势）。
"""
from __future__ import annotations

import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import mne
import numpy as np
import torch
from scipy.signal import welch
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import (L, dev, dval, load_all, cons_order, groups, family, S034, THR)  # noqa: E402

ROOT = HERE.parents[1]
if not hasattr(np, "in1d"):  # .venv 的 mne 1.4 在 numpy 2.x 下读 BDF 需要该别名
    np.in1d = np.isin
RAWF = {"rel_delta": 48, "rel_theta": 53, "rel_alpha": 58, "rel_beta": 63, "rel_gamma": 68,
        "hf_ratio_feat": 80, "rms_feat": 75, "flat_frac": 76, "spike_frac": 78, "nan_frac": 81}


def auc(y, s):
    y = np.asarray(y).astype(bool)
    r = np.argsort(np.argsort(s)) + 1
    n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def lr_top(Xtr, ytr, Xte, k=800, C=1e-2):
    c = torch.argsort(dval(Xtr, ytr).abs(), descending=True)[:k]
    return L.fit_lr(Xtr[:, c], ytr, C)(Xte[:, c]).cpu().numpy()


def raw_profile(q):
    d = ROOT / "EEG_dataset" / q
    bdf = next(d.glob("*.bdf"))
    t0 = datetime.strptime(re.search(r"_(\d{8}_\d{6}\.\d{3})_", bdf.name).group(1), "%Y%m%d_%H%M%S.%f")
    raw = mne.io.read_raw_bdf(str(bdf), preload=True, verbose="ERROR")
    x = raw.get_data()[0] * 1e6
    ev = list(csv.DictReader(open(d / f"{q}_events.csv", encoding="utf-8-sig")))
    st = [e for e in ev if e["event"] in ("trial_start", "trial_end")]
    segs, cur = [], None
    for e in st:
        t = (datetime.fromisoformat(e["local_time"][:26]) - t0).total_seconds()
        if e["event"] == "trial_start":
            cur = t
        elif cur is not None:
            segs.append((cur, t))
            cur = None
    ep, nz = [], 0
    for a, b in segs[:4]:
        s = x[max(0, int(a * 250)):int(b * 250)]
        for i in range(0, len(s) - 500, 500):
            e = s[i:i + 500]
            nz += 1
            if np.any(e == 0) or e.std() < 0.5:  # BAD_ACQ_SKIP 补零（可能带直流偏置）
                continue
            ep.append(e - e.mean())
    E = np.array(ep)
    f, P = welch(E, 250, nperseg=500, axis=1)
    P = P.mean(0)
    band = lambda lo, hi: P[(f >= lo) & (f < hi)].sum()
    ptp = np.ptp(E, 1)
    return dict(line50=band(48, 52) / band(1, 100), emg_hf=band(55, 100) / band(1, 45),
                alpha_raw=band(8, 13) / band(1, 45), theta_raw=band(4, 8) / band(1, 45),
                beta_raw=band(13, 30) / band(1, 45), amp_std_uv=float(np.median(E.std(1))),
                bad_zero_frac=1 - len(ep) / max(nz, 1), bad_ptp_frac=float(np.mean(ptp > 5 * np.median(ptp))))


def main():
    subs, subs6, D, F = load_all()
    G = {k: torch.tensor(v, device=dev) for k, v in groups().items()}
    dall = torch.stack([dval(F[q], D[q][1]) for q in subs6])
    rows = []
    for q in subs6 + [S034]:
        y, blk = D[q][1], D[q][2]
        X = F[q]
        r = dict(subject=q, special=int(q == S034))
        calm = blk <= 3
        Xr = D[q][0][torch.tensor(calm, device=dev)]
        for k, j in RAWF.items():
            r[k] = float(Xr[:, j].mean())
        r.update(raw_profile(q))
        if q == S034:
            dq = None
            for k in ("d_top20", "d_cal_max", "auc_cv5", "auc_xblk", "drift_cos", "drift_sign", "bias_shift",
                      "alpha_react", "sim_max", "sim_mean", "sim_top10", *[f"d_{g}" for g in G]):
                r[k] = np.nan
            rows.append(r)
            continue
        d1, d2, cons, order = cons_order(X, y, blk)
        r["d_top20"] = float(cons[order[:20]].mean())
        dq = dval(X[calm], y[calm])
        r["d_cal_max"] = float(dq.abs().max())
        r["alpha_react"] = float(dval(Xr[:, 58:59], y[calm])[0])
        # 校准段内 5 折：每块切 5 段连续片段，第 i 折取各块第 i 段
        ic = np.where(calm)[0]
        seg = np.zeros(len(ic), int)
        for b in range(4):
            ib = np.where(blk[ic] == b)[0]
            seg[ib] = np.minimum((np.arange(len(ib)) * 5) // len(ib), 4)
        Xc, yc = X[ic], y[ic]
        sc = np.zeros(len(ic))
        for i in range(5):
            te = seg == i
            tr = torch.tensor(~te, device=dev)
            sc[te] = lr_top(Xc[tr], yc[~te], Xc[torch.tensor(te, device=dev)])
        r["auc_cv5"] = auc(yc, sc)
        p1, p2 = blk <= 1, (blk >= 2) & (blk <= 3)
        s12 = lr_top(X[p1], y[p1], X[p2])
        s21 = lr_top(X[p2], y[p2], X[p1])
        s11 = lr_top(X[p1], y[p1], X[p1])
        r["auc_xblk"] = 0.5 * (auc(y[p2], s12) + auc(y[p1], s21))
        sd = s11.std() + 1e-6
        mid1 = 0.5 * (s11[y[p1] == 1].mean() + s11[y[p1] == 0].mean())
        mid2 = 0.5 * (s12[y[p2] == 1].mean() + s12[y[p2] == 0].mean())
        r["bias_shift"] = float(abs(mid2 - mid1) / sd)
        r["drift_cos"] = float(torch.nn.functional.cosine_similarity(d1, d2, 0))
        top = torch.argsort(d1.abs() + d2.abs(), descending=True)[:200]
        r["drift_sign"] = float((torch.sign(d1[top]) == torch.sign(d2[top])).float().mean())
        cos = torch.nn.functional.cosine_similarity(dall, dq[None], 1)
        cos[subs6.index(q)] = float("nan")
        cs = cos[~torch.isnan(cos)].sort(descending=True).values
        r["sim_max"], r["sim_mean"], r["sim_top10"] = float(cs[0]), float(cs.mean()), float(cs[:10].mean())
        for g, cols in G.items():
            r[f"d_{g}"] = float(cons[cols].sort(descending=True).values[:10].mean())
        rows.append(r)
        print(q, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)
    with open(HERE / "profile.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    correlate(rows)


def correlate(rows):
    best = json.load(open(HERE.parent / "22_subject" / "best.json"))
    tab = {r["subject"]: r for r in csv.DictReader(open(HERE / "synth_table.csv", encoding="utf-8-sig"))}
    meta = json.load(open(HERE / "cands_meta.json"))
    E = np.load(HERE / "cands_eval.npz")
    ev, es = E["ev"], list(E["subs"])
    base = meta["base"]
    fams = np.array([family(n) for n in base])
    jt = [i for i, t in enumerate(meta["thr"]) if not t.startswith("q")]
    qt = [i for i, t in enumerate(meta["thr"]) if t.startswith("q")]
    ie8 = base.index("ens8")
    tg = {}
    for r in rows:
        q = r["subject"]
        e = ev[es.index(q)][0]
        b = e[:len(base)]
        t = dict(ens8=float(b[ie8, 0]), oracle=best[q]["bacc"])
        t["gap"] = t["oracle"] - t["ens8"]
        jbest = np.nanmax(b[fams == "joint"])
        for fm in ("own", "idx", "sim", "b23", "g", "f"):
            t[f"adv_{fm}"] = float(np.nanmax(b[fams == fm]) - jbest) if np.any(~np.isnan(b[fams == fm])) else np.nan
        t["adv_quantile"] = float(np.nanmax(e[:, qt]) - np.nanmax(e[:, jt]))
        t["adv_median_vs_t0"] = float(np.nanmax(e[:, 1:4]) - np.nanmax(e[:, 0]))
        fs = tab[q]["families"].split("+")
        t["win_has_own"] = float("own" in fs)
        t["win_pure_joint"] = float(tab[q]["families"] in ("joint", "ens8+joint"))
        t["win_sim_b23"] = float("sim" in fs or "b23" in fs)
        t["win_quantile"] = float(tab[q]["thr_type"] == "分位数")
        t["win_rounds_ge3"] = float(tab[q]["round"] not in ("1", "2"))
        tg[q] = t
    keys = [k for k in rows[0] if k not in ("subject", "special")]
    rs = [r for r in rows if not r["special"]]
    out = []
    for tk in tg[rs[0]["subject"]]:
        tv = np.array([tg[r["subject"]][tk] for r in rs])
        for k in keys:
            pv = np.array([r[k] for r in rs], float)
            m = ~np.isnan(tv) & ~np.isnan(pv)
            rho, p = spearmanr(pv[m], tv[m])
            out.append(dict(target=tk, profile=k, rho=round(float(rho), 3), p=float(p), n=int(m.sum())))
    out = [o for o in out if not np.isnan(o["p"])]
    ps = np.array([o["p"] for o in out])
    o_ = np.argsort(ps)
    bh = np.empty(len(ps))
    bh[o_] = np.minimum.accumulate((ps[o_] * len(ps) / (np.arange(len(ps)) + 1))[::-1])[::-1]
    for o, b in zip(out, bh):
        o["q_fdr"] = float(min(b, 1))
    with open(HERE / "profile_corr.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    with open(HERE / "profile_targets.json", "w") as f:
        json.dump(tg, f, indent=1)
    for o in sorted(out, key=lambda o: o["p"]):
        if o["p"] < .05:
            print(f"{o['target']:18s} {o['profile']:14s} rho={o['rho']:+.3f} p={o['p']:.4f} q={o['q_fdr']:.3f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "corr":
        rows = [{k: (v if k == "subject" else float(v or "nan")) for k, v in r.items()}
                for r in csv.DictReader(open(HERE / "profile.csv", encoding="utf-8"))]
        correlate(rows)
    else:
        main()
