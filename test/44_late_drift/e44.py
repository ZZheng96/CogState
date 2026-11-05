"""E44：低组后段失效诊断与因果修复消融。

stage diag   ：用 E42 LOSO 逐帧分数做逐块诊断，并对每个标准化窗口单独做 LR LOSO。
stage train V：重训特征变体（LR+3MLP，融合尺度只用训练折），分数存 e44_scores_V.npz。
stage eval   ：分数层变体（因果去趋势、门控 HMM）× 特征变体，全帧/块4–5，分组分解。
全部严格因果；块号只用于评价与诊断；E43 分组只用于诊断分解，不参与选参。
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent
sys.path.insert(0, str(TESTS / "17_compliant"))
import loso as L  # noqa: E402
from multiscale import fit_mlp  # noqa: E402

spec = importlib.util.spec_from_file_location("e42lib", TESTS / "42_e17_loso_hmm" / "e42.py")
E42 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E42)
metrics, summary = E42.metrics, E42.summary
DEV = "cuda"
REV = ("004", "009", "010", "050")
RESULT = HERE / "e44.json"
# 特征变体：("w", n) 为过去 n 帧滑窗；("all",) 为全部过去帧；("ema", h) 为半衰期 h 帧的指数遗忘统计
FEATURE_VARIANTS = {
    "e42": [("w", 60), ("w", 120), ("w", 300), ("w", 600), ("all",)],
    "no_all": [("w", 60), ("w", 120), ("w", 300), ("w", 600)],
    "short": [("w", 60), ("w", 120), ("w", 300)],
    "ema": [("w", 60), ("w", 120), ("w", 300), ("w", 600), ("ema", 600), ("ema", 1800)],
}


def groups():
    with open(TESTS / "43_group_analysis" / "group_members.csv", encoding="utf-8-sig") as f:
        return {r["subject"]: r["group"] for r in csv.DictReader(f)}


def ema_stats(X, hl):
    """因果指数遗忘均值/标准差（含当前帧），按权重归一化；hl=inf 等价于全部过去帧。"""
    lam = 0.5 ** (1.0 / hl)
    n = X.shape[0]
    t = torch.arange(n, device=X.device, dtype=torch.float64)
    g = lam ** (-t)[:, None]
    Xd = X.double()
    w = (lam ** t)[:, None]
    s1 = (Xd * g).cumsum(0) * w
    s2 = (Xd ** 2 * g).cumsum(0) * w
    sw = g.cumsum(0) * w
    m = s1 / sw
    v = s2 / sw - m ** 2
    return m.float(), v.clamp_min(1e-12).sqrt().float()


def block_feats(D, subs, kind):
    out = {}
    for q in subs:
        X = D[q][0]
        if kind[0] == "w":
            m, sd = L.run_stats_t(X, kind[1])
        elif kind[0] == "all":
            m, sd = L.run_stats_t(X, 99999)
        else:
            m, sd = ema_stats(X, kind[1])
        out[q] = ((X - m) / sd).clamp(-10, 10)
    return out


def build(D, subs, kinds):
    F = None
    for k in kinds:
        Fk = block_feats(D, subs, k)
        F = Fk if F is None else {q: torch.cat([F[q], Fk[q]], 1) for q in subs}
        del Fk
        torch.cuda.empty_cache()
    return F


def loso_scores(D, subs, F, use_mlp=True, tag=""):
    scores = {}
    for i, q in enumerate(subs):
        tr = [p for p in subs if p != q]
        X = torch.cat([F[p] for p in tr])
        y = np.concatenate([D[p][1] for p in tr])
        flr = L.fit_lr(X, y, 0.001)
        if not use_mlp:
            sl = float(torch.cat([flr(F[p]) for p in tr]).std())
            scores[q] = (flr(F[q]) / sl).cpu().numpy()
        else:
            fms = [fit_mlp(X, y, seed=s) for s in range(3)]
            sl = float(torch.cat([flr(F[p]) for p in tr]).std())
            sm = float(torch.cat([torch.stack([f(F[p]) for f in fms]).mean(0) for p in tr]).std())
            scores[q] = (flr(F[q]) / sl + torch.stack([f(F[q]) for f in fms]).mean(0) / sm).cpu().numpy()
            del fms
        del X, flr
        torch.cuda.empty_cache()
        print(f"[{tag} {i + 1}/57] {q} {metrics(D[q][1], scores[q])}", flush=True)
    return scores


def eval_rows(D, subs, scores):
    full = [metrics(D[q][1], scores[q]) for q in subs]
    late = [metrics(D[q][1][D[q][2] >= 4], scores[q][D[q][2] >= 4]) for q in subs]
    return full, late


def decompose(subs, rows, grp):
    out = {"all": summary(rows)}
    for name, keep in (("high", lambda q: grp[q] == "high"), ("low", lambda q: grp[q] == "low"),
                       ("rev4", lambda q: q in REV)):
        out[name] = summary([r for q, r in zip(subs, rows) if keep(q)])
    return out


# ---------------- 诊断 ----------------
def stage_diag(subs, D):
    grp = groups()
    s42 = np.load(TESTS / "42_e17_loso_hmm" / "e42_scores.npz")
    per = {}
    for q in subs:
        y, b, s = D[q][1], D[q][2], s42[q]
        per[q] = {"block_mean": [float(s[b == k].mean()) if (b == k).any() else float("nan") for k in range(6)],
                  "block_y": [int(y[b == k][0]) if (b == k).any() else -1 for k in range(6)],
                  "auc_b01": metrics(y[b <= 1], s[b <= 1])["auc"],
                  "auc_b23": metrics(y[(b >= 2) & (b <= 3)], s[(b >= 2) & (b <= 3)])["auc"],
                  "auc_b45": metrics(y[b >= 4], s[b >= 4])["auc"]}
        # 块内晚期（每块后半）均值，查看块内是否回落/漂移
        per[q]["block_mean_2nd_half"] = []
        for k in range(6):
            idx = np.where(b == k)[0]
            per[q]["block_mean_2nd_half"].append(float(s[idx[len(idx) // 2:]].mean()) if len(idx) else float("nan"))
    agg = {}
    for name, keep in (("high", lambda q: grp[q] == "high"), ("low", lambda q: grp[q] == "low"),
                       ("rev4", lambda q: q in REV), ("all", lambda q: True)):
        qs = [q for q in subs if keep(q)]
        bm = np.array([per[q]["block_mean"] for q in qs])
        np.seterr(invalid="ignore")
        agg[name] = {"n": len(qs), "block_mean": np.nanmean(bm, 0).round(3).tolist(),
                     "auc_b01": float(np.nanmean([per[q]["auc_b01"] for q in qs])),
                     "auc_b23": float(np.nanmean([per[q]["auc_b23"] for q in qs])),
                     "auc_b45": float(np.nanmean([per[q]["auc_b45"] for q in qs])),
                     # relax 均值随时间的变化(块4−块0)、task(块5−块1)，以及后段分离度
                     "relax_shift_b4_b0": float(np.nanmean(bm[:, 4] - bm[:, 0])),
                     "task_shift_b5_b1": float(np.nanmean(bm[:, 5] - bm[:, 1])),
                     "sep_b01": float(np.nanmean(bm[:, 1] - bm[:, 0])),
                     "sep_b45": float(np.nanmean(bm[:, 5] - bm[:, 4])),
                     "n_sep45_neg": int((bm[:, 5] < bm[:, 4]).sum()),
                     "n_relax4_above_task1": int((bm[:, 4] > bm[:, 1]).sum())}
    for k, v in agg.items():
        print(k, json.dumps(v, ensure_ascii=False), flush=True)
    # 每个标准化窗口单独建模（LR LOSO）
    win = {}
    for kind in [("raw",), ("w", 60), ("w", 120), ("w", 300), ("w", 600), ("all",), ("ema", 600)]:
        if kind[0] == "raw":
            F = {q: D[q][0] for q in subs}
        else:
            F = block_feats(D, subs, kind)
        sc = loso_scores(D, subs, F, use_mlp=False, tag="diag" + "_".join(map(str, kind)))
        full, late = eval_rows(D, subs, sc)
        win["_".join(map(str, kind))] = {"full": decompose(subs, full, grp), "late45": decompose(subs, late, grp)}
        del F
        torch.cuda.empty_cache()
    out = {"per_subject": per, "groups": agg, "window_lr": win}
    (HERE / "e44_diag.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in win.items():
        print(k, "full", {g: round(v["full"][g]["auc"], 3) for g in v["full"]},
              "late", {g: round(v["late45"][g]["auc"], 3) for g in v["late45"]})


# ---------------- 训练特征变体 ----------------
def stage_train(subs, D, name):
    F = build(D, subs, FEATURE_VARIANTS[name])
    sc = loso_scores(D, subs, F, use_mlp=True, tag=name)
    np.savez_compressed(HERE / f"e44_scores_{name}.npz", subjects=np.array(subs), **sc)


# ---------------- 分数层后处理 ----------------
def pad(subs, scores):
    T = max(len(scores[q]) for q in subs)
    S = torch.zeros(len(subs), T, device=DEV)
    for i, q in enumerate(subs):
        S[i, :len(scores[q])] = torch.tensor(scores[q], device=DEV)
    return S


def ema_mean(S, hl):
    """逐行因果 EMA 均值（含当前帧，首帧初始化）。"""
    a = 1 - 0.5 ** (1 / hl)
    m = S[:, 0].clone()
    out = torch.empty_like(S)
    out[:, 0] = m
    for t in range(1, S.shape[1]):
        m = m + a * (S[:, t] - m)
        out[:, t] = m
    return out


def hmm(S, log_switch):
    """两状态对称前向 HMM；log_switch (B,T) 允许逐帧转移概率；temperature=1。"""
    log_stay = torch.log1p(-torch.exp(log_switch))
    lp = torch.full((S.shape[0], 2), -math.log(2), device=DEV)
    out = torch.empty_like(S)
    for t in range(S.shape[1]):
        z = S[:, t]
        em = torch.stack((-torch.nn.functional.softplus(z), -torch.nn.functional.softplus(-z)), 1)
        if t:
            ls, lw = log_stay[:, t], log_switch[:, t]
            lp = torch.stack((torch.logaddexp(lp[:, 0] + ls, lp[:, 1] + lw),
                              torch.logaddexp(lp[:, 0] + lw, lp[:, 1] + ls)), 1)
        u = lp + em
        lp = u - torch.logsumexp(u, 1, keepdim=True)
        out[:, t] = lp[:, 1] - lp[:, 0]
    return out


def post_variants(S):
    """返回 {名称: (B,T)}。所有变换只用当前及过去分数。"""
    B, T = S.shape
    const = lambda d: torch.full((B, T), -math.log(d), device=DEV)  # noqa: E731
    V = {"raw": S, "hmm_d300": hmm(S, const(300))}
    for hl in (300, 600, 1200):
        Z = S - ema_mean(S, hl)
        V[f"dt{hl}"] = Z
        V[f"dt{hl}_hmm"] = hmm(Z, const(300))
        V[f"mix{hl}_hmm"] = hmm(0.5 * (S + Z), const(300))
    # 变化检测门控：短/长 EMA 差超过 k 倍因果分数标准差时，dwell 由 300 降为 30
    m30, m300 = ema_mean(S, 30), ema_mean(S, 300)
    sd = (ema_mean(S ** 2, 300) - m300 ** 2).clamp_min(1e-6).sqrt()
    for k in (0.5, 1.0):
        fire = (m30 - m300).abs() > k * sd
        V[f"gate{k}_hmm"] = hmm(S, torch.where(fire, const(30), const(300)))
    # 后验翻转率门控：过去120帧原始符号翻转率高（>0.2）则关闭 HMM（直接用原始分数）
    sg = (S > 0).float()
    flip = torch.zeros_like(S)
    flip[:, 1:] = (sg[:, 1:] != sg[:, :-1]).float()
    rate = torch.cat([torch.zeros(B, 1, device=DEV), flip.cumsum(1)], 1)
    idx = torch.arange(1, T + 1, device=DEV)
    lo = (idx - 120).clamp_min(0)
    rate = (rate[:, idx] - rate[:, lo]) / (idx - lo)
    H = V["hmm_d300"]
    V["fliprate_gate"] = torch.where(rate > 0.2, S, H)
    return V


def stage_eval(subs, D):
    grp = groups()
    srcs = {"e42": TESTS / "42_e17_loso_hmm" / "e42_scores.npz"}
    for name in FEATURE_VARIANTS:
        p = HERE / f"e44_scores_{name}.npz"
        if name != "e42" and p.exists():
            srcs[name] = p
    res, table = {}, []
    for src, path in srcs.items():
        data = np.load(path)
        scores = {q: data[q] for q in subs}
        V = post_variants(pad(subs, scores))
        for pv, M in V.items():
            M = M.cpu().numpy()
            sc = {q: M[i, :len(scores[q])] for i, q in enumerate(subs)}
            full, late = eval_rows(D, subs, sc)
            key = f"{src}|{pv}"
            res[key] = {"full": decompose(subs, full, grp), "late45": decompose(subs, late, grp),
                        "per_subject_full_bacc": [r["bacc"] for r in full],
                        "per_subject_late_bacc": [r["bacc"] for r in late]}
            f, l = res[key]["full"]["all"], res[key]["late45"]["all"]
            table.append((key, f, l))
            print(f"{key:24s} full {f['bacc']:.4f}/{f['auc']:.4f}/{f['f1']:.4f} n90={f['n90']} "
                  f"late {l['bacc']:.4f}/{l['auc']:.4f}/{l['f1']:.4f} "
                  f"low_late {res[key]['late45']['low']['bacc']:.4f}", flush=True)
    # 嵌套 LOSO：每个目标排除本人，用其余56人 (全帧bACC+块4–5 bACC)/2 选变体
    keys = list(res)
    E = np.array([[0.5 * (a + b) for a, b in zip(res[k]["per_subject_full_bacc"], res[k]["per_subject_late_bacc"])]
                  for k in keys])  # (K,57)
    chosen = []
    for i in range(len(subs)):
        chosen.append(keys[int(np.argmax(np.delete(E, i, axis=1).mean(1)))])
    nested_full, nested_late = [], []
    for i, q in enumerate(subs):
        k = chosen[i]
        src, pv = k.split("|")
        nested_full.append(None)
    # 重新计算所选变体的逐人完整指标
    cache = {}
    for i, q in enumerate(subs):
        k = chosen[i]
        if k not in cache:
            src, pv = k.split("|")
            data = np.load(srcs[src])
            scores = {p: data[p] for p in subs}
            M = post_variants(pad(subs, scores))[pv].cpu().numpy()
            cache[k] = {p: M[j, :len(scores[p])] for j, p in enumerate(subs)}
        s = cache[k][q]
        y, b = D[q][1], D[q][2]
        nested_full[i] = metrics(y, s)
        nested_late.append(metrics(y[b >= 4], s[b >= 4]))
    res["nested_select"] = {"full": decompose(subs, nested_full, grp), "late45": decompose(subs, nested_late, grp),
                            "chosen_counts": {k: chosen.count(k) for k in set(chosen)}}
    for k in res:
        res[k].pop("per_subject_full_bacc", None)
        res[k].pop("per_subject_late_bacc", None)
    RESULT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    n = res["nested_select"]
    print("nested", n["chosen_counts"], n["full"]["all"], n["late45"]["all"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("diag", "train", "eval"))
    ap.add_argument("variant", nargs="?")
    a = ap.parse_args()
    t0 = time.time()
    subs, D = L.load()
    if a.stage == "diag":
        stage_diag(subs, D)
    elif a.stage == "train":
        stage_train(subs, D, a.variant)
    else:
        stage_eval(subs, D)
    print(f"elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
