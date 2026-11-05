"""E45：在低组 29 人 LOSO 折上筛选候选方法（每折训练仍用其余 56 人）。

stage post        ：后处理类(d)及短/长窗分数融合(c2)，直接复用 e42_scores.npz（和 e44_scores_short.npz）。
stage train V [--all]：训练类候选，只跑低组 29 折（--all 跑 57 折），分数存 e45_scores_V.npz。
    a_cons  ：块间效应一致性特征选择（训练被试逐块对效应的 mean/std），K 由训练被试内按块对留一的内层 LR 选择
    b_late2 ：训练被试块4–5 样本权重 ×2（推理不需要流逝时间，合规）
    c_xgb   ：GPU XGBoost，与 E17 分数(e42)等权相加
stage eval        ：全部候选 → dt1200_hmm（后处理候选自带），低组全帧/块4–5，晋级判定，写 e45.json。
全部严格因果；块号/标签只用于训练被试（a、b）和评价；E43 分组只用于决定跑哪些折与汇总。
"""
from __future__ import annotations

import argparse
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


def _imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


E42 = _imp("e42lib", TESTS / "42_e17_loso_hmm" / "e42.py")
E44 = _imp("e44lib", TESTS / "44_late_drift" / "e44.py")
metrics, summary = E42.metrics, E42.summary
DEV = "cuda"
RESULT = HERE / "e45.json"
S42 = TESTS / "42_e17_loso_hmm" / "e42_scores.npz"
SSHORT = TESTS / "44_late_drift" / "e44_scores_short.npz"


def const(B, T, d):
    return torch.full((B, T), -math.log(d), device=DEV)


def dt_hmm(S, hl=1200, dwell=300, temp=1.0):
    Z = S - E44.ema_mean(S, hl)
    return E44.hmm(Z / temp, const(*S.shape, dwell))


def causal_quantile(s, W):
    """s (T,) → 当前分数在过去 W 帧（含当前）中的分位数，映射为 logit。"""
    T = s.shape[0]
    i = torch.arange(T, device=DEV)
    mask = (i[None, :] <= i[:, None]) & (i[:, None] - i[None, :] < W)
    less = ((s[None, :] < s[:, None]) & mask).sum(1).float()
    eq = ((s[None, :] == s[:, None]) & mask).sum(1).float()
    n = mask.sum(1).float()
    q = (less + 0.5 * eq + 0.5) / (n + 1)
    return torch.logit(q)


def post_candidates(subs, s42, sshort):
    S = E44.pad(subs, s42)
    B, T = S.shape
    lens = [len(s42[q]) for q in subs]
    V = {"base_dt1200_hmm": dt_hmm(S)}
    # d1 多半衰期去趋势组合
    for hls in ((300, 1200), (600, 2400), (300, 1200, 4800), (1200, 4800)):
        Z = torch.stack([S - E44.ema_mean(S, h) for h in hls]).mean(0)
        V["d1_multi" + "_".join(map(str, hls))] = E44.hmm(Z, const(B, T, 300))
    # d2 因果分位数归一化（过去 W 帧排名，与固定参考无关，可抵消单调漂移）
    for W in (600, 1200, 99999):
        Q = torch.zeros_like(S)
        for i, q in enumerate(subs):
            Q[i, :lens[i]] = causal_quantile(S[i, :lens[i]], W)
        V[f"d2_quant{W}_hmm"] = E44.hmm(Q, const(B, T, 300))
        Z = S - E44.ema_mean(S, 1200)
        V[f"d2_quant{W}+dt1200_hmm"] = E44.hmm(0.5 * (Q + Z), const(B, T, 300))
    # d3 保守门控：更长驻留 / 降温 / 迟滞（|后验|<m 时保持上一判决符号）
    V["d3_dwell600"] = dt_hmm(S, dwell=600)
    V["d3_dwell1200"] = dt_hmm(S, dwell=1200)
    V["d3_temp2"] = dt_hmm(S, temp=2.0)
    H = V["base_dt1200_hmm"]
    for m in (1.0, 2.0):
        out = H.clone()
        prev = H[:, 0].sign()
        for t in range(T):
            h = H[:, t]
            prev = torch.where(h.abs() >= m, h.sign(), prev)
            # 判决符号由迟滞决定，幅值保留（AUC 用连续分数）
            out[:, t] = torch.where(prev > 0, h.abs().clamp_min(1e-3), -h.abs().clamp_min(1e-3))
        V[f"d3_hyst{m}"] = out
    # c2 短窗/长窗子模型融合（E44 short 分数）
    if sshort is not None:
        S2 = E44.pad(subs, sshort)
        Z1, Z2 = S - E44.ema_mean(S, 1200), S2 - E44.ema_mean(S2, 1200)
        V["c2_short_static0.5"] = E44.hmm(0.5 * (Z1 + Z2), const(B, T, 300))
        # 动态加权：按各自因果分数离散度（双峰分离越好，std 越大）加权
        sd1 = (E44.ema_mean(Z1 ** 2, 600) - E44.ema_mean(Z1, 600) ** 2).clamp_min(1e-6).sqrt()
        sd2 = (E44.ema_mean(Z2 ** 2, 600) - E44.ema_mean(Z2, 600) ** 2).clamp_min(1e-6).sqrt()
        w1 = sd1 / (sd1 + sd2)
        V["c2_short_dynamic"] = E44.hmm(2 * (w1 * Z1 + (1 - w1) * Z2), const(B, T, 300))
    return {k: {q: v[i, :lens[i]].cpu().numpy() for i, q in enumerate(subs)} for k, v in V.items()}


# ---------------- 训练 ----------------
def fit_lr_w(X, y, C, w=None, iters=200):
    if w is None:
        return L.fit_lr(X, y, C, iters)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    wv = torch.zeros(X.shape[1], device=DEV, requires_grad=True)
    b = torch.zeros(1, device=DEV, requires_grad=True)
    yt = torch.tensor(y, device=DEV, dtype=torch.float32)
    wt = torch.tensor(w, device=DEV, dtype=torch.float32)
    wt = torch.where(yt == 1, wt * (wt[yt == 0].sum() / wt[yt == 1].sum()), wt)
    wt = wt / wt.mean()
    opt = torch.optim.LBFGS([wv, b], max_iter=iters, line_search_fn="strong_wolfe")
    lam = 1.0 / (C * len(y))

    def closure():
        opt.zero_grad()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(Z @ wv + b, yt, reduction="none") * wt).mean() \
            + 0.5 * lam * (wv ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    wv, b = wv.detach(), b.detach()
    return lambda X: ((X - mu) / sd) @ wv + b


def fit_mlp_w(X, y, w=None, hid=128, ep=30, wd=1e-3, drop=0.3, seed=0):
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    net = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(X.shape[1], hid), torch.nn.GELU(),
                              torch.nn.Dropout(drop), torch.nn.Linear(hid, 1)).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=wd)
    yt = torch.tensor(y, device=DEV, dtype=torch.float32)
    wt = torch.ones_like(yt) if w is None else torch.tensor(w, device=DEV, dtype=torch.float32)
    wt = torch.where(yt == 1, wt * (wt[yt == 0].sum() / wt[yt == 1].sum()), wt)
    wt = wt / wt.mean()
    n = len(y)
    for _ in range(ep):
        for idx in torch.randperm(n, device=DEV).split(512):
            opt.zero_grad()
            (torch.nn.functional.binary_cross_entropy_with_logits(net(Z[idx]).squeeze(1), yt[idx], reduction="none")
             * wt[idx]).mean().backward()
            opt.step()
    net.eval()

    def f(X):
        with torch.no_grad():
            return net((X - mu) / sd).squeeze(1)
    return f


def ensemble(X, y, tr_feats, te_feat, w=None):
    """E17：LR(C=.001)+3 种子 MLP，融合尺度只由训练折估计。"""
    flr = fit_lr_w(X, y, 0.001, w)
    fms = [fit_mlp_w(X, y, w, seed=s) for s in range(3)]
    sl = float(torch.cat([flr(f) for f in tr_feats]).std())
    sm = float(torch.cat([torch.stack([m(f) for m in fms]).mean(0) for f in tr_feats]).std())
    return (flr(te_feat) / sl + torch.stack([m(te_feat) for m in fms]).mean(0) / sm).cpu().numpy()


def pair_effects(F, D, subs):
    """每人每块对 (0,1)(2,3)(4,5) 的标准化效应 (task−relax)/pooled sd → {q: (3,d)}，缺失为 nan。"""
    out = {}
    for q in subs:
        X, y, b = F[q], D[q][1], D[q][2]
        rows = []
        for k in range(3):
            r = torch.tensor((np.isin(b, (2 * k, 2 * k + 1))) & (y == 0), device=DEV)
            t = torch.tensor((np.isin(b, (2 * k, 2 * k + 1))) & (y == 1), device=DEV)
            if r.sum() < 30 or t.sum() < 30:
                rows.append(torch.full((X.shape[1],), float("nan"), device=DEV))
                continue
            sd = torch.sqrt(0.5 * (X[r].var(0) + X[t].var(0))) + 1e-6
            rows.append((X[t].mean(0) - X[r].mean(0)) / sd)
        out[q] = torch.stack(rows)
    return out


def consistency(E):
    """E (n,d) 效应 → |mean|/std，惩罚块间/被试间方向不一致的特征。"""
    m = torch.nanmean(E, 0)
    s = torch.sqrt(torch.nanmean((E - m) ** 2, 0)) + 1e-3
    return (m / s).abs()


KS = (500, 1000, 2000, 99999)


KS_SMALL = (100, 250, 500)


def select_K(F, D, tr, PE, ks=KS):
    """训练被试内按块对留一：特征一致性只用其余两对，LR 在其余两对上训练，在留出对上算逐人 AUC。"""
    score = {K: [] for K in ks}
    for k in range(3):
        keep = [j for j in range(3) if j != k]
        cons = consistency(torch.cat([PE[p][keep] for p in tr]))
        order = torch.argsort(cons, descending=True)
        trm = {p: ~np.isin(D[p][2], (2 * k, 2 * k + 1)) for p in tr}
        tem = {p: np.isin(D[p][2], (2 * k, 2 * k + 1)) for p in tr}
        for K in ks:
            idx = order[:min(K, len(order))]
            X = torch.cat([F[p][torch.tensor(trm[p], device=DEV)][:, idx] for p in tr])
            y = np.concatenate([D[p][1][trm[p]] for p in tr])
            f = L.fit_lr(X, y, 0.001)
            for p in tr:
                if tem[p].sum() and len(set(D[p][1][tem[p]])) == 2:
                    s = f(F[p][torch.tensor(tem[p], device=DEV)][:, idx]).cpu().numpy()
                    score[K].append(metrics(D[p][1][tem[p]], s)["auc"])
            del X, f
            torch.cuda.empty_cache()
    mean = {K: float(np.mean(v)) for K, v in score.items()}
    return max(mean, key=mean.get), mean


def stage_train(subs, D, variant, folds):
    F = E42.build_features(subs, D)
    s42 = np.load(S42)
    out, info = {}, {}
    PE = pair_effects(F, D, subs) if variant.startswith("a_cons") else None
    for i, q in enumerate(folds):
        t0 = time.time()
        tr = [p for p in subs if p != q]
        y = np.concatenate([D[p][1] for p in tr])
        if variant.startswith("a_cons"):
            K, inner = select_K(F, D, tr, PE, KS_SMALL if variant == "a_cons_s" else KS)
            cons = consistency(torch.cat([PE[p] for p in tr]))
            idx = torch.argsort(cons, descending=True)[:min(K, cons.shape[0])]
            trf = [F[p][:, idx] for p in tr]
            out[q] = ensemble(torch.cat(trf), y, trf, F[q][:, idx])
            info[q] = {"K": K, "inner_auc": inner}
            del trf
        elif variant.startswith("b_late"):
            wl = float(variant[6:])
            w = np.concatenate([np.where(D[p][2] >= 4, wl, 1.0) for p in tr])
            trf = [F[p] for p in tr]
            out[q] = ensemble(torch.cat(trf), y, trf, F[q], w)
        elif variant == "c_xgb":
            import xgboost as xgb
            # 无 cupy：经主机 numpy 传入，训练/预测仍在 GPU (device=cuda)
            X = torch.cat([F[p] for p in tr]).cpu().numpy()
            dm = xgb.QuantileDMatrix(X, label=y, max_bin=64)
            # 驱动仅 CUDA 11.4，使用 xgboost 1.7.6 (CUDA 11.8 构建) 的 gpu_hist
            prm = dict(tree_method="gpu_hist", predictor="gpu_predictor", objective="binary:logistic", max_depth=4, eta=0.1,
                       subsample=0.8, colsample_bytree=0.3, min_child_weight=20, reg_lambda=5.0,
                       scale_pos_weight=float((y == 0).sum() / (y == 1).sum()), max_bin=64, seed=0)
            bst = xgb.train(prm, dm, 300)
            del dm
            tr_m = bst.inplace_predict(X, predict_type="margin")
            sx = float(np.std(tr_m.get() if hasattr(tr_m, "get") else tr_m))
            del X
            te_m = bst.inplace_predict(F[q].cpu().numpy(), predict_type="margin")
            te_m = te_m.get() if hasattr(te_m, "get") else np.asarray(te_m)
            out[q + "_xgb"] = te_m / sx
            # e42 分数本身已是 LR/std+MLP/std（两项），XGB 按等权第三项加入
            out[q] = s42[q] + te_m / sx
        torch.cuda.empty_cache()
        print(f"[{variant} {i + 1}/{len(folds)}] {q} {metrics(D[q][1], out[q])} {info.get(q, '')} "
              f"{time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(HERE / f"e45_scores_{variant}.npz", **out)
    if info:
        (HERE / f"e45_info_{variant}.json").write_text(json.dumps(info, indent=1), encoding="utf-8")


# ---------------- 评价 ----------------
def rows_of(D, qs, sc):
    full = [metrics(D[q][1], sc[q]) for q in qs]
    late = [metrics(D[q][1][D[q][2] >= 4], sc[q][D[q][2] >= 4]) for q in qs]
    return summary(full), summary(late)


def stage_eval(subs, D):
    grp = E44.groups()
    low = [q for q in subs if grp[q] == "low"]
    s42 = {q: v for q, v in np.load(S42).items() if q in subs}
    sshort = {q: v for q, v in np.load(SSHORT).items() if q in subs} if SSHORT.exists() else None
    cands = post_candidates(subs, s42, sshort)
    kind = {k: "post" for k in cands}
    for p in sorted(HERE.glob("e45_scores_*.npz")):
        name = p.stem[len("e45_scores_"):]
        data = np.load(p)
        have = [q for q in subs if q in data.files]
        sc = {q: data[q] for q in have}
        sub_full = len(have) == len(subs)
        # 训练类：缺失折用 e42 填充以便统一过 dt1200_hmm；只在已训练的人上汇总
        filled = {q: sc.get(q, s42[q]) for q in subs}
        H = dt_hmm(E44.pad(subs, filled)).cpu().numpy()
        cands[name] = {q: H[i, :len(filled[q])] for i, q in enumerate(subs) if q in sc}
        kind[name] = "train_all" if sub_full else "train_low"
        if name == "c_xgb" and all(q + "_xgb" in data.files for q in have):
            xs = {q: data[q + "_xgb"] for q in have}
            filled = {q: xs.get(q, s42[q]) for q in subs}
            H = dt_hmm(E44.pad(subs, filled)).cpu().numpy()
            cands["c_xgb_only"] = {q: H[i, :len(filled[q])] for i, q in enumerate(subs) if q in xs}
            kind["c_xgb_only"] = kind[name]
    # 组合候选：原始分数等权平均后再过 dt1200_hmm（e42 / short 为全量 LOSO 分数）
    raw = {"e42": s42, "short": sshort}
    for p in sorted(HERE.glob("e45_scores_*.npz")):
        data = np.load(p)
        raw[p.stem[len("e45_scores_"):]] = {q: data[q] for q in subs if q in data.files}
    for combo in (("a_cons", "short"), ("b_late2", "short"), ("b_late4", "short"), ("a_cons", "b_late2"),
                  ("a_cons", "b_late2", "short"), ("a_cons", "b_late4", "short"), ("c_xgb", "short")):
        if not all(raw.get(c) for c in combo):
            continue
        have = [q for q in subs if all(q in raw[c] for c in combo)]
        filled = {q: (np.mean([raw[c][q] for c in combo], 0) if q in have else s42[q]) for q in subs}
        H = dt_hmm(E44.pad(subs, filled)).cpu().numpy()
        name = "combo_" + "+".join(combo)
        cands[name] = {q: H[i, :len(filled[q])] for i, q in enumerate(subs) if q in have}
        kind[name] = "train_all" if len(have) == len(subs) else "train_low"
    res = {}
    bf, bl = rows_of(D, low, cands["base_dt1200_hmm"])
    Bf, Bl = rows_of(D, subs, cands["base_dt1200_hmm"])
    for k, sc in cands.items():
        if not all(q in sc for q in low):
            continue
        lf, ll = rows_of(D, low, sc)
        promote = (ll["bacc"] - bl["bacc"] >= 0.02) and (ll["auc"] - bl["auc"] >= 0.02)
        r = {"kind": kind[k], "low_full": lf, "low_late45": ll,
             "d_low_late_bacc": ll["bacc"] - bl["bacc"], "d_low_late_auc": ll["auc"] - bl["auc"], "promote": promote}
        if all(q in sc for q in subs):
            af, al = rows_of(D, subs, sc)
            r["all_full"], r["all_late45"] = af, al
            r["not_below_base_all"] = all(af[m] >= Bf[m] - 1e-9 and al[m] >= Bl[m] - 1e-9 for m in ("bacc", "auc", "f1"))
        res[k] = r
        print(f"{k:28s} {kind[k]:9s} low full {lf['bacc']:.4f}/{lf['auc']:.4f}/{lf['f1']:.4f} "
              f"late {ll['bacc']:.4f}/{ll['auc']:.4f}/{ll['f1']:.4f} "
              f"d {r['d_low_late_bacc']:+.4f}/{r['d_low_late_auc']:+.4f} {'PROMOTE' if promote else ''}"
              + (f" | all {r['all_full']['bacc']:.4f}/{r['all_full']['auc']:.4f}/{r['all_full']['f1']:.4f} "
                 f"late {r['all_late45']['bacc']:.4f}/{r['all_late45']['auc']:.4f}/{r['all_late45']['f1']:.4f}"
                 if "all_full" in r else ""), flush=True)
    out = {"protocol": "低组29人LOSO；所有候选最后都经 dt1200_hmm 或自身后处理；晋级=低组块4–5 bACC 与 AUC 均 ≥ 基线+0.02",
           "low_subjects": low, "baseline_low": {"full": bf, "late45": bl},
           "baseline_all": {"full": Bf, "late45": Bl}, "candidates": res}
    for name in ("a_cons", "a_cons_s"):
        p = HERE / f"e45_info_{name}.json"
        if p.exists():
            info = json.loads(p.read_text(encoding="utf-8"))
            ks = sorted({v["K"] for v in info.values()})
            out[f"{name}_K_counts"] = {str(K): sum(v["K"] == K for v in info.values()) for K in ks}
    RESULT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("post", "train", "eval"))
    ap.add_argument("variant", nargs="?")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    subs, D = L.load()
    if a.stage == "train":
        grp = E44.groups()
        folds = subs if a.all else [q for q in subs if grp[q] == "low"]
        stage_train(subs, D, a.variant, folds)
    else:
        stage_eval(subs, D)
    print(f"elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
