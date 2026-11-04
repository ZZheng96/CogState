"""E27：通用 + 个体子空间  w_i = a·w0 + U_k z_i（+ 因果均值/分位阈值）。

嵌套 LOSO：对被试 i，w0/U 只用其余 56 人逐人 LR 权重（单位化后均值 / PCA）；
z_i 只用本人块 0–3 标签估计（ridge 收缩向 0，λ 在校准块内按「留一对块（R,T）」4 折 AUC 选）。
阈值：e26lib THR2 85 种因果阈值，按 LOSO（其余 56 人平均测试 bACC 最大）选，被评被试不参与。
特征集：raw（878，本人校准段 z）、rc300/rc600（减过去 w 帧因果均值后再 z）、cat（E19–E23 的 4390 维因果多窗 z）。
034 只有块 3（全 task）：需本人标签的模型一律退化为 k0（纯通用）。
运行：.venv\\Scripts\\python.exe test/27_common_individual/e27.py
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
sys.path.insert(0, str(T / "17_compliant"))
sys.path.insert(0, str(T / "18_calib"))
import loso as L  # noqa: E402
from calib2 import WS as CWS  # noqa: E402

spec = importlib.util.spec_from_file_location("e26lib", T / "26_merge2" / "e26lib.py")
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)

dev = "cuda"
S034 = "034"
KS = (1, 2, 3, 5)
LAM_K = torch.tensor([1e-3, 1e-2, 1e-1, 1.0, 10.0], device=dev)
LAM_D = torch.tensor([0.03, 0.1, 0.3, 1.0, 3.0], device=dev)
LAM_WJ, LAM_POOL = 0.3, 0.01
FOLDS = [((0, 1), (2, 3)), ((0, 3), (2, 1)), ((2, 1), (0, 3)), ((2, 3), (0, 1))]  # (held, train)


def bal_w(y, mask=None):
    y = torch.as_tensor(y, device=dev).float()
    m = torch.ones_like(y) if mask is None else mask.float()
    w = torch.zeros_like(y)
    for c in (0, 1):
        s = m * (y == c).float()
        w = w + s / s.sum().clamp(min=1) * .5
    return w


def lbfgs_lr(X, y, sw, lams, s0=None, iters=120):
    """批量 λ 的加权 ridge LR；s0 为自由系数的偏置项（不罚）。返回 predict(X, s0)->(n,L)。"""
    Lm, d = len(lams), X.shape[1]
    W = torch.zeros(Lm, d, device=dev, requires_grad=True)
    b = torch.zeros(Lm, device=dev, requires_grad=True)
    a = torch.zeros(Lm, device=dev, requires_grad=True)
    yt = torch.as_tensor(y, device=dev).float()[:, None]
    params = [W, b] + ([a] if s0 is not None else [])
    opt = torch.optim.LBFGS(params, max_iter=iters, line_search_fn="strong_wolfe")

    def logit(Xx, ss):
        o = Xx @ W.T + b
        return o + ss[:, None] * a if ss is not None else o

    def closure():
        opt.zero_grad()
        l = (torch.nn.functional.binary_cross_entropy_with_logits(logit(X, s0), yt.expand(-1, Lm), reduction="none")
             * sw[:, None]).sum() + 0.5 * (lams[:, None] * W ** 2).sum()
        l.backward()
        return l
    opt.step(closure)
    W, b, a = W.detach(), b.detach(), a.detach()
    f = lambda Xx, ss=None: logit(Xx, ss)  # noqa: E731
    f.W = W
    return f


def newton_lr(X, y, sw, lams, pen, iters=30):
    """小维度批量 λ 的 ridge LR（Newton）。X (n,p)，pen (p,) 罚掩码。返回 β (L,p)。"""
    n, p = X.shape
    yt = torch.as_tensor(y, device=dev).float()
    B = torch.zeros(len(lams), p, device=dev)
    R = lams[:, None] * pen[None] + 1e-4
    for _ in range(iters):
        pr = torch.sigmoid(X @ B.T)  # (n,L)
        g = ((pr - yt[:, None]) * sw[:, None]).T @ X + R * B
        Hw = pr * (1 - pr) * sw[:, None]
        H = torch.einsum("nl,np,nq->lpq", Hw, X, X) + torch.diag_embed(R)
        B = B - torch.linalg.solve(H, g[..., None])[..., 0]
    return B


def auc_cols(S, y):
    """S (n,L)，y bool (n,) → (L,) AUC。"""
    r = S.argsort(0).argsort(0).float() + 1
    npos, nneg = y.sum().float(), (~y).sum().float()
    return (r[y].sum(0) - npos * (npos + 1) / 2) / (npos * nneg)


def zc(s, cal):
    return (s - s[cal].mean()) / (s[cal].std() + 1e-6)


def build_set(name, D, subs):
    out = {}
    for q in subs:
        X, blk = D[q][0], D[q][2]
        cal = torch.tensor(blk == 3 if q == S034 else blk <= 3, device=dev)
        if name.startswith("rc"):
            m, _ = L.run_stats_t(X, int(name[2:]))
            X = X - m
        elif name == "cat":
            X = torch.cat([L.feats({q: D[q]}, [q], w)[q] for w in CWS], 1)
        X = ((X - X[cal].mean(0)) / (X[cal].std(0) + 1e-6)).clamp(-10, 10)
        out[q] = X
    return out


def run_set(name, D, subs):
    F = build_set(name, D, subs)
    reg = [q for q in subs if q != S034]
    # 逐人 LR（全 6 块，单位化）→ 用于 w0 / U
    Wj = {}
    for q in reg:
        y = D[q][1]
        f = lbfgs_lr(F[q], y, bal_w(y), torch.tensor([LAM_WJ], device=dev))
        w = f.W[0]
        Wj[q] = w / w.norm()
    Wall = torch.stack([Wj[q] for q in reg])
    # pool LR：其余人全部帧，一次拼接，用 mask 排除本人
    Xall = torch.cat([F[q] for q in subs])
    yall = np.concatenate([D[q][1] for q in subs])
    sid = np.concatenate([[i] * len(D[q][1]) for i, q in enumerate(subs)])
    swj = torch.cat([bal_w(D[q][1]) for q in subs])
    sidt = torch.tensor(sid, device=dev)
    cfgs = ["pool", "k0"] + [f"k{k}" for k in KS] + ["kcv", "res", "own", "kcv+pool", "kcv+own", "res+pool"]
    EV, LOG = {c: [] for c in cfgs}, {}
    for i, q in enumerate(subs):
        y, blk = D[q][1], D[q][2]
        Z = F[q]
        cal = torch.tensor(blk == 3 if q == S034 else blk <= 3, device=dev)
        tst = torch.tensor(np.where(blk >= 4)[0], device=dev)
        yy = torch.tensor(y[blk >= 4], device=dev).bool()
        # pool
        sw = swj * (sidt != i).float() / (len(subs) - 1)
        fp = lbfgs_lr(Xall, yall, sw, torch.tensor([LAM_POOL], device=dev), iters=150)
        S = {"pool": fp(Z)[:, 0]}
        # w0 / U
        Wo = Wall[[k for k, o in enumerate(reg) if o != q]]
        w0 = Wo.mean(0)
        U = torch.linalg.svd(Wo - w0, full_matrices=False).Vh[:max(KS)].T  # (d,5)
        s0 = Z @ w0
        S["k0"] = s0
        info = {}
        if q == S034:
            for c in [f"k{k}" for k in KS] + ["kcv", "res", "own"]:
                S[c] = s0
        else:
            s0n = zc(s0, cal)
            P = Z @ U
            P = (P - P[cal].mean(0)) / (P[cal].std(0) + 1e-6)
            Xd = torch.cat([torch.ones(len(y), 1, device=dev), s0n[:, None], P], 1)
            yb = torch.tensor(y, device=dev).bool()
            bt = torch.tensor(blk, device=dev)
            cvk = {}
            for k in KS:
                pen = torch.tensor([0, 0] + [1] * k, device=dev).float()
                Xk = Xd[:, :2 + k]
                A = []
                for held, trn in FOLDS:
                    mt = (bt == trn[0]) | (bt == trn[1])
                    mh = (bt == held[0]) | (bt == held[1])
                    B = newton_lr(Xk[mt], y[mt.cpu().numpy()], bal_w(y[mt.cpu().numpy()]), LAM_K, pen)
                    A.append(auc_cols(Xk[mh] @ B.T, yb[mh]))
                A = torch.stack(A).mean(0)
                li = int(A.argmax())
                B = newton_lr(Xk[cal], y[cal.cpu().numpy()], bal_w(y[cal.cpu().numpy()]), LAM_K[li:li + 1], pen)
                S[f"k{k}"] = Xk @ B[0]
                cvk[k] = (float(A[li]), float(LAM_K[li]))
            kb = max(cvk, key=lambda k: cvk[k][0])
            S["kcv"] = S[f"k{kb}"]
            info.update(cv={k: v for k, v in cvk.items()}, kcv=kb)
            # 全维：res = a·s0 + δ（δ 收缩向 0 ⇒ w 收缩向 w0 方向）；own = 仅 δ
            for nm, off in (("res", s0n), ("own", None)):
                A = []
                for held, trn in FOLDS:
                    mt = (bt == trn[0]) | (bt == trn[1])
                    mh = (bt == held[0]) | (bt == held[1])
                    yt_ = y[mt.cpu().numpy()]
                    f = lbfgs_lr(Z[mt], yt_, bal_w(yt_), LAM_D, None if off is None else off[mt])
                    A.append(auc_cols(f(Z[mh], None if off is None else off[mh]), yb[mh]))
                A = torch.stack(A).mean(0)
                li = int(A.argmax())
                yc = y[cal.cpu().numpy()]
                f = lbfgs_lr(Z[cal], yc, bal_w(yc), LAM_D[li:li + 1], None if off is None else off[cal])
                S[nm] = f(Z, off)[:, 0]
                info[nm] = (float(A[li]), float(LAM_D[li]))
        for a_, b_ in (("kcv", "pool"), ("kcv", "own"), ("res", "pool")):
            S[f"{a_}+{b_}"] = zc(S[a_], cal) + zc(S[b_], cal)
        M = torch.stack([zc(S[c], cal) for c in cfgs])
        R = E.metrics(M, tst, yy).cpu().numpy()  # (ncfg,85,3)
        for c, r in zip(cfgs, R):
            EV[c].append(r)
        LOG[q] = info
        print(name, q, " ".join(f"{c}={np.nanmax(R[ci, :, 0]):.2f}" for ci, c in enumerate(cfgs[:9])),
              info.get("kcv", ""), flush=True)
        torch.cuda.empty_cache()
    del F, Xall
    torch.cuda.empty_cache()
    return {c: np.stack(v) for c, v in EV.items()}, LOG


def loso_thr(V):
    """V (nq,85,3) → 每人按其余人平均 bACC 选阈值后的 (nq,3) 与所选阈值索引。"""
    nq = V.shape[0]
    tot = np.nansum(V[:, :, 0], 0)
    cnt = (~np.isnan(V[:, :, 0])).sum(0)
    out, sel = np.zeros((nq, 3)), []
    for i in range(nq):
        m = (tot - np.nan_to_num(V[i, :, 0])) / np.maximum(cnt - (~np.isnan(V[i, :, 0])), 1)
        t = int(np.nanargmax(m))
        sel.append(t)
        out[i] = V[i, t]
    return out, sel


def summ(a):
    return dict(bacc=float(a[:, 0].mean()), auc=float(a[:, 1].mean()), f1=float(a[:, 2].mean()),
                n90=int((a[:, 0] >= .9).sum()), min=float(a[:, 0].min()))


def main():
    subs, D = L.load()
    sets = sys.argv[1:] or ["raw", "rc300", "rc600", "cat"]
    res = json.load(open(HERE / "e27.json")) if (HERE / "e27.json").exists() else dict(table={}, per_subject={})
    allEV = {}
    for s in sets:
        EV, LOG = run_set(s, D, subs)
        np.savez_compressed(HERE / f"ev_{s}.npz", **{c.replace("+", "_"): v for c, v in EV.items()})
        for c, V in EV.items():
            a, sel = loso_thr(V)
            fx = np.nanmean(V[:, :, 0], 0)
            res["table"][f"{s}|{c}"] = dict(loso=summ(a), thr_mode=E.THR2[int(np.bincount(sel).argmax())][0],
                                            best_fixed_thr=E.THR2[int(fx.argmax())][0], best_fixed_bacc=float(fx.max()))
            res["per_subject"][f"{s}|{c}"] = {q: [float(x) for x in a[i]] for i, q in enumerate(subs)}
            allEV[f"{s}|{c}"] = V
            t = res["table"][f"{s}|{c}"]["loso"]
            print(f"{s:6s} {c:10s} bACC {t['bacc']:.3f} AUC {t['auc']:.3f} F1 {t['f1']:.3f} n90 {t['n90']:2d} "
                  f"min {t['min']:.3f} thr {res['table'][f'{s}|{c}']['thr_mode']}", flush=True)
        res.setdefault("cvlog", {})[s] = LOG
        json.dump(res, open(HERE / "e27.json", "w"), indent=1)
    # 嵌套 LOSO 选（配置, 阈值）：其余人平均 bACC 最大
    keys = [k for k in res["table"]]
    EVs = {}
    for k in keys:
        s, c = k.split("|")
        EVs[k] = allEV[k] if k in allEV else np.load(HERE / f"ev_{s}.npz")[c.replace("+", "_")]
    if all(v is not None for v in EVs.values()):
        A = np.stack([EVs[k] for k in keys])  # (nc,nq,85,3)
        nq = A.shape[1]
        out, pick = np.zeros((nq, 3)), []
        for i in range(nq):
            o = np.delete(A[:, :, :, 0], i, 1)
            m = np.nanmean(o, 1)
            c, t = np.unravel_index(np.nanargmax(m), m.shape)
            out[i] = A[c, i, t]
            pick.append(f"{keys[c]}@{E.THR2[t][0]}")
        res["nested_select"] = dict(summ=summ(out), picks=dict(zip(subs, pick)))
        print("nested select", summ(out), set(pick))
        json.dump(res, open(HERE / "e27.json", "w"), indent=1)


if __name__ == "__main__":
    main()
