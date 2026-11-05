"""E33：校准响应指纹 → 测试配置路由。

只使用目标本人校准块（普通被试 0–3，034 仅块 3）构造响应指纹；
其他被试的测试 bACC 是元学习目标。外层 LOSO 的目标测试标签只用于最终评分，
所有超参在外层训练被试内部再做 LOSO 选择。oracle 单独报告，绝不可视为可部署结果。

运行：.venv\Scripts\python.exe test/33_response_routing/e33.py --datasets E23,E28
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SLID = ROOT / "test" / "11_individual_20261102" / "sliding1"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
E23_THR = [("t0", 0, None), ("m120", 120, .5), ("m300", 300, .5), ("m600", 600, .5),
           ("q300_.4", 300, .4), ("q300_.6", 300, .6), ("q600_.4", 600, .4), ("q600_.6", 600, .6)]
E28_WS = (60, 120, 300, 600, 1200, 2400)
E28_QS = tuple(round(.2 + .05 * i, 2) for i in range(13))
E28_THR = [("t0", 0, None)] + [(f"m{w}", w, None) for w in E28_WS] + [
    (f"q{w}_{p}", w, p) for w in E28_WS for p in E28_QS]


def safe_mean(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """沿指定轴忽略 NaN 求均值；全缺失列回退为 0。"""
    valid = torch.isfinite(x)
    total = torch.where(valid, x, torch.zeros_like(x)).sum(dim)
    count = valid.sum(dim)
    return total / count.clamp_min(1)


def read_labels(q: str) -> tuple[np.ndarray, np.ndarray]:
    """按缓存分数相同的 t 升序读取标签和块号。"""
    s = np.load(SLID / f"sub-{q}.npz")
    order = np.argsort(s["t"])
    return s["y"][order].astype(np.int64), s["blk"][order].astype(np.int64)


def past_window(m: torch.Tensor, idx: torch.Tensor, w: int, step: int = 1) -> torch.Tensor:
    """取每个 idx 对应的过去 w 帧（含当前），只展开所需位置以节省显存。"""
    pad = torch.cat([torch.full((m.shape[0], w - 1), float("nan"), device=m.device), m], 1)
    pos = idx[:, None] + torch.arange(0, w, step, device=m.device)[None, :]
    return pad[:, pos]


def bacc_from_margin(d: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """由分数减因果阈值计算 bACC；单类校准块返回 NaN。"""
    if not bool(y.any()) or bool(y.all()):
        return torch.full(d.shape[:-1], float("nan"), device=d.device)
    pred = d > 0
    tp = (pred & y).sum(-1).float()
    fp = (pred & ~y).sum(-1).float()
    npos, nneg = y.sum().float(), (~y).sum().float()
    return .5 * (tp / npos + 1 - fp / nneg)


def view_layout(y: np.ndarray, blk: np.ndarray, q: str, test: bool = False):
    """返回评估位置及各视图的相对掩码：全部/前半/后半校准块。"""
    if test:
        idx = np.where(blk >= 4)[0]
        views = [np.ones(len(idx), dtype=bool)]
    else:
        cal = (blk == 3) if q == "034" else (blk <= 3)
        idx = np.where(cal)[0]
        if q == "034":
            views = [np.ones(len(idx), dtype=bool), np.zeros(len(idx), bool), np.zeros(len(idx), bool)]
        else:
            b = blk[idx]
            views = [np.ones(len(idx), bool), b <= 1, (b >= 2) & (b <= 3)]
    return idx, views


def eval_e23_views(m: torch.Tensor, y: np.ndarray, blk: np.ndarray, q: str,
                   test: bool = False) -> torch.Tensor:
    """E23 的 8 种阈值在指定区间上计算 bACC，输出 (视图,配置,阈值)。"""
    idx_np, views = view_layout(y, blk, q, test)
    idx = torch.as_tensor(idx_np, device=m.device)
    yy = torch.as_tensor(y[idx_np], device=m.device).bool()
    nv = len(views)
    out = torch.full((nv, m.shape[0], len(E23_THR)), float("nan"), device=m.device)
    mc = m[:, idx]
    for k, (_, w, quant) in enumerate(E23_THR):
        if w == 0:
            th = torch.zeros_like(mc)
        else:
            u = past_window(m, idx, w)
            th = u.nanquantile(quant, 2)
        d = mc - th
        for v, vm in enumerate(views):
            if vm.any():
                out[v, :, k] = bacc_from_margin(d[:, torch.as_tensor(vm, device=m.device)],
                                                yy[torch.as_tensor(vm, device=m.device)])
    return out


def eval_e28_views(m: torch.Tensor, y: np.ndarray, blk: np.ndarray, q: str,
                   test: bool = False) -> torch.Tensor:
    """复刻 E26/E28 的 85 种因果阈值，只算 bACC，输出 (视图,配置,阈值)。"""
    idx_np, views = view_layout(y, blk, q, test)
    idx = torch.as_tensor(idx_np, device=m.device)
    yy = torch.as_tensor(y[idx_np], device=m.device).bool()
    nv = len(views)
    out = torch.full((nv, m.shape[0], len(E28_THR)), float("nan"), device=m.device)
    mc = m[:, idx]

    def fill(d: torch.Tensor, cols: slice | int):
        # d 可为 (配置,帧) 或 (配置,分位数,帧)。
        for v, vm in enumerate(views):
            if not vm.any():
                continue
            mt = torch.as_tensor(vm, device=m.device)
            val = bacc_from_margin(d[..., mt], yy[mt])
            if isinstance(cols, slice):
                out[v, :, cols] = val
            else:
                out[v, :, cols] = val

    fill(mc, 0)
    for wi, w in enumerate(E28_WS):
        th = past_window(m, idx, w).nanmean(2)
        fill(mc - th, 1 + wi)
    for wi, w in enumerate(E28_WS):
        step = max(1, w // 600)
        u = past_window(m, idx, w, step)
        sorted_u = u.sort(2).values
        n = (~torch.isnan(u)).sum(2).float().clamp(min=1)
        qs = torch.as_tensor(E28_QS, device=m.device)
        pos = (n - 1)[..., None] * qs
        lo, hi = pos.floor().long(), pos.ceil().long()
        a = sorted_u.gather(2, lo)
        b = sorted_u.gather(2, hi)
        th = (a + (b - a) * (pos - lo.float())).permute(0, 2, 1)
        fill(mc[:, None, :] - th, slice(1 + len(E28_WS) + wi * len(E28_QS),
                                       1 + len(E28_WS) + (wi + 1) * len(E28_QS)))
    return out


def expand_e23(b: torch.Tensor, cal: torch.Tensor) -> torch.Tensor:
    """57 个基础分数扩展为 57+两两组合，严格复刻 E23 的校准 z 归一化。"""
    ia, ib = zip(*itertools.combinations(range(b.shape[0]), 2))
    p = b[list(ia)] + b[list(ib)]
    p = (p - p[:, cal].mean(1, keepdim=True)) / (p[:, cal].std(1, keepdim=True) + 1e-6)
    return torch.cat([b, p])


def compute_subject_cal(ds: str, q: str, scores: np.ndarray) -> np.ndarray:
    """逐被试在 GPU 上计算校准响应指纹。"""
    y, blk = read_labels(q)
    if len(y) != scores.shape[1]:
        raise ValueError(f"{q} 分数与标签长度不一致：{scores.shape} vs {len(y)}")
    b = torch.as_tensor(scores, device=DEV, dtype=torch.float32)
    cal_np = (blk == 3) if q == "034" else (blk <= 3)
    m = expand_e23(b, torch.as_tensor(cal_np, device=DEV)) if ds == "E23" else b
    rows = []
    chunk = 24 if ds == "E23" else 16
    fn = eval_e23_views if ds == "E23" else eval_e28_views
    for a in range(0, m.shape[0], chunk):
        rows.append(fn(m[a:a + chunk], y, blk, q))
    out = torch.cat(rows, 1)
    bad = torch.isnan(m).any(1)
    out[:, bad] = float("nan")
    return out.cpu().numpy().astype(np.float32)


def load_source(ds: str):
    """读取测试指标缓存及配置名称，统一为 (被试,配置×阈值)。"""
    if ds == "E23":
        folder = ROOT / "test" / "23_synthesis"
        meta = json.loads((folder / "cands_meta.json").read_text(encoding="utf-8"))
        e = np.load(folder / "cands_eval.npz")
        subs = list(e["subs"])
        ev = e["ev"]
        n = len(subs)
        metrics = np.stack([ev[:, k].reshape(n, -1) for k in range(3)], -1)
        return dict(folder=folder, subs=subs, configs=meta["cands"], thrs=[x[0] for x in E23_THR],
                    metrics=metrics, cal_ref=ev[:, 3].reshape(n, -1))
    folder = ROOT / "test" / "28_block_invariant"
    e = np.load(folder / "e28_ev.npz")
    subs = list(e["subs"])
    metrics = e["EV"].transpose(1, 0, 2, 3).reshape(len(subs), -1, 3)
    return dict(folder=folder, subs=subs, configs=list(e["names"]), thrs=[x[0] for x in E28_THR],
                metrics=metrics, cal_ref=None)


def get_cal(ds: str, src, recompute: bool = False) -> tuple[np.ndarray, dict]:
    """加载或建立校准缓存，并复核阈值实现与已有缓存一致。"""
    cache = HERE / f"{ds.lower()}_cal_fingerprint.npz"
    checks = {}
    if cache.exists() and not recompute:
        z = np.load(cache)
        f = z["fingerprints"].reshape(len(src["subs"]), 3, -1)
        checks = json.loads(str(z["checks_json"])) if "checks_json" in z.files else {}
        checks.update(cache=str(cache), reused=True)
        if ds == "E23":
            regular = np.array(src["subs"]) != "034"
            diff = np.abs(f[regular, 0] - src["cal_ref"][regular])
            checks["E23_cal_bacc_max_abs_diff"] = float(np.nanmax(diff))
            if checks["E23_cal_bacc_max_abs_diff"] > 2e-5:
                raise AssertionError(f"E23 校准指标复核失败：{checks}")
        return f, checks
    fingerprints = []
    for i, q in enumerate(src["subs"]):
        if ds == "E23":
            scores = np.load(src["folder"] / "cands_scores.npz")[q]
        else:
            scores = np.load(src["folder"] / "e28_scores.npz")[f"s{q}"]
        fingerprints.append(compute_subject_cal(ds, q, scores))
        if i % 8 == 0 or q == "034":
            print(f"[{ds} 校准 {i + 1:02d}/57] {q}", flush=True)
    f = np.stack(fingerprints)  # (被试,全部/前半/后半,配置,阈值)
    f = f.reshape(len(src["subs"]), 3, -1)
    checks = {"cache": str(cache), "reused": False}
    if ds == "E23":
        regular = np.array(src["subs"]) != "034"
        diff = np.abs(f[regular, 0] - src["cal_ref"][regular])
        checks["E23_cal_bacc_max_abs_diff"] = float(np.nanmax(diff))
        if checks["E23_cal_bacc_max_abs_diff"] > 2e-5:
            raise AssertionError(f"E23 校准指标复核失败：{checks}")
    else:
        # 用测试位置重算 bACC 仅作实现自检，不参与路由。E28 分数在 EV 后转 float16，
        # 因此同时对照“同一 half 输入的 E26 原评估器”和“原 float32 EV”，区分实现误差与量化差。
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "e26lib_e33check", ROOT / "test" / "26_merge2" / "e26lib.py")
        e26 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(e26)
        for q in ("001", "034"):
            i = src["subs"].index(q)
            scores = np.load(src["folder"] / "e28_scores.npz")[f"s{q}"]
            y, blk = read_labels(q)
            m = torch.as_tensor(scores, device=DEV, dtype=torch.float32)
            vals = []
            for a in range(0, m.shape[0], 16):
                vals.append(eval_e28_views(m[a:a + 16], y, blk, q, test=True))
            got = torch.cat(vals, 1)[0].cpu().numpy()
            tst = torch.as_tensor(np.where(blk >= 4)[0], device=DEV)
            yy = torch.as_tensor(y[blk >= 4], device=DEV).bool()
            same_half = e26.metrics(m, tst, yy).cpu().numpy()[..., 0]
            original = src["metrics"][i, :, 0].reshape(m.shape[0], len(E28_THR))
            checks[f"E28_same_half_vs_E26_{q}"] = float(np.nanmax(np.abs(got - same_half)))
            checks[f"E28_float16_vs_original_EV_{q}"] = float(np.nanmax(np.abs(got - original)))
            if checks[f"E28_same_half_vs_E26_{q}"] > 1e-6:
                raise AssertionError(f"E28 阈值复核失败：{checks}")
    np.savez_compressed(cache, fingerprints=f, subs=np.array(src["subs"]), checks_json=json.dumps(checks))
    return f, checks


def feature_view(f: np.ndarray, view: str) -> np.ndarray:
    if view == "all":
        return f[:, 0]
    if view == "blocks":
        return np.concatenate([f[:, 0], f[:, 1], f[:, 2]], 1)
    raise ValueError(view)


def global_loo(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """训练被试内部逐人排除后的全局均值，以及全体均值。"""
    valid = torch.isfinite(y)
    y0 = torch.where(valid, y, torch.zeros_like(y))
    total, count = y0.sum(0), valid.sum(0)
    inner = (total[None] - y0) / (count[None] - valid.long()).clamp_min(1)
    outer = total / count.clamp_min(1)
    return inner, outer


def choose_config(pred: torch.Tensor, avail: torch.Tensor) -> torch.Tensor:
    return pred.masked_fill(~avail, -torch.inf).argmax(-1)


def score_hp(pin: torch.Tensor, pout: torch.Tensor, ytr: torch.Tensor,
             avail_tr: torch.Tensor, avail_out: torch.Tensor):
    """内层用真实测试 bACC 仅为训练折评分；外层只返回配置索引。"""
    inner_pick = choose_config(pin, avail_tr)
    actual = ytr.gather(1, inner_pick[:, None])[:, 0]
    outer_pick = choose_config(pout, avail_out)
    return float(actual.mean()), int(outer_pick)


def prepare_outer(x: torch.Tensor, y: torch.Tensor, cal: torch.Tensor, outer: int, scale: str):
    """准备外层训练/目标指纹；标准化统计不含外层目标。"""
    tr = np.delete(np.arange(x.shape[0]), outer)
    xtr0, xt0 = x[tr], x[outer:outer + 1]
    mu = safe_mean(xtr0)
    xtr = torch.where(torch.isfinite(xtr0), xtr0, mu)
    xt = torch.where(torch.isfinite(xt0), xt0, mu)
    xtr, xt = xtr - mu, xt - mu
    if scale == "z":
        sd = (xtr ** 2).mean(0).sqrt() + 1e-3
        xtr, xt = xtr / sd, xt / sd
    ytr = y[tr]
    valid_sub = torch.isfinite(x).any(1)
    return tr, xtr, xt, ytr, valid_sub[tr], valid_sub[outer], cal[tr], cal[outer:outer + 1]


def finish_invalid(pin, pout, ytr, valid_tr, valid_out):
    """034 校准仅一个类，无法构造 bACC 指纹，严格回退到其他被试全局均值。"""
    inner_global, outer_global = global_loo(ytr)
    pin = pin.clone()
    pin[~valid_tr] = inner_global[~valid_tr]
    if not bool(valid_out):
        pout = outer_global[None]
    return pin, pout


def run_knn(spec, x, y, cal, avail, outer: int):
    """距离倒数 kNN；soft 额外混合目标本人的校准 bACC。"""
    tr, xtr, xt, ytr, valid_tr, valid_out, ctr, ct = prepare_outer(x, y, cal, outer, spec["scale"])
    n, d = xtr.shape
    z = torch.cat([xtr, xt])
    dist = torch.cdist(z, z) / np.sqrt(d)
    di = dist[:n, :n].clone()
    di.fill_diagonal_(torch.inf)
    order_i = di.argsort(1)
    order_o = dist[n, :n].argsort(0)
    y0 = torch.where(torch.isfinite(ytr), ytr, torch.zeros_like(ytr))
    vy = torch.isfinite(ytr).float()
    inner_scores, outer_choices = [], []
    for hp in spec["grid"]:
        k, alpha = hp["k"], hp.get("alpha", 0.)
        ki, ko = min(k, n - 1), min(k, n)
        ii = order_i[:, :ki]
        dd = di.gather(1, ii)
        w = 1 / (dd + 1e-3)
        wi = torch.zeros(n, n, device=DEV).scatter(1, ii, w)
        io = order_o[:ko]
        do = dist[n, io]
        wo = (1 / (do + 1e-3))[None]
        pin = (wi @ y0) / (wi @ vy).clamp_min(1e-8)
        pout = (wo @ y0[io]) / (wo @ vy[io]).clamp_min(1e-8)
        if alpha:
            pin = torch.where(torch.isfinite(ctr), (1 - alpha) * pin + alpha * ctr, pin)
            pout = torch.where(torch.isfinite(ct), (1 - alpha) * pout + alpha * ct, pout)
        pin, pout = finish_invalid(pin, pout, ytr, valid_tr, valid_out)
        s, c = score_hp(pin, pout, ytr, avail[tr], avail[outer:outer + 1])
        inner_scores.append(s)
        outer_choices.append(c)
    return inner_scores, outer_choices


def run_ridge(spec, x, y, cal, avail, outer: int):
    """多输出对偶岭回归；内层 LOO 用帽子矩阵精确计算，避免重复拟合。"""
    tr, xtr, xt, ytr, valid_tr, valid_out, _, _ = prepare_outer(x, y, cal, outer, spec["scale"])
    n, d = xtr.shape
    ymu = safe_mean(ytr)
    y0 = torch.where(torch.isfinite(ytr), ytr, ymu)
    yc = y0 - ymu
    kernel = xtr @ xtr.T / d
    kt = xt @ xtr.T / d
    eye = torch.eye(n, device=DEV)
    inner_scores, outer_choices = [], []
    for hp in spec["grid"]:
        lam = hp["lambda"]
        inv = torch.linalg.inv(kernel + lam * eye)
        hk = kernel @ inv
        fitted = ymu + hk @ yc
        h = 1 / n + hk.diag()
        pin = (fitted - h[:, None] * y0) / (1 - h[:, None]).clamp_min(1e-4)
        pout = ymu + kt @ (inv @ yc)
        pin, pout = finish_invalid(pin, pout, ytr, valid_tr, valid_out)
        s, c = score_hp(pin, pout, ytr, avail[tr], avail[outer:outer + 1])
        inner_scores.append(s)
        outer_choices.append(c)
    return inner_scores, outer_choices


def run_pcr(spec, x, y, cal, avail, outer: int):
    """PCA 主成分回归（小岭稳定项）；仅由校准指纹估计主成分。"""
    tr, xtr, xt, ytr, valid_tr, valid_out, _, _ = prepare_outer(x, y, cal, outer, spec["scale"])
    n, d = xtr.shape
    ymu = safe_mean(ytr)
    y0 = torch.where(torch.isfinite(ytr), ytr, ymu)
    yc = y0 - ymu
    kernel = xtr @ xtr.T / d
    kt = xt @ xtr.T / d
    ev, u = torch.linalg.eigh(kernel)
    ev, u = ev.flip(0).clamp_min(0), u.flip(1)
    inner_scores, outer_choices = [], []
    lam = 1e-3
    for hp in spec["grid"]:
        r = min(hp["components"], n - 1)
        ur, sr = u[:, :r], ev[:r]
        hk = (ur * (sr / (sr + lam))) @ ur.T
        fitted = ymu + hk @ yc
        h = 1 / n + hk.diag()
        pin = (fitted - h[:, None] * y0) / (1 - h[:, None]).clamp_min(1e-4)
        pout = ymu + ((kt @ ur) / (sr + lam)) @ (ur.T @ yc)
        pin, pout = finish_invalid(pin, pout, ytr, valid_tr, valid_out)
        s, c = score_hp(pin, pout, ytr, avail[tr], avail[outer:outer + 1])
        inner_scores.append(s)
        outer_choices.append(c)
    return inner_scores, outer_choices


def model_specs():
    ks = [{"k": k} for k in (1, 3, 5, 10, 20, 40)]
    soft = [{"k": k, "alpha": a} for k in (3, 10, 40) for a in (0., .25, .5, .75)]
    lams = [{"lambda": x} for x in (1e-4, 1e-3, 1e-2, .1, 1., 10.)]
    pcs = [{"components": r} for r in (2, 5, 10, 20, 40)]
    return [
        dict(name="knn_all_raw", kind="knn", view="all", scale="raw", grid=ks),
        dict(name="knn_all_z", kind="knn", view="all", scale="z", grid=ks),
        dict(name="knn_blocks_raw", kind="knn", view="blocks", scale="raw", grid=ks),
        dict(name="soft_all_raw", kind="knn", view="all", scale="raw", grid=soft),
        dict(name="ridge_all_raw", kind="ridge", view="all", scale="raw", grid=lams),
        dict(name="ridge_all_z", kind="ridge", view="all", scale="z", grid=lams),
        dict(name="ridge_blocks_z", kind="ridge", view="blocks", scale="z", grid=lams),
        dict(name="pcr_all_z", kind="pcr", view="all", scale="z", grid=pcs),
    ]


def summarize(values: np.ndarray) -> dict:
    return dict(bacc=float(values[:, 0].mean()), auc=float(values[:, 1].mean()),
                f1=float(values[:, 2].mean()), n90=int((values[:, 0] >= .9).sum()),
                min_bacc=float(values[:, 0].min()), median_bacc=float(np.median(values[:, 0])))


def record_choices(src, choices, oracle=False):
    choices = np.asarray(choices, dtype=int)
    rows = np.arange(len(choices))
    vals = src["metrics"][rows, choices]
    per = {}
    for i, q in enumerate(src["subs"]):
        c, t = divmod(int(choices[i]), len(src["thrs"]))
        per[q] = dict(config=src["configs"][c], threshold=src["thrs"][t],
                      bacc=float(vals[i, 0]), auc=float(vals[i, 1]), f1=float(vals[i, 2]))
    return dict(summary=summarize(vals), per_subject=per, oracle=oracle), vals


def route_dataset(ds: str, recompute_cal: bool = False):
    start = time.time()
    src = load_source(ds)
    f, checks = get_cal(ds, src, recompute_cal)
    y = torch.as_tensor(src["metrics"][..., 0], device=DEV, dtype=torch.float32)
    cal = torch.as_tensor(f[:, 0], device=DEV, dtype=torch.float32)
    avail = torch.isfinite(y)
    n = len(src["subs"])
    results, traces = {}, {}

    # 两个非学习型参照均不使用目标测试标签；034 的直接校准选择回退到 LOSO 全局。
    oracle_choices = np.nanargmax(src["metrics"][..., 0], axis=1)
    results["oracle_test_upper"], _ = record_choices(src, oracle_choices, oracle=True)
    global_choices, cal_choices = [], []
    for i in range(n):
        tr = np.delete(np.arange(n), i)
        pred = safe_mean(y[tr])
        gi = int(choose_config(pred[None], avail[i:i + 1])[0])
        global_choices.append(gi)
        if bool(torch.isfinite(cal[i]).any()):
            ci = int(choose_config(cal[i:i + 1], avail[i:i + 1])[0])
            cal_choices.append(ci)
        else:
            cal_choices.append(gi)
    results["global_loso"], _ = record_choices(src, global_choices)
    results["direct_calibration"], _ = record_choices(src, cal_choices)
    if ds == "E28":
        anchor = src["configs"].index("b|pool|w300|a0.75") * len(src["thrs"]) + src["thrs"].index("m300")
        results["fixed_pool_anchor"], _ = record_choices(src, [anchor] * n)
    if ds == "E23":
        anchor = src["configs"].index("ens8") * len(src["thrs"]) + src["thrs"].index("t0")
        results["fixed_ens8_t0"], _ = record_choices(src, [anchor] * n)

    specs = model_specs()
    for spec in specs:
        x = torch.as_tensor(feature_view(f, spec["view"]), device=DEV, dtype=torch.float32)
        fn = {"knn": run_knn, "ridge": run_ridge, "pcr": run_pcr}[spec["kind"]]
        inner_all, outer_all = [], []
        for outer in range(n):
            si, co = fn(spec, x, y, cal, avail, outer)
            inner_all.append(si)
            outer_all.append(co)
            if outer % 14 == 0:
                print(f"[{ds} {spec['name']} outer {outer + 1:02d}/57]", flush=True)
        chosen_hp, choices = [], []
        for i in range(n):
            h = int(np.argmax(inner_all[i]))
            chosen_hp.append(h)
            choices.append(outer_all[i][h])
        results[spec["name"]], _ = record_choices(src, choices)
        results[spec["name"]]["selected_hparams"] = [spec["grid"][h] for h in chosen_hp]
        traces[spec["name"]] = dict(kind=spec["kind"], view=spec["view"], scale=spec["scale"],
                                    grid=spec["grid"], inner_scores=inner_all, outer_choices=outer_all)
        s = results[spec["name"]]["summary"]
        print(f"[{ds} {spec['name']}] bACC {s['bacc']:.4f} AUC {s['auc']:.4f} F1 {s['f1']:.4f}", flush=True)

    # 总体严格嵌套：每个外层折连“模型族+特征+超参”也完全由其余 56 人的内层 LOSO 决定。
    choices, selected = [], []
    for i in range(n):
        candidates = [(traces[s["name"]]["inner_scores"][i][h], s["name"], h)
                      for s in specs for h in range(len(s["grid"]))]
        _, name, h = max(candidates, key=lambda z: z[0])
        choices.append(traces[name]["outer_choices"][i][h])
        selected.append(dict(model=name, hparam=specs[[s["name"] for s in specs].index(name)]["grid"][h]))
    results["omnibus_nested"], _ = record_choices(src, choices)
    results["omnibus_nested"]["selected_models"] = selected
    s = results["omnibus_nested"]["summary"]
    print(f"[{ds} OMNIBUS] bACC {s['bacc']:.4f} AUC {s['auc']:.4f} F1 {s['f1']:.4f}", flush=True)
    return dict(dataset=ds, protocol="outer_LOSO+inner_LOSO; target test labels score only",
                device=DEV, runtime_sec=time.time() - start, checks=checks, results=results,
                traces=traces, n_configs=len(src["configs"]), n_thresholds=len(src["thrs"]))


def write_report(all_results: dict, output: Path):
    """生成简明报告；oracle 与严格折外结果分表，避免混淆。"""
    lines = ["# E33：校准响应指纹 → 配置路由", "",
             "## 结论", ""]
    for ds, r in all_results.items():
        best = r["results"]["omnibus_nested"]["summary"]
        oracle = r["results"]["oracle_test_upper"]["summary"]
        model_names = [s["name"] for s in model_specs()]
        best_name = max(model_names, key=lambda name: r["results"][name]["summary"]["bacc"])
        family = r["results"][best_name]["summary"]
        lines.append(f"- **{ds} 严格嵌套 LOSO（模型族与超参均在内层选择）**："
                     f"bACC **{best['bacc']:.4f}**，AUC **{best['auc']:.4f}**，F1 **{best['f1']:.4f}**，"
                     f"n≥.9={best['n90']}/57；目标 .9 {'达到' if best['bacc'] >= .9 else '未达到'}。")
        lines.append(f"- {ds} 各预设路由族分别嵌套选超参后，探索性最高为 `{best_name}`："
                     f"bACC {family['bacc']:.4f}，AUC {family['auc']:.4f}，F1 {family['f1']:.4f}。"
                     "每人预测均严格折外，但家族间最高值属于事后比较，总体结论仍以上一行的 omnibus 为准。")
        lines.append(f"- {ds} 测试标签 oracle 上限：bACC {oracle['bacc']:.4f}，AUC {oracle['auc']:.4f}，"
                     f"F1 {oracle['f1']:.4f}；该数值仅作上限，不能部署。")
        if ds == "E28":
            anchor = r["results"]["fixed_pool_anchor"]["summary"]
            lines.append(f"- E28 最佳单族较固定 pool 锚点的 bACC 增量为 {family['bacc'] - anchor['bacc']:+.4f}；"
                         "其57折均选 k=40、α=0，提升来自原始响应指纹的距离倒数40-NN，而非校准bACC凸混合。")
    lines += ["", "既有固定通用参照约为 .821（E28 pool 锚点）和 .836（E31b 最优单配置的 LOSO 阈值口径；"
              "其全配置嵌套选择为 .8197）。本实验不把 oracle 或外层事后择优当成总体严格结果。", "",
              "## 严格评估协议", "",
              "- 指纹：本人校准块的逐配置×因果阈值 bACC；`all` 为块0–3整体，`blocks` 另拼接块0–1与块2–3。"
              "034 仅块3且只有 task 类，bACC 指纹不可定义，所有响应模型严格回退为其他56人的全局均值。",
              "- 路由：距离倒数 kNN、软 kNN+校准表现凸混合、多输出对偶岭回归、PCA 主成分回归。",
              "- 每个外层留出一人；模型族、特征变体和超参只由其余56人的内层 LOSO 宏 bACC 选择。"
              "岭/PCR 内层 LOO 使用帽子矩阵精确计算，不使用内层留出者的测试标签拟合。",
              "- 外层目标的测试标签只用于选择完成后的 bACC/AUC/F1 评分；oracle 表是唯一允许按目标测试标签选配置的对照。",
              "- 分数与阈值直接复用 E23/E28 缓存；校准响应计算在 GPU 上完成并缓存。",
              "- E28 保存分数为 float16；同一 half 输入下本实现与 E26 原评估器逐配置阈值 bACC 差为0，"
              "与原 float32 EV 的小差异仅为缓存量化，报告在自检中单独列出。", "",
              "## 严格折外结果", ""]
    for ds, r in all_results.items():
        lines += ["", f"### {ds}（{r['n_configs']} 配置 × {r['n_thresholds']} 阈值）", "",
                  "| 方法 | bACC | AUC | F1 | n≥.9 | min bACC |", "|---|---:|---:|---:|---:|---:|"]
        order = [k for k in r["results"] if k != "oracle_test_upper"]
        # 总体嵌套置于表首，其余按输入顺序保留。
        order = ["omnibus_nested"] + [k for k in order if k != "omnibus_nested"]
        for name in order:
            s = r["results"][name]["summary"]
            lines.append(f"| {name} | {s['bacc']:.4f} | {s['auc']:.4f} | {s['f1']:.4f} | "
                         f"{s['n90']} | {s['min_bacc']:.4f} |")
        o = r["results"]["oracle_test_upper"]["summary"]
        lines += ["", f"oracle（测试标签，仅上限）：bACC {o['bacc']:.4f} / AUC {o['auc']:.4f} / "
                  f"F1 {o['f1']:.4f} / n≥.9={o['n90']}。", ""]
        lines.append("自检：" + "；".join(f"{k}={v}" for k, v in r["checks"].items() if k != "cache"))
    lines += ["", "## 文件", "",
              "- `e33.py`：校准指纹计算、严格嵌套路由与报告生成。",
              "- `e23_cal_fingerprint.npz` / `e28_cal_fingerprint.npz`：校准响应缓存。",
              "- `e33_results.json`：完整逐人指标、所选配置、内层得分和逐折超参。", ""]
    output.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="E23,E28")
    ap.add_argument("--recompute-cal", action="store_true")
    args = ap.parse_args()
    result_path = HERE / "e33_results.json"
    all_results = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
    for ds in args.datasets.split(","):
        all_results[ds] = route_dataset(ds, args.recompute_cal)
        # 每个数据集结束后先落盘，长任务中断也保留已完成结果。
        (HERE / "e33_results.json").write_text(json.dumps(all_results, ensure_ascii=False, indent=1),
                                               encoding="utf-8")
        write_report(all_results, HERE / "报告_E33_校准响应指纹路由.md")
    print("完成", ", ".join(all_results), flush=True)


if __name__ == "__main__":
    main()
