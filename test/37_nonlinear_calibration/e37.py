"""E37：校准一致特征 + 个体 HistGB + E31 连续分数融合。

本人模型只使用块 0--3；034 无 relax 校准，回退到 E31 纯 pool 分数。
配置、融合权重和 E26 的 85 种因果阈值均由其余 56 人平均测试 bACC 选择。

sanity：.venv\Scripts\python.exe test/37_nonlinear_calibration/e37.py --sanity
完整：  .venv\Scripts\python.exe test/37_nonlinear_calibration/e37.py
小网格：.venv\Scripts\python.exe test/37_nonlinear_calibration/e37.py --grid small
权重细化（不重训）：.venv\Scripts\python.exe test/37_nonlinear_calibration/e37.py --refine
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = Path(__file__).resolve().parent
T = HERE.parent
sys.path.insert(0, str(T / "17_compliant"))


def _imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


E26 = _imp("e26lib", T / "26_merge2" / "e26lib.py")
E31 = _imp("e31_general", T / "31_general_algorithm" / "general.py")
DEV = "cuda"
BASE_NAME = "b|pool|w300|a0.75"
WEIGHTS = (0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5)  # HistGB 的权重，其余为 E31 pool
DEFAULT_CONFIGS = (
    dict(name="k200_i80_l7", k=200, max_iter=80, max_leaf_nodes=7, min_samples_leaf=20, l2=1.0),
)
SMALL_CONFIGS = DEFAULT_CONFIGS + (
    dict(name="k100_i80_l7", k=100, max_iter=80, max_leaf_nodes=7, min_samples_leaf=20, l2=1.0),
    dict(name="k400_i80_l7", k=400, max_iter=80, max_leaf_nodes=7, min_samples_leaf=20, l2=1.0),
)


def zcal(score, cal):
    """score 为 GPU 一维向量，cal 为同设备布尔掩码。"""
    return (score - score[cal].mean()) / (score[cal].std() + 1e-6)


def select_features(X, y, blk, k):
    """仅用块 0--3：类效应量 × 两个 (R,T) 块对的方向/幅度一致性。"""
    yt = torch.as_tensor(y, device=DEV)
    bt = torch.as_tensor(blk, device=DEV)
    cal = bt <= 3
    Xc = X[cal]
    yc = yt[cal]
    m0, m1 = Xc[yc == 0].mean(0), Xc[yc == 1].mean(0)
    v0 = Xc[yc == 0].var(0, unbiased=False)
    v1 = Xc[yc == 1].var(0, unbiased=False)
    effect = (m1 - m0) / (0.5 * (v0 + v1)).sqrt().clamp_min(1e-6)
    pair_d = []
    for r, t in ((0, 1), (2, 3)):
        pair_d.append(X[bt == t].mean(0) - X[bt == r].mean(0))
    d1, d2 = pair_d
    agree = d1 * d2 > 0
    consistency = 2 * (d1.abs() * d2.abs()).sqrt() / (d1.abs() + d2.abs() + 1e-6)
    merit = effect.abs() * torch.where(agree, consistency, torch.zeros_like(consistency))
    merit = torch.nan_to_num(merit, nan=0.0, posinf=0.0, neginf=0.0)
    idx = torch.topk(merit, k=k).indices
    return idx, int((merit > 0).sum()), int(agree.sum())


def fit_score(X, y, blk, cfg, base):
    """返回 HistGB 校准 z 分数及筛选诊断；034 直接使用纯 pool 回退。"""
    calnp = blk <= 3
    if len(np.unique(y[calnp])) < 2:
        return base.clone(), dict(fallback="pure_pool", n_positive=0, n_agree=0)
    idx, n_positive, n_agree = select_features(X, y, blk, cfg["k"])
    idxnp = idx.cpu().numpy()
    Xn = X[:, idx].cpu().numpy()
    model = HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_iter=cfg["max_iter"],
        max_leaf_nodes=cfg["max_leaf_nodes"],
        min_samples_leaf=cfg["min_samples_leaf"],
        l2_regularization=cfg["l2"],
        class_weight="balanced",
        early_stopping=False,
        random_state=2026,
    )
    model.fit(Xn[calnp], y[calnp])
    raw_score = model.decision_function(Xn)  # 正 logit 对应 task（y=1）
    score = torch.as_tensor(raw_score, dtype=torch.float32, device=DEV)
    cal = torch.as_tensor(calnp, device=DEV)
    return zcal(score, cal), dict(
        fallback=None, n_positive=n_positive, n_agree=n_agree,
        selected_features=idxnp.tolist(), train_n=int(calnp.sum()), score_kind="decision_function_logit",
    )


def summary(values):
    values = np.asarray(values, dtype=float)
    return dict(
        bacc=float(np.nanmean(values[:, 0])), auc=float(np.nanmean(values[:, 1])),
        f1=float(np.nanmean(values[:, 2])), n90=int(np.nansum(values[:, 0] >= 0.9)),
        min=float(np.nanmin(values[:, 0])),
    )


def per_subject(subs, values):
    return {q: dict(bacc=float(v[0]), auc=float(v[1]), f1=float(v[2])) for q, v in zip(subs, values)}


def selection(EV, configs, subs):
    """EV (C,W,Q,85,3)，分别计算严格 LOSO、全 57 人固定配置和 oracle 上限。"""
    B = EV[..., 0]
    nq = len(subs)
    nested = np.zeros((nq, 3))
    picks = {}
    for i, q in enumerate(subs):
        mean = np.nanmean(np.delete(B, i, axis=2), axis=2)
        c, w, t = np.unravel_index(np.nanargmax(mean), mean.shape)
        nested[i] = EV[c, w, i, t]
        picks[q] = dict(config=configs[c]["name"], weight=float(WEIGHTS[w]), threshold=E26.THR2[t][0],
                        train_mean_bacc=float(mean[c, w, t]))
    mean_all = np.nanmean(B, axis=2)
    cf, wf, tf = np.unravel_index(np.nanargmax(mean_all), mean_all.shape)
    fixed_values = EV[cf, wf, :, tf]
    oracle = np.zeros((nq, 3))
    oracle_picks = {}
    for i, q in enumerate(subs):
        c, w, t = np.unravel_index(np.nanargmax(B[:, :, i, :]), B[:, :, i, :].shape)
        oracle[i] = EV[c, w, i, t]
        oracle_picks[q] = dict(config=configs[c]["name"], weight=float(WEIGHTS[w]), threshold=E26.THR2[t][0])
    nested_result = dict(
        **summary(nested), per_subject=per_subject(subs, nested), picks=picks,
        config_counts=dict(Counter(p["config"] for p in picks.values())),
        weight_counts=dict(Counter(str(p["weight"]) for p in picks.values())),
        threshold_counts=dict(Counter(p["threshold"] for p in picks.values())),
    )
    fixed_result = dict(
        **summary(fixed_values), config=configs[cf]["name"], weight=float(WEIGHTS[wf]),
        threshold=E26.THR2[tf][0], selection_mean_bacc=float(mean_all[cf, wf, tf]),
        selection_bias="配置、权重和阈值由包含测试者本人的全部57人择优，存在选择偏差，不是完全折外结果。",
        per_subject=per_subject(subs, fixed_values),
    )
    oracle_result = dict(**summary(oracle), picks=oracle_picks, warning="逐人测试段择优，仅作不可部署上限。")
    return nested_result, fixed_result, oracle_result


def run(sanity=False, grid="default"):
    t0 = time.time()
    configs = list(DEFAULT_CONFIGS if grid == "default" else SMALL_CONFIGS)
    subs, raw, features = E31.prepare_features()
    cache = np.load(T / "28_block_invariant" / "e28_scores.npz")
    names = cache["names"].tolist()
    base_i = names.index(BASE_NAME)
    assert set(subs) == {k[1:] for k in cache.files if k.startswith("s")}
    run_subs = ["001"] if sanity else subs
    assert set(run_subs) <= set(subs)
    print(f"features ready: {len(subs)} subjects, {features[subs[0]].shape[1]} dims, {time.time()-t0:.1f}s", flush=True)

    ev_rows, score_rows, diagnostics = [], {}, {}
    for qi, q in enumerate(run_subs):
        X = features[q]
        y, blk = raw[q][1], raw[q][2]
        calnp = blk == 3 if q == "034" else blk <= 3
        cal = torch.as_tensor(calnp, device=DEV)
        base = torch.as_tensor(cache[f"s{q}"][base_i].astype(np.float32), device=DEV)
        base = zcal(base, cal)
        hs, fused = [], []
        diagnostics[q] = []
        for cfg in configs:
            h, info = fit_score(X, y, blk, cfg, base)
            diagnostics[q].append(info)
            hs.append(h)
            fused.append(torch.stack([zcal(w * h + (1 - w) * base, cal) for w in WEIGHTS]))
        H = torch.stack(hs)  # (C,T)
        F = torch.stack(fused)  # (C,W,T)
        tst = torch.as_tensor(np.where(blk >= 4)[0], device=DEV)
        yy = torch.as_tensor(y[blk >= 4], device=DEV).bool()
        metric = E26.metrics(F.reshape(-1, F.shape[-1]), tst, yy).cpu().numpy()
        ev_rows.append(metric.reshape(len(configs), len(WEIGHTS), len(E26.THR2), 3))
        for ci in range(len(configs)):
            score_rows[f"hgb_c{ci}_{q}"] = H[ci].cpu().numpy()
            score_rows[f"fused_c{ci}_{q}"] = F[ci].cpu().numpy()
        score_rows[f"base_{q}"] = base.cpu().numpy()
        best = np.nanmax(ev_rows[-1][..., 0])
        print(f"[{qi+1:02d}/{len(run_subs)}] {q} diagnostic_max_bACC={best:.3f} {time.time()-t0:.1f}s", flush=True)
        del H, F, metric
        torch.cuda.empty_cache()

    EV = np.stack(ev_rows, axis=2)  # (C,W,Q,85,3)
    if sanity:
        # sanity 只验证维度、有限性、034 回退和固定融合/阈值能完整运行，不做模型择优。
        ti = [x[0] for x in E26.THR2].index("m300")
        wi = WEIGHTS.index(0.5)
        values = EV[0, wi, :, ti]
        assert EV.shape[0] == 1 and EV.shape[1] == len(WEIGHTS) and EV.shape[3] == 85
        assert np.isfinite(EV).all()
        for key in score_rows:
            assert np.isfinite(score_rows[key]).all(), key
        if "034" in run_subs:
            assert np.allclose(score_rows["hgb_c0_034"], score_rows["base_034"])
        result = dict(
            status="passed", subjects=run_subs, config=configs[0], weights=list(WEIGHTS), n_thresholds=85,
            sanity_fixed=dict(config=configs[0]["name"], weight=0.5, threshold="m300", **summary(values),
                              per_subject=per_subject(run_subs, values)),
            note="仅运行检查；diagnostic_max 使用测试标签，不视作实验结果。", elapsed_s=round(time.time()-t0, 1),
        )
        (HERE / "e37_sanity.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print("SANITY PASSED", result["sanity_fixed"], flush=True)
        return result

    nested, fixed, oracle = selection(EV, configs, subs)
    ev28 = np.load(T / "28_block_invariant" / "e28_ev.npz")
    assert ev28["subs"].tolist() == subs
    bi = ev28["names"].tolist().index(BASE_NAME)
    ti = [x[0] for x in E26.THR2].index("m300")
    baseline_values = ev28["EV"][bi, :, ti]
    result = dict(
        protocol="本人HistGB仅用块0-3；034回退纯pool；完全折外结果逐人由其余56人平均测试bACC共同选择配置、HistGB融合权重和85种因果阈值。",
        data=dict(n_subjects=len(subs), feature_dim=int(features[subs[0]].shape[1]), center_w=300,
                  feature_selection="校准类效应量×(0,1)/(2,3)块对方向与幅度一致性", test_blocks=">=4"),
        configs=configs, weights=list(WEIGHTS), thresholds=[x[0] for x in E26.THR2], base=BASE_NAME,
        baseline_e31_fixed=dict(**summary(baseline_values), config=BASE_NAME, threshold="m300",
                                per_subject=per_subject(subs, baseline_values)),
        nested_loso=nested, fixed_all57=fixed, oracle_upper=oracle,
        diagnostics=diagnostics, elapsed_s=round(time.time()-t0, 1),
    )
    (HERE / "e37.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(
        HERE / "e37_scores.npz", EV=EV.astype(np.float32), configs=np.array([c["name"] for c in configs]),
        weights=np.array(WEIGHTS), thresholds=np.array([x[0] for x in E26.THR2]), subs=np.array(subs), **score_rows,
    )
    for name, value in (("E31 fixed", result["baseline_e31_fixed"]), ("nested LOSO", nested),
                        ("fixed all57 (biased)", fixed), ("oracle upper", oracle)):
        print(f"{name}: bACC={value['bacc']:.6f} AUC={value['auc']:.6f} F1={value['f1']:.6f} "
              f"n90={value['n90']} min={value['min']:.6f}", flush=True)
    print(f"fixed pick: {fixed['config']} w={fixed['weight']} {fixed['threshold']}; elapsed={result['elapsed_s']}s", flush=True)
    return result


def refine():
    """只重评保存的连续分数；旧权重 0/.25/.5 的 85 阈值指标原样复用。"""
    t0 = time.time()
    path = HERE / "e37_scores.npz"
    previous = json.loads((HERE / "e37.json").read_text(encoding="utf-8"))
    old = np.load(path)
    subs = old["subs"].tolist()
    configs = previous["configs"]
    assert len(configs) == 1 and configs[0]["name"] == "k200_i80_l7"
    old_weights = old["weights"].tolist()
    old_ev = old["EV"].copy()
    score_rows = {k: old[k].copy() for k in old.files if k.startswith("hgb_") or k.startswith("base_")}
    old.close()
    _, raw = E31.L.load()
    assert list(raw) == subs
    reused = [i for i, w in enumerate(WEIGHTS) if w in old_weights]
    missing = [i for i, w in enumerate(WEIGHTS) if w not in old_weights]
    EV = np.zeros((1, len(WEIGHTS), len(subs), len(E26.THR2), 3), dtype=np.float32)
    for wi in reused:
        EV[:, wi] = old_ev[:, old_weights.index(WEIGHTS[wi])]
    for qi, q in enumerate(subs):
        y, blk = raw[q][1], raw[q][2]
        calnp = blk == 3 if q == "034" else blk <= 3
        cal = torch.as_tensor(calnp, device=DEV)
        h = torch.as_tensor(score_rows[f"hgb_c0_{q}"], device=DEV)
        base = torch.as_tensor(score_rows[f"base_{q}"], device=DEV)
        F = torch.stack([zcal(w * h + (1 - w) * base, cal) for w in WEIGHTS])[None]
        tst = torch.as_tensor(np.where(blk >= 4)[0], device=DEV)
        yy = torch.as_tensor(y[blk >= 4], device=DEV).bool()
        metric = E26.metrics(F[0, missing], tst, yy).cpu().numpy()
        for j, wi in enumerate(missing):
            EV[0, wi, qi] = metric[j]
        score_rows[f"fused_c0_{q}"] = F[0].cpu().numpy()
        print(f"[{qi+1:02d}/{len(subs)}] {q} refined {time.time()-t0:.1f}s", flush=True)
        del F, metric
        torch.cuda.empty_cache()
    nested, fixed, oracle = selection(EV, configs, subs)
    ev28 = np.load(T / "28_block_invariant" / "e28_ev.npz")
    bi = ev28["names"].tolist().index(BASE_NAME)
    ti = [x[0] for x in E26.THR2].index("m300")
    baseline_values = ev28["EV"][bi, :, ti]
    previous.update(
        weights=list(WEIGHTS), thresholds=[x[0] for x in E26.THR2],
        baseline_e31_fixed=dict(**summary(baseline_values), config=BASE_NAME, threshold="m300",
                                per_subject=per_subject(subs, baseline_values)),
        nested_loso=nested, fixed_all57=fixed, oracle_upper=oracle,
        refinement=dict(histgb_retrained=False, weights=list(WEIGHTS), reused_weights=[WEIGHTS[i] for i in reused],
                        newly_evaluated_weights=[WEIGHTS[i] for i in missing], elapsed_s=round(time.time()-t0, 1)),
    )
    previous["elapsed_s"] = round(previous["elapsed_s"] + time.time() - t0, 1)
    (HERE / "e37.json").write_text(json.dumps(previous, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(
        path, EV=EV, configs=np.array([c["name"] for c in configs]), weights=np.array(WEIGHTS),
        thresholds=np.array([x[0] for x in E26.THR2]), subs=np.array(subs), **score_rows,
    )
    for name, value in (("E31 fixed", previous["baseline_e31_fixed"]), ("nested LOSO", nested),
                        ("fixed all57 (biased)", fixed), ("oracle upper", oracle)):
        print(f"{name}: bACC={value['bacc']:.6f} AUC={value['auc']:.6f} F1={value['f1']:.6f} "
              f"n90={value['n90']} min={value['min']:.6f}", flush=True)
    print(f"fixed pick: {fixed['config']} w={fixed['weight']} {fixed['threshold']}; "
          f"refine elapsed={previous['refinement']['elapsed_s']}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description="E37 个体非线性校准与E31融合")
    parser.add_argument("--sanity", action="store_true", help="只在001上运行小型检查")
    parser.add_argument("--grid", choices=("default", "small"), default="default",
                        help="default=k200；small另含k100/k400，均由外层LOSO选择")
    parser.add_argument("--refine", action="store_true", help="仅用缓存分数细化融合权重，不重训HistGB")
    args = parser.parse_args()
    refine() if args.refine else run(sanity=args.sanity, grid=args.grid)


if __name__ == "__main__":
    main()
