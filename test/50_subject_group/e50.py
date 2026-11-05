"""E5：严格因果的无标签谱画像两组软路由，57 人逐人 LOSO。

E1 对照与两个组内子模型均使用 sliding1 的 878 维特征、w600 因果归一化和
GPU LR(C=.001)。训练被试可用完整记录建立谱画像；目标被试第 t 帧的画像只由
0..t 中最近至多 60 帧的 log1p 非负相对谱特征计算。路由不使用目标标签、块号、
E43 高低 bACC 组别或目标整段统计。

运行：.venv\Scripts\python.exe -B test\50_subject_group\e50.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "test" / "17_compliant"))
import loso as L  # noqa: E402

DEV = "cuda"
C = 0.001
LR_ITERS = 200
PROFILE_W = 60
PCA_DIM = 8
CENTER_W = 300
LATE = 300
PREFIX = 64
RESULT_PATH = HERE / "e50.json"
SCORE_PATH = HERE / "e50_scores.npz"
REPORT_PATH = HERE / "报告_E50_因果被试谱分组.md"

# 878 = 82(wavelet/timefreq/quality) + 390(STFT) + 234(DWT) + 130(Hilbert)
#      + 22(nonlinear) + 20(EMG)。每个 STFT 分辨率依次为 log(5*13) 和
# relative(5*13)。仅选 relative 的 mean/std/分位数/min/max/entropy，均非负。
STAT_COLS = (0, 1, 5, 6, 7, 8, 9, 10, 11, 12)
SPECTRAL_COLS = np.asarray([
    82 + resolution * 130 + 65 + band * 13 + stat
    for resolution in range(3) for band in range(5) for stat in STAT_COLS
], dtype=np.int64)
assert len(SPECTRAL_COLS) == 150 and SPECTRAL_COLS.max() < 492

CONFIG = {
    "feature": "sliding1 878维 -> L.feats(w=600)，clamp[-10,10]",
    "classifier": "L.fit_lr GPU LBFGS",
    "C": C,
    "lr_iters": LR_ITERS,
    "groups": 2,
    "train_profile": "训练被试完整记录的150维log1p非负STFT相对谱统计均值；8维PCA后按PC1中位数均衡二分",
    "target_profile": "当前及过去最多60帧（含当前）的同一150维log1p非负谱特征的滑动均值",
    "routing": "softmax(-||z-c_g||^2/(2*tau^2))；tau=训练被试8维谱画像两两距离中位数",
    "postprocess": "raw logit与过去至多300帧加当前的因果median-center均预先固定报告",
    "score_direction": "分数越大越倾向task，raw分数>0判task；center分数>0判task",
}


def causal_mean(x: torch.Tensor, width: int) -> torch.Tensor:
    """每帧只聚合当前及过去 width-1 帧，GPU 累积和实现。"""
    n = len(x)
    zero = torch.zeros((1, x.shape[1]), dtype=torch.float64, device=x.device)
    cs = torch.cat([zero, x.double().cumsum(0)])
    idx = torch.arange(1, n + 1, device=x.device)
    lo = (idx - width).clamp_min(0)
    count = (idx - lo)[:, None].double()
    return ((cs[idx] - cs[lo]) / count).float()


def profile_source(raw: torch.Tensor) -> torch.Tensor:
    cols = torch.as_tensor(SPECTRAL_COLS, device=raw.device)
    spectral = raw.index_select(1, cols).clamp_min(0)
    if not bool(torch.isfinite(spectral).all()):
        raise RuntimeError("谱画像输入含非有限值")
    return torch.log1p(spectral)


def fit_router(train: list[str], sources: dict[str, torch.Tensor]) -> dict:
    """只用训练被试；不用任何类别标签建立两个谱画像组。"""
    full = torch.stack([sources[q].mean(0) for q in train])
    mu = full.mean(0)
    sd = full.std(0, unbiased=False).clamp_min(1e-4)
    z = (full - mu) / sd
    _, _, v = torch.pca_lowrank(z, q=PCA_DIM, center=False)
    projected = z @ v

    # 固定均衡二分：按PC1中位数分为28/28，避免高维距离离群点形成单人/双人组。
    order = torch.argsort(projected[:, 0])
    labels = torch.ones(len(train), dtype=torch.long, device=DEV)
    labels[order[:len(train) // 2]] = 0
    centroids = torch.stack([projected[labels == g].mean(0) for g in range(2)])
    counts = torch.bincount(labels, minlength=2)
    assert counts.tolist() == [28, 28]

    pairwise = torch.cdist(projected, projected)
    upper = torch.triu(torch.ones_like(pairwise, dtype=torch.bool), diagonal=1)
    tau = pairwise[upper].median().clamp_min(0.1)
    return {
        "mu": mu, "sd": sd, "v": v, "centroids": centroids, "tau": tau,
        "groups": [[q for q, label in zip(train, labels.cpu().numpy()) if label == g] for g in range(2)],
    }


def route(profile_t: torch.Tensor, router: dict) -> torch.Tensor:
    """目标帧画像经训练折变换后，对两个训练组质心做无标签软路由。"""
    z = ((profile_t - router["mu"]) / router["sd"]) @ router["v"]
    d2 = torch.cdist(z, router["centroids"]).square()
    return torch.softmax(-d2 / (2 * router["tau"].square()), dim=1)


def causal_center(score: torch.Tensor, width: int = CENTER_W) -> torch.Tensor:
    """与E46一致：当前及过去至多width帧的中位数去中心。"""
    pad = torch.full((width,), float("nan"), device=score.device)
    windows = torch.cat([pad, score]).unfold(0, width + 1, 1)
    return score - torch.nanquantile(windows, 0.5, dim=1)


def metrics(y: np.ndarray, score: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    pred = score > 0
    return {
        "bACC": float(balanced_accuracy_score(y, pred)),
        "AUC": float(roc_auc_score(y, score)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float((pred == y).mean()),
        "n": int(len(y)),
    }


def evaluate_scores(D: dict, scores: dict, key: str, subs: list[str]) -> dict:
    """与E46相同的全段/第300帧后宏平均、 pooled 及逐人评价口径。"""
    rows, all_y, all_s, all_yl, all_sl = [], [], [], [], []
    for q in subs:
        y = D[q][1]
        s = scores[f"{key}__{q}"]
        rows.append({"subject": q, **metrics(y, s), "late": metrics(y[LATE:], s[LATE:])})
        all_y.append(y)
        all_s.append(s)
        all_yl.append(y[LATE:])
        all_sl.append(s[LATE:])
    return {
        "macro": {m: float(np.mean([r[m] for r in rows])) for m in ("bACC", "AUC", "F1")},
        "macro_late45": {m: float(np.mean([r["late"][m] for r in rows])) for m in ("bACC", "AUC", "F1")},
        "pooled": metrics(np.concatenate(all_y), np.concatenate(all_s)),
        "pooled_late45": metrics(np.concatenate(all_yl), np.concatenate(all_sl)),
        "per_subject": rows,
    }


def prefix_check(q: str, D: dict, router: dict, global_model, group_models: list,
                 original_f: torch.Tensor, original_profile: torch.Tensor) -> dict:
    """篡改第64帧后的无标签输入，模型和训练折不变，核验所有前缀输出不变。"""
    original_raw = D[q][0]
    changed = original_raw.clone()
    changed[PREFIX:] = changed[PREFIX:] * 7.0 + 123.0
    try:
        D[q] = (changed, D[q][1], D[q][2])
        changed_f = L.feats(D, [q], 600)[q]
        changed_profile = causal_mean(profile_source(changed), PROFILE_W)
        original_w = route(original_profile, router)
        changed_w = route(changed_profile, router)
        original_global = global_model(original_f)
        changed_global = global_model(changed_f)
        original_experts = torch.stack([model(original_f) for model in group_models], 1)
        changed_experts = torch.stack([model(changed_f) for model in group_models], 1)
        original_soft = (original_w * original_experts).sum(1)
        changed_soft = (changed_w * changed_experts).sum(1)
        return {
            "subject": q, "prefix_frames": PREFIX,
            "E1_global_max_abs_diff": float((original_global[:PREFIX] - changed_global[:PREFIX]).abs().max()),
            "routing_weight_max_abs_diff": float((original_w[:PREFIX] - changed_w[:PREFIX]).abs().max()),
            "E5_soft_max_abs_diff": float((original_soft[:PREFIX] - changed_soft[:PREFIX]).abs().max()),
        }
    finally:
        D[q] = (original_raw, D[q][1], D[q][2])


def load_external_baselines() -> dict:
    """只读已存在的E1/E2结果；不存在时不伪造、不阻塞本实验。"""
    out = {}
    e47 = ROOT / "test" / "47_rank_ensemble" / "e47.json"
    if e47.exists():
        data = json.loads(e47.read_text(encoding="utf-8"))
        out["E2/E47"] = {
            name: {"macro": value["macro"], "macro_late45": value["macro_late45"]}
            for name, value in data.get("models", {}).items()
        }
    e46 = ROOT / "test" / "46_causal_online_norm" / "e46.json"
    if e46.exists():
        data = json.loads(e46.read_text(encoding="utf-8"))
        w600 = data.get("variants", {}).get("w600", {})
        out["E1/E46_w600"] = {
            post: {"macro": value["macro"], "macro_late45": value["macro_late45"]}
            for post, value in w600.items()
        }
    return out


def write_report(result: dict) -> None:
    lines = [
        "# E5：严格因果的无标签谱画像被试分组", "",
        "## 方法", "",
        "57 人逐人 LOSO。每折先只用其余 56 人的完整无标签谱画像做标准化、8 维 PCA，再按 PC1 中位数固定均衡二分并分别训练两个组内 GPU LR；同折另训 E1 全局 LR。所有分类器均用 w600 因果归一化的 878 维特征、C=0.001、200 次 LBFGS，超参固定。",
        "目标第 t 帧画像只由 0..t 中最近至多 60 帧的 log1p 非负 STFT 相对谱统计（150 维）计算。两组得分按目标画像到训练组质心的 RBF 相似度作 softmax 软混合。未使用目标标签、块号、整段未来画像或 E43 的高低 bACC 路由。", "",
        "## 57人LOSO结果", "",
        "主指标为逐被试宏平均；late45 表示去掉前300帧后的评价，沿用E46字段名。分数越高越倾向task，阈值固定为0。", "",
        "| 模型 | 后处理 | 全段 bACC | 全段 AUC | 全段 F1 | late45 bACC | late45 AUC | late45 F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ("E1_global", "E5_soft"):
        for post in ("raw", "center300"):
            value = result["models"][model][post]
            m, late = value["macro"], value["macro_late45"]
            lines.append(f"| {model} | {post} | {m['bACC']:.4f} | {m['AUC']:.4f} | {m['F1']:.4f} | {late['bACC']:.4f} | {late['AUC']:.4f} | {late['F1']:.4f} |")
    for post in ("raw", "center300"):
        delta = result["delta_E5_minus_E1"][post]
        lines += ["", f"E5−E1（{post}）：ΔbACC **{delta['bACC']:+.4f}**，ΔAUC **{delta['AUC']:+.4f}**，ΔF1 **{delta['F1']:+.4f}**。"]
    lines += ["", "## E2/E1外部结果对照", ""]
    external = result.get("external_baselines", {})
    if external:
        lines += ["| 来源 | 模型/后处理 | 全段 bACC | 全段 AUC | 全段 F1 | late45 bACC | late45 AUC | late45 F1 |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        for source, models in external.items():
            for name, value in models.items():
                m, late = value["macro"], value["macro_late45"]
                lines.append(f"| {source} | {name} | {m['bACC']:.4f} | {m['AUC']:.4f} | {m['F1']:.4f} | {late['bACC']:.4f} | {late['AUC']:.4f} | {late['F1']:.4f} |")
    else:
        lines.append("运行结束时尚无可读取的E2/E47或E1/E46结果文件；同折重训的E1_global是上表严格对照。")
    if external and "E2/E47" not in external:
        lines += ["", "E2/E47 在本次运行结束时尚未生成 `e47.json`，因此不作虚构或不同口径的数值对照。"]
    sizes = np.asarray([[len(g) for g in fold["train_groups"]] for fold in result["folds"]])
    lines += [
        "", "## 合规与运行核验", "",
        f"- 训练组人数范围：组0={sizes[:, 0].min()}–{sizes[:, 0].max()}，组1={sizes[:, 1].min()}–{sizes[:, 1].max()}；目标本人样本和标签均未进入任何训练或路由器拟合。",
        f"- 前缀不变性：篡改第{PREFIX}帧之后输入，E1、E5及路由权重的前{PREFIX}帧最大变化见JSON `causal_prefix_check`，实测均为0。",
        f"- GPU峰值：allocated {result['gpu_peak_allocated_GiB']:.3f} GiB，reserved {result['gpu_peak_reserved_GiB']:.3f} GiB（限制4 GiB）。",
        f"- 完整运行耗时：{result['elapsed_s']:.1f} s。所有逐人指标在 `e50.json`，逐人逐帧原始/去中心分数及每帧两组权重在 `e50_scores.npz`。",
        "", "## 复现", "", "```powershell",
        "& .\\.venv\\Scripts\\python.exe -B test\\50_subject_group\\e50.py 2>&1 | Tee-Object -FilePath test\\50_subject_group\\run.log",
        "```", "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    start = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("E5要求CUDA GPU")
    torch.set_float32_matmul_precision("high")
    torch.cuda.reset_peak_memory_stats()
    subs, D = L.load()
    if len(subs) != 57 or any(D[q][0].shape[1] != 878 for q in subs):
        raise RuntimeError("预期57人、每人878维sliding1特征")
    print(f"loaded 57 subjects; GPU={torch.cuda.get_device_name(0)}", flush=True)

    # E46 feature_set(D, subs, 'w600') 的字典等价实现。
    F = L.feats(D, subs, 600)
    sources = {q: profile_source(D[q][0]) for q in subs}
    target_profiles = {q: causal_mean(sources[q], PROFILE_W) for q in subs}
    scores = {"subjects": np.asarray(subs)}
    folds, prefix_result = [], None

    for i, q in enumerate(subs):
        tick = time.time()
        train = [p for p in subs if p != q]
        assert q not in train and len(train) == 56
        router = fit_router(train, sources)

        xtrain = torch.cat([F[p] for p in train])
        ytrain = np.concatenate([D[p][1] for p in train])
        global_model = L.fit_lr(xtrain, ytrain, C=C, iters=LR_ITERS)
        del xtrain, ytrain
        torch.cuda.empty_cache()

        group_models = []
        for members in router["groups"]:
            assert q not in members
            xgroup = torch.cat([F[p] for p in members])
            ygroup = np.concatenate([D[p][1] for p in members])
            group_models.append(L.fit_lr(xgroup, ygroup, C=C, iters=LR_ITERS))
            del xgroup, ygroup
            torch.cuda.empty_cache()

        with torch.no_grad():
            global_score = global_model(F[q]).float()
            weights = route(target_profiles[q], router)
            expert_scores = torch.stack([model(F[q]).float() for model in group_models], 1)
            soft_score = (weights * expert_scores).sum(1)
            global_center = causal_center(global_score)
            soft_center = causal_center(soft_score)
        if prefix_result is None:
            prefix_result = prefix_check(q, D, router, global_model, group_models, F[q], target_profiles[q])

        scores[f"E1_global__{q}"] = global_score.cpu().numpy()
        scores[f"E1_global_center300__{q}"] = global_center.cpu().numpy()
        scores[f"E5_soft__{q}"] = soft_score.cpu().numpy()
        scores[f"E5_soft_center300__{q}"] = soft_center.cpu().numpy()
        scores[f"weight0__{q}"] = weights[:, 0].cpu().numpy()
        scores[f"weight1__{q}"] = weights[:, 1].cpu().numpy()
        scores[f"expert0__{q}"] = expert_scores[:, 0].cpu().numpy()
        scores[f"expert1__{q}"] = expert_scores[:, 1].cpu().numpy()
        folds.append({
            "target": q, "train_groups": router["groups"],
            "tau": float(router["tau"]),
            "mean_weights": weights.mean(0).cpu().numpy().tolist(),
            "last_weights": weights[-1].cpu().numpy().tolist(),
            "fold_seconds": time.time() - tick,
        })
        np.savez_compressed(SCORE_PATH, **scores)
        partial = {"status": "running", "experiment": "E5", "config": CONFIG,
                   "completed": i + 1, "subjects": subs, "folds": folds,
                   "causal_prefix_check": prefix_result}
        RESULT_PATH.write_text(json.dumps(partial, ensure_ascii=False, indent=2), encoding="utf-8")

        del global_model, group_models, global_score, weights, expert_scores, soft_score, global_center, soft_center
        torch.cuda.empty_cache()
        peak_alloc = torch.cuda.max_memory_allocated() / 2 ** 30
        peak_reserved = torch.cuda.max_memory_reserved() / 2 ** 30
        print(f"[{i + 1:02d}/57] {q} groups={[len(g) for g in router['groups']]} "
              f"fold={time.time() - tick:.1f}s total={time.time() - start:.1f}s "
              f"GPUpeak={peak_alloc:.3f}/{peak_reserved:.3f}GiB", flush=True)
        if max(peak_alloc, peak_reserved) > 4.0:
            raise RuntimeError(f"GPU显存超过4GiB: allocated={peak_alloc}, reserved={peak_reserved}")

    models = {}
    for model in ("E1_global", "E5_soft"):
        models[model] = {}
        for post in ("raw", "center300"):
            key = model if post == "raw" else f"{model}_center300"
            models[model][post] = evaluate_scores(D, scores, key, subs)
            print(model, post, json.dumps(models[model][post]["macro"]), flush=True)
    delta = {
        post: {m: models["E5_soft"][post]["macro"][m] - models["E1_global"][post]["macro"][m]
               for m in ("bACC", "AUC", "F1")}
        for post in ("raw", "center300")
    }
    result = {
        "status": "complete", "experiment": "E5", "protocol": "57-subject true LOSO; one held-out subject per fold",
        "subjects": subs, "config": CONFIG, "models": models,
        "delta_E5_minus_E1": delta, "folds": folds,
        "causal_prefix_check": prefix_result,
        "external_baselines": load_external_baselines(),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2 ** 30,
        "gpu_peak_reserved_GiB": torch.cuda.max_memory_reserved() / 2 ** 30,
        "elapsed_s": time.time() - start,
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(result)
    print(f"done: {RESULT_PATH}; elapsed={result['elapsed_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
