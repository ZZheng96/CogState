"""E40：无本人标签、无块位置依赖的严格因果 LOSO 基线。

特征：E27/E31 的 4390 维 cat -> 从录制首帧累计的因果 z -> 过去300帧因果去均值。
模型：每个目标仅用其余56人的全部样本训练类别平衡池化 LR（alpha=0, lambda=.01）。
分数：raw logit 与从录制首帧累计的因果 z 两种；阈值使用 E26 的85种因果阈值。

运行：
  .venv\Scripts\python.exe test/40_unlabeled_causal/e40.py --sanity 001
  .venv\Scripts\python.exe test/40_unlabeled_causal/e40.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent
sys.path.insert(0, str(TESTS / "17_compliant"))
sys.path.insert(0, str(TESTS / "18_calib"))
import loso as L  # noqa: E402
from calib2 import WS as CAT_WS  # noqa: E402


def _import(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


E26 = _import("e26lib", TESTS / "26_merge2" / "e26lib.py")
E27 = _import("e27", TESTS / "27_common_individual" / "e27.py")

DEV = "cuda"
CENTER_W = 300
LAM = 0.01
SCORE_PATH = HERE / "e40_scores.npz"
RESULT_PATH = HERE / "e40.json"
REPORT_PATH = HERE / "报告_E40_无标签严格因果基线.md"
FIXED = {"fixed_t0": "t0", "fixed_m300": "m300"}


def summary(values: np.ndarray) -> dict:
    a = np.asarray(values, dtype=float)
    return {
        "bacc": float(np.nanmean(a[:, 0])),
        "auc": float(np.nanmean(a[:, 1])),
        "f1": float(np.nanmean(a[:, 2])),
        "n90": int(np.nansum(a[:, 0] >= 0.9)),
        "min": float(np.nanmin(a[:, 0])),
    }


def per_subject(subs: list[str], values: np.ndarray, extra: list[dict] | None = None) -> dict:
    out = {}
    for i, q in enumerate(subs):
        row = {"bacc": float(values[i, 0]), "auc": float(values[i, 1]), "f1": float(values[i, 2])}
        if extra is not None:
            row.update(extra[i])
        out[q] = row
    return out


def build_features(raw: dict, subs: list[str]) -> dict:
    """不使用校准段或块号；所有统计量均只由当前及过去帧计算。"""
    features = {}
    for j, q in enumerate(subs):
        one = {q: raw[q]}
        cat = torch.cat([L.feats(one, [q], w)[q] for w in CAT_WS], dim=1)
        if cat.shape[1] != 4390:
            raise RuntimeError(f"{q}: 预期4390维，实际{cat.shape[1]}")
        mean, std = L.run_stats_t(cat, 99999)
        cat.sub_(mean).div_(std).clamp_(-10, 10)
        mean300, _ = L.run_stats_t(cat, CENTER_W)
        cat.sub_(mean300)
        features[q] = cat.contiguous()
        print(f"features [{j + 1:02d}/{len(subs)}] {q} {tuple(cat.shape)}", flush=True)
    return features


def causal_score_z(score: torch.Tensor) -> torch.Tensor:
    mean, std = L.run_stats_t(score[:, None], 99999)
    return ((score[:, None] - mean) / std)[:, 0]


def fit_target(target: str, subs: list[str], raw: dict, features: dict) -> dict[str, torch.Tensor]:
    """目标样本和目标标签均不进入训练张量；无任何被试编号特判。"""
    others = [q for q in subs if q != target]
    xtrain = torch.cat([features[q] for q in others])
    ytrain = np.concatenate([raw[q][1] for q in others])
    wtrain = torch.cat([E27.bal_w(raw[q][1]) for q in others]) / len(others)
    model = E27.lbfgs_lr(xtrain, ytrain, wtrain, torch.tensor([LAM], device=DEV), iters=150)
    with torch.no_grad():
        raw_score = model(features[target])[:, 0].float()
        zc_score = causal_score_z(raw_score)
    if not bool(torch.isfinite(raw_score).all() and torch.isfinite(zc_score).all()):
        raise RuntimeError(f"{target}: 分数含非有限值")
    out = {"raw": raw_score, "zc": zc_score}
    del model, xtrain, wtrain
    torch.cuda.empty_cache()
    return out


def load_scores() -> dict:
    if not SCORE_PATH.exists():
        return {}
    data = np.load(SCORE_PATH)
    return {q: {"raw": data[f"raw_{q}"], "zc": data[f"zc_{q}"]} for q in data["subjects"].tolist()}


def save_scores(scores: dict):
    arrays = {"subjects": np.asarray(sorted(scores))}
    for q, value in scores.items():
        for variant in ("raw", "zc"):
            s = value[variant]
            arrays[f"{variant}_{q}"] = s.detach().cpu().numpy() if torch.is_tensor(s) else np.asarray(s)
    np.savez_compressed(SCORE_PATH, **arrays)


def loso_threshold(values: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """阈值只由其余56人的后段 bACC 选择，然后才读取目标结果。"""
    nq = values.shape[0]
    total = np.nansum(values[:, :, 0], axis=0)
    count = np.isfinite(values[:, :, 0]).sum(axis=0)
    out = np.zeros((nq, 3))
    selected = []
    for i in range(nq):
        other_sum = total - np.nan_to_num(values[i, :, 0])
        other_count = count - np.isfinite(values[i, :, 0]).astype(int)
        mean = other_sum / np.maximum(other_count, 1)
        chosen = int(np.nanargmax(mean))
        selected.append(chosen)
        out[i] = values[i, chosen]
    return out, selected


def gather_selected(values: np.ndarray, selected: list[int]) -> np.ndarray:
    return np.stack([values[i, selected[i]] for i in range(len(selected))])


def classification_metrics(y: np.ndarray, score: np.ndarray) -> dict:
    """在有限评分帧上评价；score>0 判 task，并附 relax->task 假阳性率。"""
    y = np.asarray(y)
    score = np.asarray(score)
    valid = np.isfinite(score)
    yy, ss = y[valid], score[valid]
    pred = ss > 0
    relax = yy == 0
    task = yy == 1
    out = {
        "task_fpr": float((pred & relax).sum() / relax.sum()) if relax.sum() else float("nan"),
        "valid_rate": float(valid.mean()),
        "n": int(len(y)), "n_relax": int(relax.sum()), "n_task": int(task.sum()),
        "both_classes": bool(np.unique(yy).size == 2),
    }
    if len(yy) and np.unique(yy).size == 2:
        tp = float((pred & task).sum())
        fp = float((pred & relax).sum())
        fn = float((~pred & task).sum())
        tn = float((~pred & relax).sum())
        out.update(bacc=float(.5 * (tp / task.sum() + tn / relax.sum())),
                   auc=float(roc_auc_score(yy, ss)),
                   f1=float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0)
    else:
        out.update(bacc=float("nan"), auc=float("nan"), f1=float("nan"))
    return out


def threshold_aux(score: torch.Tensor, y: np.ndarray, n: int, threshold_indices: list[int]) -> list[dict]:
    """预热段只评价有限阈值帧，不使用任何标签计算阈值。"""
    idx = torch.arange(n, device=DEV)
    thresholds = E26.thr_bank(score[None], idx)[0].detach().cpu().numpy()
    s = score[:n].detach().cpu().numpy()
    yy = y[:n]
    return [classification_metrics(yy, s - thresholds[ti]) for ti in threshold_indices]


def evaluate_scores(subs: list[str], raw: dict, scores: dict, elapsed_s: float = 0.0) -> dict:
    segments = {}
    for q in subs:
        n = len(raw[q][1])
        segments[q] = {
            "test": np.where(raw[q][2] >= 4)[0],
            "warmup300": np.arange(min(300, n)),
            "warmup600": np.arange(min(600, n)),
        }
    result = {
        "protocol": "57人LOSO；目标本人样本/标签完全排除；块号仅用于后段评价；阈值严格由其余56人选择",
        "config": {
            "cat_windows": list(CAT_WS), "feature_dim": 4390,
            "feature_z": "run_stats_t(w=99999)，含当前帧，从录制首帧累计，clamp[-10,10]",
            "center": "run_stats_t(w=300)因果去均值", "alpha": 0.0, "lambda": LAM,
            "score_zc": "raw logit 的 run_stats_t(w=99999) 累计因果z",
            "score_direction": "分值越大越倾向task；E26调整后分数>0判task",
            "threshold_count": len(E26.THR2), "elapsed_s": round(elapsed_s, 1),
        },
        "variants": {},
    }
    for variant in ("raw", "zc"):
        ev = {segment: [] for segment in ("test", "warmup300", "warmup600")}
        tensors = {}
        for q in subs:
            s = scores[q][variant]
            s = s.to(DEV) if torch.is_tensor(s) else torch.tensor(s, device=DEV, dtype=torch.float32)
            tensors[q] = s
            y = raw[q][1]
            for segment in ev:
                ix = segments[q][segment]
                yy = torch.tensor(y[ix], device=DEV).bool()
                metric = E26.metrics(s[None], torch.tensor(ix, device=DEV), yy)[0].cpu().numpy()
                ev[segment].append(metric)
        ev = {k: np.stack(v) for k, v in ev.items()}
        _, selected = loso_threshold(ev["test"])
        modes = {name: [E26.THR2.index(next(x for x in E26.THR2 if x[0] == thr))] * len(subs)
                 for name, thr in FIXED.items()}
        modes["loso"] = selected
        variant_result = {"test": {}, "warmup300": {}, "warmup600": {}, "threshold_names": [x[0] for x in E26.THR2]}
        for segment in ev:
            for mode, indices in modes.items():
                values = gather_selected(ev[segment], indices)
                aux = None
                if segment != "test":
                    aux = []
                    for i, q in enumerate(subs):
                        n = len(segments[q][segment])
                        aux.extend(threshold_aux(tensors[q], raw[q][1], n, [indices[i]]))
                entry = {
                    **summary(values),
                    "per_subject": per_subject(subs, values, aux),
                }
                if aux is not None:
                    values = np.asarray([[x["bacc"], x["auc"], x["f1"]] for x in aux])
                    entry = {**summary(values), "per_subject": per_subject(subs, values, aux)}
                    entry["task_fpr"] = float(np.nanmean([x["task_fpr"] for x in aux]))
                    entry["valid_rate"] = float(np.mean([x["valid_rate"] for x in aux]))
                    entry["both_classes_subjects"] = int(sum(x["both_classes"] for x in aux))
                if segment == "test":
                    if mode in FIXED:
                        entry["threshold"] = FIXED[mode]
                    else:
                        entry["selected_thresholds"] = {q: E26.THR2[indices[i]][0] for i, q in enumerate(subs)}
                variant_result[segment][mode] = entry
        result["variants"][variant] = variant_result
        for mode in modes:
            m = variant_result["test"][mode]
            print(f"E40 {variant:3s} {mode:10s} bACC={m['bacc']:.4f} AUC={m['auc']:.4f} "
                  f"F1={m['f1']:.4f} n90={m['n90']} min={m['min']:.4f}", flush=True)
    return result


def write_report(result: dict):
    lines = [
        "# E40：无本人标签、无块位置依赖的严格因果基线", "",
        "## 方法", "",
        "- 复用 `L.load/L.feats`、E26 因果阈值和 E27/E31 池化 LR。4390 维 cat 后不再使用块0–3统计量，改为从录制首帧开始的累计因果 z，再做过去300帧因果去均值。",
        "- 每个目标只使用其余56人的样本和标签训练 LR（alpha=0，lambda=0.01）；本人只做逐帧推理。所有被试（包括034）走同一路径。",
        "- raw 保留原始 logit；zc 仅用从录制首帧累计的分数均值/标准差。严格 LOSO 阈值由其余56人后段 bACC 选择。块号只用于既有的块4–5最终评价。", "",
        "## 后段（块4–5）结果", "",
        "| 分数 | 阈值方式 | bACC | AUC | F1 | n≥.9 | min |", "|---|---|---:|---:|---:|---:|---:|",
    ]
    for variant in ("raw", "zc"):
        for mode in ("fixed_t0", "fixed_m300", "loso"):
            m = result["variants"][variant]["test"][mode]
            lines.append(f"| {variant} | {mode} | {m['bacc']:.4f} | {m['auc']:.4f} | {m['f1']:.4f} | {m['n90']} | {m['min']:.4f} |")
    lines += ["", "## 录制开头预热段", "",
              "task 假阳性率定义为 relax 被误判为 task 的比例；有效率为分数和阈值均有限的帧比例。只有同时含两类时才解释 bACC/AUC/F1。", "",
              "| 帧段 | 分数 | 阈值方式 | 两类人数 | bACC | AUC | F1 | task假阳性率 | 有效率 |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for segment in ("warmup300", "warmup600"):
        for variant in ("raw", "zc"):
            for mode in ("fixed_t0", "fixed_m300", "loso"):
                m = result["variants"][variant][segment][mode]
                lines.append(f"| {segment} | {variant} | {mode} | {m['both_classes_subjects']}/57 | "
                             f"{m['bacc']:.4f} | {m['auc']:.4f} | {m['f1']:.4f} | "
                             f"{m['task_fpr']:.4f} | {m['valid_rate']:.4f} |")
    lines += ["", "## 034 与输出", "",
              "034 没有编号特判或校准回退，与其余被试一样完全排除本人训练样本/标签，并直接使用纯他人模型。逐人指标见 `e40.json`。",
              "完整逐帧 raw/zc 分数保存在 `e40_scores.npz`，供 E41 做严格前向自适应。", ""]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def sanity(target: str, subs: list[str], raw: dict, features: dict) -> dict:
    score = fit_target(target, subs, raw, features)
    rows = {}
    for variant, s in score.items():
        y, blk = raw[target][1], raw[target][2]
        ix = np.where(blk >= 4)[0]
        values = E26.metrics(s[None], torch.tensor(ix, device=DEV), torch.tensor(y[ix], device=DEV).bool())[0].cpu().numpy()
        rows[variant] = {thr: dict(zip(("bacc", "auc", "f1"), map(float, values[[x[0] for x in E26.THR2].index(thr)])))
                         for thr in ("t0", "m300")}
    out = {
        "target": target, "n_frames": len(raw[target][1]), "feature_dim": int(features[target].shape[1]),
        "train_subjects": len(subs) - 1, "own_training_samples": 0, "own_training_labels": 0,
        "zc_first_frame": float(score["zc"][0]), "all_scores_finite": True, "test": rows,
        "note": "034与其他编号调用完全相同的fit_target；预热段完整统计见全量e40.json",
    }
    (HERE / "sanity.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SANITY", json.dumps(out, ensure_ascii=False), flush=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity", metavar="SUBJECT", help="只训练并检查一个被试，不写全量分数缓存")
    args = parser.parse_args()
    start = time.time()
    subs, raw = L.load()
    if args.sanity:
        if args.sanity not in subs:
            raise ValueError(f"未知被试 {args.sanity}")
        features = build_features(raw, subs)
        sanity(args.sanity, subs, raw, features)
        return
    scores = load_scores()
    missing = [q for q in subs if q not in scores]
    if missing:
        features = build_features(raw, subs)
        for j, q in enumerate(missing):
            value = fit_target(q, subs, raw, features)
            scores[q] = {k: v.detach().cpu().numpy() for k, v in value.items()}
            save_scores(scores)
            print(f"score [{j + 1:02d}/{len(missing)}] {q} saved; elapsed={time.time() - start:.0f}s", flush=True)
        del features
        torch.cuda.empty_cache()
    result = evaluate_scores(subs, raw, scores, time.time() - start)
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(result)
    print(f"saved {RESULT_PATH} and {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
