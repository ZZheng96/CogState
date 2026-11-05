"""E41：E40 连续分数上的严格前向 HMM 与因果 EMA 自适应。

HMM：两状态对称 sticky 转移，log-domain 前向后验；第t帧只用第t帧分数和t-1后验。
EMA：均值从首帧初始化并逐帧更新，输出当前分数减当前因果 EMA 均值。
 dwell/温度与 EMA 半衰期均由其余56人的后段 bACC 做严格 LOSO 选择。
任何目标标签只用于最终评价，不参与该目标的分数生成或超参选择。

运行：.venv\Scripts\python.exe test/41_causal_adaptation/e41.py
"""
from __future__ import annotations

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
sys.path.insert(0, str(TESTS / "40_unlabeled_causal"))
import loso as L  # noqa: E402
import e40 as E40  # noqa: E402

DEV = "cuda"
SCORE_PATH = TESTS / "40_unlabeled_causal" / "e40_scores.npz"
E40_RESULT_PATH = TESTS / "40_unlabeled_causal" / "e40.json"
RESULT_PATH = HERE / "e41.json"
REPORT_PATH = HERE / "报告_E41_严格前向HMM与EMA.md"
DWELLS = (30, 60, 120, 300, 600, 1200)
TEMPERATURES = (0.25, 0.5, 1.0, 2.0, 4.0)
HALFLIVES = (30, 60, 120, 300, 600)
HMM_PARAMS = [(d, t) for d in DWELLS for t in TEMPERATURES]
EMA_PARAMS = list(HALFLIVES)


def hmm_forward_bank(score: torch.Tensor) -> torch.Tensor:
    """返回 (len(HMM_PARAMS),T) 后验 log-odds；无任何后向/全段操作。"""
    dwell = torch.tensor([p[0] for p in HMM_PARAMS], device=score.device, dtype=torch.float32)
    temperature = torch.tensor([p[1] for p in HMM_PARAMS], device=score.device, dtype=torch.float32)
    log_stay = torch.log1p(-1 / dwell)
    log_switch = -torch.log(dwell)
    logpost = torch.full((len(HMM_PARAMS), 2), -math.log(2), device=score.device)
    out = torch.empty((len(HMM_PARAMS), len(score)), device=score.device)
    for t in range(len(score)):
        z = score[t] / temperature
        emission = torch.stack((-torch.nn.functional.softplus(z),
                                -torch.nn.functional.softplus(-z)), dim=1)
        if t:
            predicted = torch.stack((
                torch.logaddexp(logpost[:, 0] + log_stay, logpost[:, 1] + log_switch),
                torch.logaddexp(logpost[:, 0] + log_switch, logpost[:, 1] + log_stay),
            ), dim=1)
        else:
            predicted = logpost
        unnormalized = predicted + emission
        logpost = unnormalized - torch.logsumexp(unnormalized, dim=1, keepdim=True)
        out[:, t] = logpost[:, 1] - logpost[:, 0]
    return out


def ema_bank(score: torch.Tensor) -> torch.Tensor:
    """返回 (len(EMA_PARAMS),T) 的 score-EMA(score)，均值含当前帧。"""
    halflife = torch.tensor(EMA_PARAMS, device=score.device, dtype=torch.float32)
    alpha = 1 - torch.exp(-math.log(2) / halflife)
    mean = score[0].expand(len(EMA_PARAMS)).clone()
    out = torch.empty((len(EMA_PARAMS), len(score)), device=score.device)
    out[:, 0] = 0
    for t in range(1, len(score)):
        mean.add_(alpha * (score[t] - mean))
        out[:, t] = score[t] - mean
    return out


def causal_prefix_check() -> dict:
    """修改第101帧以后的输入，前100帧输出必须完全不变。"""
    generator = torch.Generator(device=DEV).manual_seed(2026)
    x = torch.randn(180, device=DEV, generator=generator)
    changed = x.clone()
    changed[100:] = changed[100:] + 20
    out = {}
    for name, function in (("hmm", hmm_forward_bank), ("ema", ema_bank)):
        a, b = function(x), function(changed)
        out[name] = float((a[:, :100] - b[:, :100]).abs().max())
    return out


def metrics3(rows: list[dict]) -> np.ndarray:
    return np.asarray([[r["bacc"], r["auc"], r["f1"]] for r in rows], dtype=float)


def evaluate_bank(y: np.ndarray, bank: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    rows = [E40.classification_metrics(y[indices], bank[c, indices]) for c in range(len(bank))]
    return metrics3(rows), rows


def nested_select(ev: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """每个目标先排除其整行，再以其余56人的宏平均 bACC 选参数。"""
    out, selected = [], []
    for i in range(len(ev)):
        other_mean = np.nanmean(np.delete(ev[:, :, 0], i, axis=0), axis=0)
        chosen = int(np.nanargmax(other_mean))
        selected.append(chosen)
        out.append(ev[i, chosen])
    return np.asarray(out), selected


def mode_result(subs: list[str], values: np.ndarray, rows: list[dict], warmup: bool) -> dict:
    result = {**E40.summary(values), "per_subject": E40.per_subject(subs, values, rows)}
    if warmup:
        result["task_fpr"] = float(np.nanmean([r["task_fpr"] for r in rows]))
        result["valid_rate"] = float(np.mean([r["valid_rate"] for r in rows]))
        result["both_classes_subjects"] = int(sum(r["both_classes"] for r in rows))
    return result


def evaluate_family(subs: list[str], raw: dict, banks: dict, family: str) -> dict:
    params = HMM_PARAMS if family == "hmm" else EMA_PARAMS
    fixed_i = params.index((300, 1.0)) if family == "hmm" else params.index(300)
    fixed_name = "fixed_d300_t1" if family == "hmm" else "fixed_h300"
    result = {"parameters": [], "variants": {}}
    for variant in ("raw", "zc"):
        ev, test_rows = [], []
        for q in subs:
            y, blk = raw[q][1], raw[q][2]
            test = np.where(blk >= 4)[0]
            values, rows = evaluate_bank(y, banks[variant][q], test)
            ev.append(values)
            test_rows.append(rows)
        ev = np.stack(ev)
        selected_values, selected = nested_select(ev)
        modes = {
            fixed_name: ([fixed_i] * len(subs), ev[:, fixed_i], [test_rows[i][fixed_i] for i in range(len(subs))]),
            "loso": (selected, selected_values, [test_rows[i][selected[i]] for i in range(len(subs))]),
        }
        vr = {"test": {}, "warmup300": {}, "warmup600": {}}
        for mode, (indices, values, rows) in modes.items():
            vr["test"][mode] = mode_result(subs, values, rows, warmup=False)
            if mode == "loso":
                vr["test"][mode]["selected_params"] = {
                    q: ({"dwell": params[indices[i]][0], "temperature": params[indices[i]][1]}
                        if family == "hmm" else {"halflife": params[indices[i]]})
                    for i, q in enumerate(subs)
                }
            else:
                vr["test"][mode]["parameters"] = ({"dwell": 300, "temperature": 1.0}
                                                   if family == "hmm" else {"halflife": 300})
            for n, segment in ((300, "warmup300"), (600, "warmup600")):
                warm_rows = []
                for i, q in enumerate(subs):
                    nn = min(n, len(raw[q][1]))
                    ix = np.arange(nn)
                    warm_rows.append(E40.classification_metrics(raw[q][1][ix], banks[variant][q][indices[i], ix]))
                vr[segment][mode] = mode_result(subs, metrics3(warm_rows), warm_rows, warmup=True)
            m = vr["test"][mode]
            print(f"E41 {family:3s} {variant:3s} {mode:14s} bACC={m['bacc']:.4f} "
                  f"AUC={m['auc']:.4f} F1={m['f1']:.4f} n90={m['n90']} min={m['min']:.4f}", flush=True)
        result["variants"][variant] = vr
    result["parameters"] = ([{"dwell": d, "temperature": t} for d, t in HMM_PARAMS]
                             if family == "hmm" else [{"halflife": h} for h in EMA_PARAMS])
    return result


def write_report(result: dict, e40_result: dict):
    lines = [
        "# E41：严格前向 sticky HMM 与因果 EMA", "",
        "## 方法与合规性", "",
        "- 输入仅为 E40 保存的 raw/zc 连续分数。HMM 在 log-domain 中递推：第t帧只使用当前发射分数和第t−1帧归一化后验；没有 Viterbi 后向、双向平滑、完整段归一化或块号输入。",
        "- 两状态对称转移的自保持概率为 `1−1/dwell`；发射对数似然比为 `score/temperature`。dwell∈{30,60,120,300,600,1200}，temperature∈{.25,.5,1,2,4}。",
        "- EMA 输出 `score−EMA(score)`，均值从首帧初始化且只按时间向前更新；半衰期∈{30,60,120,300,600}。",
        "- `loso` 是主结果：每个目标的 dwell/温度或半衰期只由其余56人的后段 bACC 选择。fixed 仅是预设参照。目标标签仅用于最终评价。", "",
        f"- 因果前缀单元检查：HMM 最大差 {result['causal_prefix_check']['hmm']:.1f}，EMA 最大差 {result['causal_prefix_check']['ema']:.1f}（修改未来后，前100帧不变）。", "",
        "## 后段（块4–5）结果", "",
        "| 方法 | 输入 | 方式 | bACC | AUC | F1 | n≥.9 | min |", "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for family in ("hmm", "ema"):
        fixed_name = "fixed_d300_t1" if family == "hmm" else "fixed_h300"
        for variant in ("raw", "zc"):
            for mode in (fixed_name, "loso"):
                m = result["families"][family]["variants"][variant]["test"][mode]
                lines.append(f"| {family} | {variant} | {mode} | {m['bacc']:.4f} | {m['auc']:.4f} | "
                             f"{m['f1']:.4f} | {m['n90']} | {m['min']:.4f} |")
    lines += ["", "## 与 E40 严格 LOSO 的比较", "",
              "| 方法 | 输入 | ΔbACC | ΔAUC | ΔF1 | 结论 |", "|---|---|---:|---:|---:|---|"]
    for family in ("hmm", "ema"):
        for variant in ("raw", "zc"):
            m = result["families"][family]["variants"][variant]["test"]["loso"]
            b = e40_result["variants"][variant]["test"]["loso"]
            delta = [m[k] - b[k] for k in ("bacc", "auc", "f1")]
            conclusion = "有增益" if delta[0] > 0 else "无 bACC 增益"
            lines.append(f"| {family} | {variant} | {delta[0]:+.4f} | {delta[1]:+.4f} | {delta[2]:+.4f} | {conclusion} |")
    lines += ["", "## 录制开头预热段", "",
              "task 假阳性率为 relax 被误判为 task 的比例；指标只在有限评分帧上计算。", "",
              "| 帧段 | 方法 | 输入 | 方式 | 两类人数 | bACC | AUC | F1 | task假阳性率 | 有效率 |",
              "|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for segment in ("warmup300", "warmup600"):
        for family in ("hmm", "ema"):
            fixed_name = "fixed_d300_t1" if family == "hmm" else "fixed_h300"
            for variant in ("raw", "zc"):
                for mode in (fixed_name, "loso"):
                    m = result["families"][family]["variants"][variant][segment][mode]
                    lines.append(f"| {segment} | {family} | {variant} | {mode} | {m['both_classes_subjects']}/57 | "
                                 f"{m['bacc']:.4f} | {m['auc']:.4f} | {m['f1']:.4f} | "
                                 f"{m['task_fpr']:.4f} | {m['valid_rate']:.4f} |")
    lines += ["", "逐人指标与逐折所选参数见 `e41.json`。所有被试使用同一前向代码路径，034 无任何编号特判。", ""]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    start = time.time()
    subs, raw = L.load()
    saved = np.load(SCORE_PATH)
    if saved["subjects"].tolist() != subs:
        raise RuntimeError("E40 分数缓存不是按当前顺序完整保存的57人结果")
    prefix_check = causal_prefix_check()
    if any(v != 0 for v in prefix_check.values()):
        raise RuntimeError(f"因果前缀检查失败: {prefix_check}")
    banks = {family: {variant: {} for variant in ("raw", "zc")} for family in ("hmm", "ema")}
    for j, q in enumerate(subs):
        for variant in ("raw", "zc"):
            score = torch.tensor(saved[f"{variant}_{q}"], device=DEV, dtype=torch.float32)
            banks["hmm"][variant][q] = hmm_forward_bank(score).cpu().numpy()
            banks["ema"][variant][q] = ema_bank(score).cpu().numpy()
        print(f"forward [{j + 1:02d}/{len(subs)}] {q}", flush=True)
    families = {family: evaluate_family(subs, raw, banks[family], family) for family in ("hmm", "ema")}
    result = {
        "protocol": "E40连续分数；严格前向；超参按其余56人后段bACC做LOSO；目标标签只用于最终评价",
        "config": {"hmm_dwells": list(DWELLS), "hmm_temperatures": list(TEMPERATURES),
                   "ema_halflives": list(HALFLIVES), "hmm_output": "后验log-odds，>0判task",
                   "ema_output": "score-因果EMA(score)，>0判task", "elapsed_s": round(time.time() - start, 1)},
        "causal_prefix_check": prefix_check,
        "families": families,
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    e40_result = json.loads(E40_RESULT_PATH.read_text(encoding="utf-8"))
    write_report(result, e40_result)
    print(f"saved {RESULT_PATH} and {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
