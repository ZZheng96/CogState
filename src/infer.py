from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from eeg_decoder import load_bdf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "test" / "17_compliant"))
from streaming import StreamingInference  # noqa: E402


class ForwardHMM:
    """两状态对称 sticky HMM；只递推当前发射分数和上一帧后验。"""

    def __init__(self, dwell: int = 300, temperature: float = 1.0):
        self.log_stay = math.log1p(-1 / dwell)
        self.log_switch = -math.log(dwell)
        self.temperature = temperature
        self.logpost = np.array([-math.log(2), -math.log(2)])
        self.started = False

    def update(self, score: float) -> float:
        z = score / self.temperature
        emission = np.array([-np.logaddexp(0, z), -np.logaddexp(0, -z)])
        if self.started:
            predicted = np.array([
                np.logaddexp(self.logpost[0] + self.log_stay, self.logpost[1] + self.log_switch),
                np.logaddexp(self.logpost[0] + self.log_switch, self.logpost[1] + self.log_stay),
            ])
        else:
            predicted = self.logpost
        unnormalized = predicted + emission
        self.logpost = unnormalized - np.logaddexp(unnormalized[0], unnormalized[1])
        self.started = True
        return float(self.logpost[1] - self.logpost[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="对 BDF 执行严格因果的分块 EEG 推理")
    parser.add_argument(
        "--model", type=Path, default=ROOT / "test" / "17_compliant" / "model_bundle.pt"
    )
    parser.add_argument("--input", type=Path, required=True, help="输入 BDF 文件")
    parser.add_argument("--output", type=Path, required=True, help="输出预测 CSV")
    parser.add_argument("--chunk-size", type=int, default=250, help="每次输入的采样点数")
    args = parser.parse_args()
    if args.chunk_size <= 0:
        raise ValueError("chunk-size 必须为正整数")

    bundle = torch.load(args.model, map_location="cpu", weights_only=False)
    if bundle.get("version") != 1:
        raise ValueError("不支持的模型格式版本")
    if float(bundle["win_s"]) > 6:
        raise ValueError("模型原始 EEG 窗口超过 6 秒")
    record = load_bdf(args.input)
    decoder = StreamingInference(bundle, fs=record["fs"], step=1.0)
    hmm = ForwardHMM(dwell=300, temperature=1.0)

    # E17 特征函数使用 BDF 原始伏特单位；无效采样以 NaN 触发因果缓冲重置。
    signal_v = np.asarray(record["signal_uv"], dtype=float) * 1e-6
    signal_v[~record["valid"]] = np.nan
    outputs = []
    for start in range(0, record["n_samples"], args.chunk_size):
        end = min(start + args.chunk_size, record["n_samples"])
        for time_s, score, _ in decoder.update(signal_v[start:end]):
            hmm_score = hmm.update(score)
            outputs.append({
                "sample_index": int(round(time_s * record["fs"])) - 1,
                "time_s": round(time_s, 6),
                "score_task": hmm_score,
                "prediction": "task" if hmm_score > 0 else "relax",
                "valid": 1,
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_index", "time_s", "score_task", "prediction", "valid"],
        )
        writer.writeheader()
        writer.writerows(outputs)
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "model": str(args.model),
        "engine": "E17_multiscale_LR_MLP",
        "chunk_size": args.chunk_size,
        "input_unit": "uV converted to V for feature extraction",
        "score_direction": "higher means task",
        "windows": len(outputs),
        "valid_windows": sum(item["valid"] for item in outputs),
        "task_windows": sum(item["prediction"] == "task" for item in outputs),
        "threshold": 0.0,
        "window_s": float(bundle["win_s"]),
        "feature_mode": "878_features_x_5_causal_scales",
        "postprocess": "forward_hmm_d300_t1",
        "protocol_prior": False,
        "protocol_weight": None,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
