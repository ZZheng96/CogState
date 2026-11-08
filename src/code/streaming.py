"""单通道 EEG 认知状态分块流式推理入口（实时/在线 API）。

算法实现位于 core.py；本文件提供流式 API 与向后兼容的命令行。

用法（与 infer.py 单文件模式相同）：
  python code/streaming.py --input <record.bdf> [--events <events.csv>] [--output <result.csv>]
                           [--chunk 1.0] [--device auto|cpu|cuda] [--model model/model_bundle.pt] [--gate deploy_gate.json]

API：
  from streaming import StreamingInference
  eng = StreamingInference("model/model_bundle.pt", fs=250, step=1.0)   # 默认加载同目录 deploy_gate.json
  for t_end, score, state in eng.update(chunk): ...   # chunk 为任意长度的一维样本数组（伏特，同 BDF/mne）
  eng.mark_invalid()   # 缺失段/不连续处：清空 6 s 缓冲，保留 running 统计
  eng.reset()          # 换被试：清空全部状态
score = HMM 后验对数几率 log(P(task)/P(relax))；state = score>0（1=task，0=relax）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import (BUNDLE, GATE, StreamingInference, metrics, read_bdf, train_bundle,  # noqa: E402,F401
                  window_features)


def run(input, events=None, output=None, chunk=1.0, device=None, model=BUNDLE, gate=None):
    from infer import run as _run
    return _run(input, events, output, chunk, device, model, gate)


def main():
    from infer import main as _main
    _main()


if __name__ == "__main__":
    main()
