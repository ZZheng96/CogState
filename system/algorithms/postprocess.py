"""因果后处理：EMA 去趋势 + 两状态 sticky 前向 HMM。

复制自 src/infer.py: ForwardHMM, CausalDetrend, PostProcessor（逐行一致）。
"""
from __future__ import annotations

import math

import numpy as np


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


class CausalDetrend:
    """因果 EMA 去趋势：返回 score - m_t，m_t 含当前帧，首帧 m_0 = score_0。"""

    def __init__(self, halflife: float = 1200):
        self.alpha = 1 - 0.5 ** (1 / halflife)
        self.mean = None

    def update(self, score: float) -> float:
        if self.mean is None:
            self.mean = score
        else:
            self.mean = self.mean + self.alpha * (score - self.mean)
        return score - self.mean


class PostProcessor:
    """融合分数 -> [可选 dt 去趋势] -> ForwardHMM。halflife=None 为纯 HMM。"""

    def __init__(self, detrend_halflife: float | None = 1200, dwell: int = 300, temperature: float = 1.0):
        self.detrend = CausalDetrend(detrend_halflife) if detrend_halflife else None
        self.hmm = ForwardHMM(dwell=dwell, temperature=temperature)

    def update(self, score: float) -> float:
        if self.detrend is not None:
            score = self.detrend.update(score)
        return self.hmm.update(score)
