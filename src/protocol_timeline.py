"""协议时间表解码器（流式、无 sklearn 依赖）。

用途：把 P5/P6 的"协议时间表"口径做成可直接部署的推理入口。
模型只有一个 1 维查表：P(task | 假定第 1 块起点之后的第 k 个 6 s 窗)，k = 已到达窗数 − u_assume。

输入：按时间顺序到达的单通道 EEG 采样点（µV），任意分块长度。
输出：每凑满 6 s 窗口返回一个 (t_end_s, p_task)；t_end_s 为窗口末端相对第一块已到达样本的秒数。

重要限制（必须随模型一起交付）：
  * 该模型**不使用 EEG 波形**，只使用"已到达样本数"与"实验范式为 6 块 R-T-R-T-R-T、首块 relax"这一先验。
  * u_assume 是"文件开头到第一块开始"的假定秒数/6；训练集 47/57 份记录的实测值为 60 s（u=10），
    其余 0–57 s。若隐藏测试集把文件裁到 trial 边界，应改用 u_assume=0。
  * 若隐藏测试集的块数、顺序或首块相位与训练集不同，本模型会系统性失效。

用法：
    from protocol_timeline import TimelineDecoder, load_table
    dec = TimelineDecoder(load_table("protocol_timeline_table.json"))
    for chunk in chunks:
        for t_end, p_task in dec.push(chunk):
            ...
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class TimelineDecoder:
    def __init__(self, table, u_assume=9, fs=250.0, window_s=6.0):
        self.table = np.asarray(table, dtype=float)
        if self.table.ndim != 1 or self.table.size < 2:
            raise ValueError("table must be a 1-D array of P(task) per window index")
        self.u_assume = int(u_assume)
        self.fs = float(fs)
        self.win = int(round(window_s * float(fs)))
        self.window_s = float(window_s)
        self.buf = np.empty(0, dtype=float)
        self.n_recv = 0
        self.n_win = 0

    def push(self, samples):
        """喂入任意长度采样点，返回本批完成的窗口 [(t_end_s, p_task), ...]。"""
        x = np.asarray(samples, dtype=float).ravel()
        if x.size:
            self.buf = np.concatenate([self.buf, x])
        out = []
        while self.buf.size >= self.win:
            self.buf = self.buf[self.win:]
            self.n_recv += self.win
            self.n_win += 1
            k = self.n_win - self.u_assume
            p = float(self.table[min(max(k, 0), self.table.size - 1)])
            out.append((self.n_recv / self.fs, p))
        return out

    def predict_at(self, t_end_s):
        """离线用：直接给窗口末端秒数取概率（不推进内部状态）。"""
        k = int(round(float(t_end_s) / self.window_s)) - self.u_assume
        return float(self.table[min(max(k, 0), self.table.size - 1)])


def save_table(path, table, u_assume, fs=250.0, window_s=6.0):
    payload = {"table": [float(v) for v in table], "u_assume": int(u_assume), "fs": float(fs),
               "window_s": float(window_s), "feature": "k = window_count - u_assume",
               "score_direction": "higher means task",
               "note": "protocol timeline prior; does NOT use EEG waveform"}
    Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_table(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return TimelineDecoder(d["table"], u_assume=d.get("u_assume", 9),
                           fs=d.get("fs", 250.0), window_s=d.get("window_s", 6.0))
