"""协议时间表解码器的流式一致性测试。

验证三件事（不需要 EEG/训练数据）：
  1. 任意分块长度得到逐窗口完全相同的 (t_end, p_task)；
  2. 前缀不变：前 m 个窗口的输出只依赖前 m*win 个采样点；
  3. save/load 往返一致，且导出表的取值/长度/方向正确。

运行：py -3.11 test/10_projects/protocol_20261102/test_protocol_stream.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from protocol_timeline import TimelineDecoder, load_table, save_table  # noqa: E402

FS = 250.0
WIN = 1500


def run(chunks):
    dec = TimelineDecoder(np.linspace(0.05, 0.95, 341), u_assume=9)
    out = []
    for c in chunks:
        out.extend(dec.push(c))
    return out


def main():
    rng = np.random.default_rng(0)
    n = WIN * 9 + 777
    x = rng.normal(size=n)

    a = run([x[i:i + 137] for i in range(0, n, 137)])
    b = run([x[i:i + 250] for i in range(0, n, 250)])
    c = run([x])                          # 一次全给
    e = run([x[i:i + 1] for i in range(0, n, 1)])   # 逐样本给
    assert a == b == c == e, "chunk-size invariance failed"
    assert len(a) == n // WIN, (len(a), n // WIN)
    assert all(abs(t - (i + 1) * 6.0) < 1e-9 for i, (t, _) in enumerate(a)), "t_end must be 6 s grid"

    # 前缀不变：先喂一半再喂剩下的，前半段输出必须逐位一致
    half = (n // WIN // 2) * WIN
    p1 = run([x[:half]])
    dec = TimelineDecoder(np.linspace(0.05, 0.95, 341), u_assume=9)
    p2 = dec.push(x[:half])
    p2 = p2 + dec.push(x[half:])
    assert p1 == p2[:len(p1)], "prefix invariance failed"

    # 未来样本不影响已输出窗口
    y = x.copy()
    y[half:] += 1e3
    dec2 = TimelineDecoder(np.linspace(0.05, 0.95, 341), u_assume=9)
    q = dec2.push(y)
    assert q[:len(p1)] == p1, "future samples leaked into past outputs"

    # save / load 往返
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.json"
        save_table(p, np.linspace(0.05, 0.95, 341), u_assume=9)
        d = load_table(p)
        assert d.u_assume == 9 and d.table.size == 341 and np.allclose(d.table, np.linspace(0.05, 0.95, 341))

    # 导出的真实表
    real = load_table(ROOT / "src" / "protocol_timeline_table.json")
    t = real.table
    assert t.ndim == 1 and t.size >= 300 and np.all((t >= 0) & (t <= 1)), "table range"
    assert real.u_assume in (0, 9, 10), real.u_assume
    # 方向：任务块中心应显著高于静息块中心（块长约 50 窗）
    assert t[75:100].mean() > t[25:50].mean() + 0.5, "score direction or block layout unexpected"
    print(f"[ok] chunk invariance / prefix / future-isolation / save-load / exported table "
          f"(len={t.size}, u={real.u_assume}, p(task@blk2)={t[75:100].mean():.3f})")


if __name__ == "__main__":
    main()
