"""决赛验证系统后端核心：数据源、真值、黑盒评估、流式运行、合规检查、训练、导出。

算法代码全部位于 system/algorithms/（包内相对导入），不依赖 src/ 或 test/。
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "system" / "cache"

import numpy as _np  # noqa: E402

if not hasattr(_np, "in1d"):  # mne 1.4 与 numpy>=2
    _np.in1d = _np.isin  # type: ignore[attr-defined]
