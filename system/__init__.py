"""决赛验证系统后端核心：数据源、真值、黑盒评估、流式运行、合规检查、训练、导出。"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "system" / "cache"
for _p in (ROOT / "src", ROOT / "test" / "17_compliant"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as _np  # noqa: E402

if not hasattr(_np, "in1d"):  # mne 1.4 与 numpy>=2
    _np.in1d = _np.isin  # type: ignore[attr-defined]
