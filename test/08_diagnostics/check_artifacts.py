"""检查文档引用的产物是否存在，以及 test 下有无未被文档提及的脚本。"""
# --- test layout bootstrap (generated 2026-11-02): locate test/ root, shared modules and cache ---
import sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_TEST = _HERE
while _TEST.name != "test" and _TEST.parent != _TEST:
    _TEST = _TEST.parent
CACHE_DIR = _TEST / "cache"
if str(_TEST) not in _sys.path:
    _sys.path.append(str(_TEST))
for _d in sorted(_TEST.iterdir()):
    if _d.is_dir() and _d.name not in {"cache", "results", "pretrained_cache", "reports", "experiments", "_orphans", "__pycache__"}:
        if str(_d) not in _sys.path:
            _sys.path.append(str(_d))
# --- end bootstrap ---

import re
from pathlib import Path

root = Path(__file__).resolve().parents[2]
md = (root/'研究报告.md').read_text(encoding='utf-8')
refs = sorted(set(re.findall(r'test/[A-Za-z0-9_.\-]+', md)))
missing = [r for r in refs if not (root/r).exists()]
print(f'文档提及 test/ 产物 {len(refs)} 个')
print('缺失:', missing if missing else '无')

all_py = {p.name for p in (root/'test').glob('*.py')}
named = set(re.findall(r'[A-Za-z0-9_.\-]+\.py', md)) | {Path(r).name for r in refs}
orphan = sorted(all_py - named)
print(f'\n未被文档提及的脚本 {len(orphan)} 个:')
for f in orphan:
    print('  ', f)
