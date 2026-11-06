"""数据源登记：内置 EEG_dataset/（source="builtin"）与导入目录 data_import/<名称>/<被试>/。

每个被试目录含一个 .bdf 与一个 *_events.csv，格式与赛题数据集一致。
"""
from __future__ import annotations

import re
from pathlib import Path

from . import ROOT

BUILTIN = ROOT / "EEG_dataset"
IMPORT_ROOT = ROOT / "data_import"
_NAME = re.compile(r"[A-Za-z0-9_\-]{1,64}")


def _check(name: str) -> str:
    if not _NAME.fullmatch(name):
        raise ValueError(f"非法名称：{name!r}")
    return name


def source_dir(source: str = "builtin") -> Path:
    return BUILTIN if source == "builtin" else IMPORT_ROOT / _check(source)


def sources() -> list[str]:
    out = ["builtin"]
    if IMPORT_ROOT.exists():
        out += sorted(p.name for p in IMPORT_ROOT.iterdir() if p.is_dir() and _NAME.fullmatch(p.name))
    return out


def subjects(source: str = "builtin") -> list[str]:
    d = source_dir(source)
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir() and any(p.glob("*.bdf")) and any(p.glob("*_events.csv")))


def paths(source: str, sid: str) -> tuple[Path, Path]:
    d = source_dir(source) / _check(sid)
    bdfs, csvs = sorted(d.glob("*.bdf")), sorted(d.glob("*_events.csv"))
    if not bdfs or not csvs:
        raise FileNotFoundError(f"{source}/{sid} 缺少 .bdf 或 *_events.csv")
    return bdfs[0], csvs[0]
