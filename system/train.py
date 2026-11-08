"""数据导入、特征构建与重训。产出 models/<名称>.pt（+ .json 元信息），并自动注册为可在线运行方案。

特征与 system/algorithms/e17.py 流式推理完全一致：6 s 窗、1 s 步长，窗口须完整落在单个 trial 内且不含缺失段。
所有数据源（含内置）均由系统内 window_features 按需构建并缓存到 system/cache/feat/<source>/。
"""
from __future__ import annotations

import contextlib
import io
import json
import shutil
import time
from pathlib import Path

import numpy as np

from . import CACHE, ROOT, dataset, runner, truth

MODELS = ROOT / "models"


# ---------------------------------------------------------------- 导入
def import_dir(job, name: str, src: str) -> dict:
    """把 src 下的被试子目录（各含 .bdf + *_events.csv）复制到 data_import/<name>/。"""
    dst = dataset.source_dir(dataset._check(name))
    src_p = Path(src)
    if not src_p.is_dir():
        raise FileNotFoundError(f"目录不存在：{src}")
    cand = [p for p in sorted(src_p.iterdir()) if p.is_dir() and any(p.glob("*.bdf")) and any(p.glob("*_events.csv"))]
    if not cand and any(src_p.glob("*.bdf")):
        cand = [src_p]  # 直接给了单个被试目录
    if not cand:
        raise ValueError("未找到含 .bdf 与 *_events.csv 的被试目录")
    ok = []
    for i, p in enumerate(cand):
        if job.cancel.is_set():
            break
        sid = dataset._check(p.name)
        d = dst / sid
        d.mkdir(parents=True, exist_ok=True)
        for f in list(p.glob("*.bdf")) + list(p.glob("*_events.csv")):
            shutil.copy2(f, d / f.name)
        m = truth.meta(name, sid)
        n_task = sum(1 for t in m["trials"] if t[2] == 1)
        job.log(f"{sid}: {m['duration']:.0f} s，trial {len(m['trials'])}（task {n_task}），缺失段 {len(m['bad'])}")
        ok.append(sid)
        job.progress = (i + 1) / len(cand)
    return {"source": name, "subjects": ok}


# ---------------------------------------------------------------- 特征
def feat_path(source: str, sid: str) -> Path:
    return CACHE / "feat" / source / f"sub-{sid}.npz"


def build_features(source: str, sid: str, log=print) -> Path:
    f = feat_path(source, sid)
    if f.exists():
        return f
    from .algorithms.features import window_features
    t, y = truth.reference_grid(source, sid)
    x = runner.load_signal_v(source, sid)
    fs, n = runner.FS, int(round(truth.WIN_S * runner.FS))
    F, Y, T = [], [], []
    t0 = time.time()
    for ti, yi in zip(t, y):
        e = int(round(ti * fs))
        w = x[e - n:e]
        if len(w) == n and np.isfinite(w).all():
            F.append(window_features(w, float(fs))); Y.append(yi); T.append(ti)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, F=np.asarray(F, np.float32), y=np.asarray(Y, np.int8), t=np.asarray(T, np.float32))
    log(f"{source}/{sid}: 特征 {len(F)} 帧，{time.time() - t0:.0f} s")
    return f


# ---------------------------------------------------------------- 训练
class _JobOut(io.TextIOBase):
    def __init__(self, job):
        self.job = job

    def write(self, s):
        for line in s.splitlines():
            if line.startswith("[train]"):  # redirect_stdout 是进程级的，过滤掉其他线程（如访问日志）的输出
                self.job.log(line.strip())
        return len(s)


def train(job, name: str, items: list[tuple[str, str]]) -> dict:
    """items: [(source, sid)]。"""
    dataset._check(name)
    if name in ("current", "e17_raw", "e76_gated", "e76_deploy", "demo_noncausal"):
        raise ValueError("名称与内置方案冲突")
    files = {}
    for i, (src, sid) in enumerate(items):
        if job.cancel.is_set():
            return {}
        files[f"{src}/{sid}"] = build_features(src, sid, job.log)
        job.progress = 0.5 * (i + 1) / len(items)
    from .algorithms.e17 import train_bundle
    MODELS.mkdir(exist_ok=True)
    out = MODELS / f"{name}.pt"
    job.log(f"开始训练：{len(files)} 名被试")
    with runner._gpu, contextlib.redirect_stdout(_JobOut(job)):
        b = train_bundle(out_path=out, files=files)
    meta = {"name": name, "subjects": list(files), "train_bacc": b["train_bacc"],
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    out.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    register_model(out)
    job.progress = 1.0
    return {**meta, "bundle": str(out.relative_to(ROOT)), "scheme": f"m_{name}"}


def register_model(pt: Path) -> None:
    j = pt.with_suffix(".json")
    meta = json.loads(j.read_text(encoding="utf-8")) if j.exists() else {}
    n = len(meta.get("subjects", []))
    runner.register(f"m_{pt.stem}", label=f"训练模型 {pt.stem}（{n} 人，+去趋势+HMM）", bundle=pt)


def models() -> list[dict]:
    out = []
    for pt in sorted(MODELS.glob("*.pt")):
        j = pt.with_suffix(".json")
        meta = json.loads(j.read_text(encoding="utf-8")) if j.exists() else {"name": pt.stem}
        meta.update(scheme=f"m_{pt.stem}", bundle=str(pt.relative_to(ROOT)), n_subjects=len(meta.get("subjects", [])))
        meta.pop("subjects", None)
        out.append(meta)
    return out


for _pt in sorted(MODELS.glob("*.pt")):
    register_model(_pt)
