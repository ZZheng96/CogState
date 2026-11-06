"""流式运行器：把数据源按块喂给黑盒解码器，逐帧产出 (t, score)，供 SSE 实时推送与缓存。

Decoder 协议：reset()；update(chunk_v) -> [(t_end 秒, score, state)]（chunk 单位伏特，NaN=缺失）。
Source 协议：可迭代，产出 (起始样本号, chunk_v)；当前实现 FileReplay（BDF 回放），
设备接入时实现同样接口的 DeviceSource 即可。
"""
from __future__ import annotations

import itertools
import threading
import time
from pathlib import Path

import numpy as np

from . import CACHE, ROOT, dataset, evaluate, truth

FS = 250
RUNNABLE: dict[str, dict] = {}


def register(key: str, *, label: str, bundle: Path, detrend: float | None = 1200, dwell: int = 300,
             post: bool = True) -> None:
    RUNNABLE[key] = {"key": key, "label": label, "bundle": Path(bundle), "detrend": detrend,
                     "dwell": dwell, "post": post}


register("current", label="当前方案 src/infer.py（E17 + 去趋势 + HMM）",
         bundle=ROOT / "test" / "17_compliant" / "model_bundle.pt")
register("e17_raw", label="E17 原始分数（无后处理）",
         bundle=ROOT / "test" / "17_compliant" / "model_bundle.pt", post=False)


def runnable() -> list[dict]:
    return [{"key": k, "label": v["label"], "bundle": str(v["bundle"].relative_to(ROOT))
             if v["bundle"].is_relative_to(ROOT) else str(v["bundle"])} for k, v in RUNNABLE.items()]


class Pipeline:
    """bundle 推理 + 因果后处理，整体满足 Decoder 协议。"""

    _bundles: dict[str, object] = {}

    def __init__(self, key: str):
        import torch
        from infer import PostProcessor
        from streaming import StreamingInference
        spec = RUNNABLE[key]
        p = str(spec["bundle"])
        if p not in self._bundles:
            self._bundles.clear()
            self._bundles[p] = torch.load(p, map_location="cpu", weights_only=False)
        self.spec = spec
        self.dec = StreamingInference(self._bundles[p], fs=FS, step=1.0)
        self._post_cls = PostProcessor
        self.reset()

    def reset(self) -> None:
        self.dec.reset()
        self.post = self._post_cls(detrend_halflife=self.spec["detrend"], dwell=self.spec["dwell"]) \
            if self.spec["post"] else None

    def update(self, chunk) -> list[tuple[float, float, float]]:
        """返回 [(t, score, raw)]。"""
        out = []
        for t, s, _ in self.dec.update(chunk):
            out.append((float(t), float(self.post.update(s)) if self.post else float(s), float(s)))
        return out


def load_signal_v(source: str, sid: str) -> np.ndarray:
    from eeg_decoder import load_bdf
    bdf, _ = dataset.paths(source, sid)
    rec = load_bdf(bdf)
    x = np.asarray(rec["signal_uv"], dtype=float) * 1e-6
    x[~rec["valid"]] = np.nan
    return x


class FileReplay:
    """BDF 文件回放。speed=0 表示不限速（尽快），1 表示实时。"""

    def __init__(self, x: np.ndarray, chunk: int = 250, speed: float = 1.0, fs: int = FS):
        self.x, self.chunk, self.speed, self.fs = x, int(chunk), float(speed), fs
        self.stop = threading.Event()

    def __iter__(self):
        t0 = time.perf_counter()
        for i in range(0, len(self.x), self.chunk):
            if self.stop.is_set():
                return
            j = min(i + self.chunk, len(self.x))
            if self.speed > 0:
                wait = t0 + j / self.fs / self.speed - time.perf_counter()
                if wait > 0 and self.stop.wait(wait):
                    return
            yield i, self.x[i:j]


def run_file(key: str, source: str, sid: str, chunk: int = 250) -> dict:
    """同步跑完一条记录（不限速），返回 t / score / raw。"""
    pipe = Pipeline(key)
    rows = [r for _, c in FileReplay(load_signal_v(source, sid), chunk, 0) for r in pipe.update(c)]
    a = np.asarray(rows, dtype=np.float64).reshape(-1, 3)
    return {"t": a[:, 0], "score": a[:, 1].astype(np.float32), "raw": a[:, 2].astype(np.float32)}


def cache_path(key: str, source: str, sid: str) -> Path:
    return CACHE / "runs" / key / source / f"{sid}.npz"


_ids = itertools.count(1)
RUNS: dict[str, "Run"] = {}
_gpu = threading.Lock()


class Run:
    """一次流式回放。frames 只追加；订阅方用游标 + Condition 增量读取。"""

    def __init__(self, key: str, source: str, sid: str, speed: float = 1.0, chunk: int = 250):
        if key not in RUNNABLE:
            raise ValueError(f"方案 {key} 不可运行")
        dataset.paths(source, sid)
        self.id = f"r{next(_ids)}"
        self.key, self.source, self.sid, self.speed, self.chunk = key, source, sid, float(speed), int(chunk)
        self.frames: list[dict] = []
        self.status, self.error, self.metrics = "starting", None, None
        self.stats = {"duration": None, "compute_s": 0.0, "max_latency_ms": 0.0, "rtf": None}
        self.cond = threading.Condition()
        self.src: FileReplay | None = None
        RUNS[self.id] = self
        threading.Thread(target=self._main, daemon=True).start()

    def info(self) -> dict:
        return {"id": self.id, "scheme": self.key, "source": self.source, "subject": self.sid,
                "speed": self.speed, "chunk": self.chunk, "status": self.status, "error": self.error,
                "n_frames": len(self.frames), "stats": self.stats, "metrics": self.metrics}

    def stop(self) -> None:
        if self.src:
            self.src.stop.set()

    def _push(self, **kw) -> None:
        with self.cond:
            for k, v in kw.items():
                setattr(self, k, v)
            self.cond.notify_all()

    def _main(self) -> None:
        try:
            self._push(status="loading")
            x = load_signal_v(self.source, self.sid)
            self.stats["duration"] = len(x) / FS
            with _gpu:
                pipe = Pipeline(self.key)
                self.src = FileReplay(x, self.chunk, self.speed)
                self._push(status="running")
                ys, ss, last = [], [], 0.0
                for _, c in self.src:
                    tic = time.perf_counter()
                    out = pipe.update(c)
                    dt = time.perf_counter() - tic
                    self.stats["compute_s"] += dt
                    self.stats["max_latency_ms"] = max(self.stats["max_latency_ms"], dt * 1e3)
                    if not out:
                        continue
                    t = np.array([o[0] for o in out])
                    y = truth.labels(self.source, self.sid, t)
                    new = [{"t": round(o[0], 3), "score": round(o[1], 4), "raw": round(o[2], 4), "y": int(v)}
                           for o, v in zip(out, y)]
                    ys.extend(y.tolist()); ss.extend(o[1] for o in out)
                    now = time.perf_counter()
                    m = None
                    if now - last > 0.5:
                        m, last = evaluate.metrics(np.array(ys), np.array(ss)), now
                    with self.cond:
                        self.frames.extend(new)
                        if m:
                            self.metrics = m
                        done_s = (self.frames[-1]["t"])
                        self.stats["rtf"] = round(done_s / max(self.stats["compute_s"], 1e-9), 1)
                        self.cond.notify_all()
            stopped = self.src.stop.is_set()
            if not stopped and self.frames:
                f = cache_path(self.key, self.source, self.sid)
                f.parent.mkdir(parents=True, exist_ok=True)
                np.savez(f, t=np.array([r["t"] for r in self.frames]),
                         score=np.array([r["score"] for r in self.frames], np.float32),
                         raw=np.array([r["raw"] for r in self.frames], np.float32))
            final = evaluate.subject(self.source, self.sid, np.array([r["t"] for r in self.frames]),
                                     np.array([r["score"] for r in self.frames])) if self.frames else None
            self._push(status="stopped" if stopped else "done", metrics=final)
        except Exception as exc:  # noqa: BLE001
            self._push(status="error", error=f"{type(exc).__name__}: {exc}")

    def follow(self, cursor: int = 0, timeout: float = 15.0):
        """产出 (新帧列表, 游标, 是否结束)；无新帧时最多阻塞 timeout 秒（用于 SSE 心跳）。"""
        while True:
            with self.cond:
                if cursor >= len(self.frames) and self.status in ("starting", "loading", "running"):
                    self.cond.wait(timeout)
                new = self.frames[cursor:]
                cursor = len(self.frames)
                ended = self.status not in ("starting", "loading", "running")
            yield new, cursor, ended
            if ended and not new:
                return
