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
from typing import Callable

import numpy as np

from . import CACHE, ROOT, dataset, evaluate, truth

FS = 250
RUNNABLE: dict[str, dict] = {}


def register(key: str, *, label: str, bundle: Path | None = None, detrend: float | None = 1200,
             dwell: int = 300, post: bool = True, factory: Callable[[], object] | None = None,
             deps: tuple[Path, ...] = ()) -> None:
    """bundle 方案：StreamingInference + PostProcessor；或 factory() 返回任意满足 Decoder 协议的对象。
    deps：额外参数文件，参与版本指纹。"""
    RUNNABLE[key] = {"key": key, "label": label, "bundle": Path(bundle) if bundle else None,
                     "detrend": detrend, "dwell": dwell, "post": post, "factory": factory, "deps": deps}


def _rel(p: Path | None) -> str | None:
    if p is None:
        return None
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def runnable() -> list[dict]:
    return [{"key": k, "label": v["label"], "bundle": _rel(v["bundle"])} for k, v in RUNNABLE.items()]


_HASH: dict[tuple[str, int, int], str] = {}


def _file_hash(f: Path) -> str:
    st = f.stat()
    k = (str(f), st.st_mtime_ns, st.st_size)
    if k not in _HASH:
        import hashlib
        _HASH[k] = hashlib.sha1(f.read_bytes()).hexdigest()[:12]
    return _HASH[k]


def version(key: str) -> str:
    """方案版本指纹（bundle/参数文件内容哈希），缓存失效判断用。"""
    files = [RUNNABLE[key]["bundle"], *RUNNABLE[key]["deps"]]
    v = "-".join(_file_hash(f) for f in files if f and f.exists())
    return v or "factory"


class Pipeline:
    """bundle 推理 + 因果后处理，整体满足 Decoder 协议。"""

    _bundles: dict[str, object] = {}

    def __new__(cls, key: str):
        f = RUNNABLE[key]["factory"]
        return f() if f else super().__new__(cls)

    def __init__(self, key: str):
        import torch
        from .algorithms.postprocess import PostProcessor
        from .algorithms.e17 import StreamingInference
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
    from .algorithms.io import load_bdf
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


def run_signal(key: str, x: np.ndarray, chunk: int = 250, pipe=None) -> dict:
    """同步把信号按块喂完（不限速），返回 t / score / raw / fed（输出该帧时已喂入的样本数）。"""
    pipe = pipe or Pipeline(key)
    pipe.reset()
    rows, fed = [], []
    for i, c in FileReplay(x, chunk, 0):
        for r in pipe.update(c):
            rows.append(r)
            fed.append(i + len(c))
    a = np.asarray(rows, dtype=np.float64).reshape(-1, 3)
    return {"t": a[:, 0], "score": a[:, 1].astype(np.float32), "raw": a[:, 2].astype(np.float32),
            "fed": np.asarray(fed, np.int64)}


def run_file(key: str, source: str, sid: str, chunk: int = 250) -> dict:
    """同步跑完一条记录（不限速），返回 t / score / raw。"""
    return run_signal(key, load_signal_v(source, sid), chunk)


def cache_path(key: str, source: str, sid: str) -> Path:
    return CACHE / "runs" / key / source / f"{sid}.npz"


def save_cache(key: str, source: str, sid: str, t, score, raw) -> None:
    f = cache_path(key, source, sid)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f, t=np.asarray(t, np.float64), score=np.asarray(score, np.float32),
             raw=np.asarray(raw, np.float32), version=version(key))


def load_cache(key: str, source: str, sid: str) -> dict | None:
    f = cache_path(key, source, sid)
    if not f.exists():
        return None
    z = np.load(f)
    if "version" not in z or str(z["version"]) != version(key):
        return None
    return {"t": z["t"], "score": z["score"], "raw": z["raw"]}


def get_or_run(key: str, source: str, sid: str, force: bool = False) -> tuple[dict, bool]:
    """返回 (结果, 是否新算)。新算时占用 GPU 锁并写缓存。"""
    if not force:
        d = load_cache(key, source, sid)
        if d is not None:
            return d, False
    with _gpu:
        d = run_file(key, source, sid)
    save_cache(key, source, sid, d["t"], d["score"], d["raw"])
    return d, True


class NonCausalDemo:
    """【违规反例】在 E17 原始分数上做 ±5 帧居中平滑：每帧要等 5 s 后的数据才输出。
    仅用于验证合规检查能识别"使用未来数据"。"""

    def __init__(self, half: int = 5):
        self.base, self.half = Pipeline("e17_raw"), half
        self.reset()

    def reset(self) -> None:
        self.base.reset()
        self.q: list[tuple[float, float]] = []

    def update(self, chunk):
        out = []
        for t, s, _ in self.base.update(chunk):
            self.q.append((t, s))
            if len(self.q) >= 2 * self.half + 1:
                c = self.q[-self.half - 1]
                v = float(np.mean([q[1] for q in self.q[-2 * self.half - 1:]]))
                out.append((c[0], v, c[1]))
        return out


E17_BUNDLE = ROOT / "system" / "algorithms" / "models" / "e17_bundle.pt"
E76_GATE = ROOT / "system" / "algorithms" / "models" / "e76_gate.json"
DEPLOY_GATE = ROOT / "system" / "algorithms" / "models" / "deploy_gate.json"   # = commit/pack/model/deploy_gate.json


class _CausalGate:
    """E76 门控的在线版：lr/mlp 分量 + |差| 及其最近 window 帧（含当前）的均值/总体标准差。"""

    def __init__(self, spec: dict):
        import json
        g = json.loads(Path(spec).read_text(encoding="utf-8")) if not isinstance(spec, dict) else spec
        self.coef, self.b = np.asarray(g["coef"], float), float(g["intercept"])
        self.w, self.shrink = int(g["window"]), float(g["shrink"])
        self.reset()

    def reset(self) -> None:
        from collections import deque
        self.q = deque(maxlen=self.w)

    def update(self, lr: float, mlp: float) -> float:
        d = abs(lr - mlp)
        self.q.append((lr, mlp, d))
        a = np.asarray(self.q)
        x = np.r_[lr, mlp, d, a.mean(0), a.std(0)]
        return self.shrink * float(x @ self.coef + self.b) + (1 - self.shrink) * (lr + mlp)


class E17Components:
    """E17 原始分数，同时逐帧给出 lr/std_lr、mlp/std_mlp 两个分量。update -> [(t, lr, mlp, raw)]。"""

    def __init__(self):
        import torch
        self.base = Pipeline("e17_raw")
        sc = self.base.dec.scorer
        comp: list[tuple[float, float]] = []

        def forward(X):
            lr = sc.lr
            s_lr = ((X - lr["mu"]) / lr["sd"]) @ lr["w"] + lr["b"]
            s_m = torch.stack([n((X - mu) / sd).squeeze(1) for mu, sd, n in sc.nets]).mean(0)
            a, b = s_lr / sc.std_lr, s_m / sc.std_mlp
            comp.append((float(a[0]), float(b[0])))
            return a + b

        sc.forward = torch.no_grad()(forward)
        self.comp = comp

    def reset(self) -> None:
        self.base.reset()
        self.comp.clear()

    def update(self, chunk) -> list[tuple[float, float, float, float]]:
        out = []
        for t, _, raw in self.base.update(chunk):
            lr, mlp = self.comp.pop(0)
            out.append((t, lr, mlp, raw))
        return out


class E76Pipeline:
    """E17 bundle 分别输出 lr/std_lr、mlp/std_mlp → E76 条件化半收缩门控 → EMA1200 + HMM300。"""

    def __init__(self, gate: Path = E76_GATE):
        from .algorithms.postprocess import PostProcessor
        self.base = E17Components()
        self.gate = _CausalGate(gate)
        self._post_cls = PostProcessor
        self.reset()

    def reset(self) -> None:
        self.base.reset()
        self.gate.reset()
        self.post = self._post_cls(detrend_halflife=1200, dwell=300)

    def update(self, chunk) -> list[tuple[float, float, float]]:
        out = []
        for t, lr, mlp, _ in self.base.update(chunk):
            g = self.gate.update(lr, mlp)
            out.append((t, float(self.post.update(g)), g))
        return out


register("e76_gated", label="E76 条件化半收缩门控（E17 LR/MLP 分量 + 因果门控 + EMA1200 + HMM300；门控在 57 人训练集内拟合）",
         bundle=E17_BUNDLE, factory=E76Pipeline, deps=(E76_GATE,))
register("e76_deploy", label="提交版（commit/pack 同算法：E17 + E74 OOF 门控 deploy_gate.json + EMA1200 + HMM300）",
         bundle=E17_BUNDLE, factory=lambda: E76Pipeline(DEPLOY_GATE), deps=(DEPLOY_GATE,))
register("current", label="当前方案（E17 + 去趋势 + HMM）", bundle=E17_BUNDLE)
register("e17_raw", label="E17 原始分数（无后处理）", bundle=E17_BUNDLE, post=False)
register("demo_noncausal", label="【反例】E17 + 双向平滑（违规，合规检查演示）", factory=NonCausalDemo)


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
                save_cache(self.key, self.source, self.sid, [r["t"] for r in self.frames],
                           [r["score"] for r in self.frames], [r["raw"] for r in self.frames])
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
