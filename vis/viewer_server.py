"""EEG 波形查看器的本地数据服务（只读，不导出、不落盘）。

按需从 EEG_dataset 读取 BDF 原始采样率（250 Hz）数据，通过 HTTP 提供给
vis/viewer.html。BDF 读取结果按被试缓存在内存中，避免重复解析。

接口：
  GET /                          -> viewer.html
  GET /api/subjects              -> 被试索引（时长、trial 数、缺失段等）
  GET /api/subject/<sid>         -> 单被试元数据（事件、trial、BAD 区间、统计）
  GET /api/signal/<sid>?start=&count=
                                 -> 原始采样率波形（Float32 小端，单位 µV）
  GET /api/window/<sid>?end=&seconds=6
                                 -> 6 秒因果窗口：原始 + 1–45 Hz 因果滤波 + 频带功率 + 方案输出
  GET /api/envelope/<sid>?t0=&t1=&bins=&filt=0|1
                                 -> 任意时间跨度的 min/max 包络（点少时返回原始样本）
  GET /api/schemes               -> 已注册方案列表（vis/schemes.py）
  GET /api/pred/<sid>?scheme=    -> 方案逐秒决策分数；可计算的方案未缓存时后台计算
  GET /api/metrics?scheme=       -> 方案 bACC / AUC / F1 汇总（逐被试 + 宏平均）

用法：
  python vis/viewer_server.py
  python vis/viewer_server.py --port 8765 --no-browser
"""
from __future__ import annotations

import argparse
import json
import re
import threading
import webbrowser
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

# mne 1.4.2 与 numpy>=2 不兼容：np.in1d 已被移除
if not hasattr(np, "in1d"):
    np.in1d = np.isin  # type: ignore[attr-defined]

import mne  # noqa: E402
from scipy.signal import butter, sosfilt, welch  # noqa: E402

import model_eval  # noqa: E402  当前方案（src/infer.py）推理与评估
import schemes  # noqa: E402  方案注册表（可切换 / 对比）

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "EEG_dataset"
VIEWER = Path(__file__).resolve().parent / "viewer.html"
TZ = timezone(timedelta(hours=8))
MAX_CHUNK = 250 * 600  # 单次最多返回 10 分钟
BANDS = (("delta", 1, 4), ("theta", 4, 8), ("alpha", 8, 13), ("beta", 13, 30), ("gamma", 30, 45))


def parse_file_start(bdf_path: Path) -> datetime:
    m = re.search(r"_(\d{8})_(\d{6})\.(\d{3})_UTC", bdf_path.name)
    if not m:
        raise ValueError(f"无法从文件名解析时间：{bdf_path.name}")
    dt = datetime.strptime(m.group(1) + m.group(2) + m.group(3), "%Y%m%d%H%M%S%f")
    return dt.replace(tzinfo=TZ)


def load_events(csv_path: Path, file_start: datetime):
    """解析事件 CSV，时间换算为相对 BDF 首样本的秒。"""
    ev = pd.read_csv(csv_path)
    ts = pd.to_datetime(ev["local_time"], utc=True).astype("int64").to_numpy() / 1e9
    t0 = file_start.timestamp()
    events, trials, pending = [], [], None
    for i, row in ev.iterrows():
        t = float(ts[i]) - t0
        event = str(row["event"]).strip()
        phase = str(row["phase"]).strip()
        key = "" if pd.isna(row.get("key")) else str(row.get("key")).strip()
        events.append({"t": round(t, 4), "event": event, "phase": phase, "key": key})
        if event == "trial_start":
            pending = (t, phase)
        elif event == "trial_end" and pending is not None:
            trials.append({
                "start": round(pending[0], 4),
                "end": round(t, 4),
                "phase": pending[1],
                "label": 1 if pending[1] == "task" else 0,
            })
            pending = None
    return events, trials


def causal_bandpass(x: np.ndarray, sfreq: float, low: float = 1.0, high: float = 45.0) -> np.ndarray:
    """因果带通（sosfilt 单向），与赛题推理约束一致。"""
    sos = butter(4, [low / (sfreq / 2), high / (sfreq / 2)], btype="band", output="sos")
    return sosfilt(sos, x)


def band_powers(x: np.ndarray, sfreq: float) -> list[dict]:
    f, p = welch(x, fs=sfreq, nperseg=min(len(x), int(2 * sfreq)), noverlap=0, detrend="constant")
    total = float(np.trapezoid(p[(f >= 1) & (f < 45)], f[(f >= 1) & (f < 45)]) + 1e-12)
    out = []
    for name, lo, hi in BANDS:
        m = (f >= lo) & (f < hi)
        power = float(np.trapezoid(p[m], f[m]))
        out.append({"band": name, "range": f"{lo}-{hi}", "power": power, "relative": power / total})
    return out


class SubjectStore:
    """按需加载并缓存被试数据（原始采样率，单位 µV）。"""

    def __init__(self, dataset: Path, cache_size: int = 4):
        self.dataset = dataset
        self.cache_size = cache_size
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()

    def subjects(self) -> list[str]:
        return sorted(p.name for p in self.dataset.iterdir() if p.is_dir() and list(p.glob("*.bdf")))

    def paths_ok(self, sid: str) -> None:
        self._paths(sid)

    def _paths(self, sid: str) -> tuple[Path, Path]:
        if not re.fullmatch(r"\d{3}", sid):
            raise ValueError("非法被试编号")
        d = self.dataset / sid
        bdfs = sorted(d.glob("*.bdf"))
        csvs = sorted(d.glob("*_events.csv"))
        if not bdfs or not csvs:
            raise FileNotFoundError(f"被试 {sid} 数据不完整")
        return bdfs[0], csvs[0]

    def get(self, sid: str) -> dict:
        with self._lock:
            if sid in self._cache:
                self._cache.move_to_end(sid)
                return self._cache[sid]
        bdf, csv = self._paths(sid)
        file_start = parse_file_start(bdf)
        raw = mne.io.read_raw_bdf(str(bdf), preload=True, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        data = raw.get_data()[0].astype(np.float32) * np.float32(1e6)  # V -> µV
        bad = [
            {"start": round(float(o), 4), "end": round(float(o + d), 4)}
            for o, d, desc in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description)
            if "BAD_ACQ_SKIP" in str(desc)
        ]
        events, trials = load_events(csv, file_start)

        mask = np.ones(len(data), dtype=bool)
        for b in bad:
            mask[max(0, int(b["start"] * sfreq)):int(b["end"] * sfreq)] = False
        valid = data[mask] if mask.any() else data

        entry = {
            "subject": sid,
            "bdf": bdf.name,
            "events_csv": csv.name,
            "file_start": file_start.isoformat(),
            "sfreq": sfreq,
            "n_samples": int(len(data)),
            "duration": round(len(data) / sfreq, 3),
            "bad_intervals": bad,
            "bad_seconds": round(sum(b["end"] - b["start"] for b in bad), 3),
            "trials": trials,
            "events": events,
            "n_trials": len(trials),
            "relax_seconds": round(sum(t["end"] - t["start"] for t in trials if t["phase"] == "relax"), 2),
            "task_seconds": round(sum(t["end"] - t["start"] for t in trials if t["phase"] == "task"), 2),
            "n_key_press": sum(1 for e in events if e["event"] == "key_press"),
            "n_task_failure": sum(1 for e in events if e["event"] == "task_failure"),
            "stats": {
                "mean": round(float(valid.mean()), 4),
                "std": round(float(valid.std()), 4),
                "min": round(float(valid.min()), 4),
                "max": round(float(valid.max()), 4),
                "p1": round(float(np.percentile(valid, 1)), 4),
                "p99": round(float(np.percentile(valid, 99)), 4),
            },
            "_data": data,
        }
        with self._lock:
            self._cache[sid] = entry
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return entry

    def _mask(self, entry: dict) -> np.ndarray:
        if "_bad" not in entry:
            m = np.zeros(entry["n_samples"], dtype=bool)
            for b in entry["bad_intervals"]:
                m[max(0, int(b["start"] * entry["sfreq"])):int(np.ceil(b["end"] * entry["sfreq"]))] = True
            entry["_bad"] = m
        return entry["_bad"]

    def filtered(self, entry: dict) -> np.ndarray:
        """整段记录单向因果滤波（缺失段置 0 后滤波，再置 NaN），等价于带充分预热的在线滤波。"""
        if "_filt" not in entry:
            bad = self._mask(entry)
            x = entry["_data"].astype(np.float64)
            x[bad] = 0.0
            y = causal_bandpass(x, entry["sfreq"]).astype(np.float32)
            y[bad] = np.nan
            entry["_filt"] = y
        return entry["_filt"]

    def envelope(self, sid: str, t0: float, t1: float, bins: int, filt: bool) -> dict:
        """[t0, t1] 区间的 min/max 包络；点数不多时直接返回原始样本。缺失段为 null。"""
        entry = self.get(sid)
        sf = entry["sfreq"]
        x = self.filtered(entry) if filt else np.where(self._mask(entry), np.nan, entry["_data"]).astype(np.float32)
        i0 = max(0, int(np.floor(t0 * sf)))
        i1 = min(len(x), int(np.ceil(t1 * sf)) + 1)
        seg = x[i0:i1]
        bins = max(16, min(int(bins), 8000))
        rnd = lambda a: [None if not np.isfinite(v) else round(float(v), 3) for v in a]  # noqa: E731
        if len(seg) <= bins * 2:
            return {"mode": "raw", "start": i0 / sf, "dt": 1 / sf, "values": rnd(seg)}
        per = int(np.ceil(len(seg) / bins))
        pad = per * int(np.ceil(len(seg) / per)) - len(seg)
        s = np.concatenate([seg, np.full(pad, np.nan, dtype=np.float32)]).reshape(-1, per)
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                lo, hi = np.nanmin(s, 1), np.nanmax(s, 1)
        return {"mode": "env", "start": i0 / sf, "dt": per / sf, "min": rnd(lo), "max": rnd(hi)}

    def signal(self, sid: str, start: int, count: int) -> tuple[np.ndarray, int]:
        entry = self.get(sid)
        data = entry["_data"]
        start = max(0, min(int(start), len(data)))
        count = max(1, min(int(count), MAX_CHUNK, len(data) - start))
        return data[start:start + count], start

    def window(self, sid: str, end: float, seconds: float) -> dict:
        """以 end 为判断时刻的因果窗口：只覆盖 [end-seconds, end]。"""
        entry = self.get(sid)
        data, sfreq = entry["_data"], entry["sfreq"]
        n = int(round(seconds * sfreq))
        i1 = int(round(end * sfreq))
        i0 = i1 - n
        if i0 < 0:
            i0, i1 = 0, min(len(data), n)
        i1 = min(i1, len(data))
        seg = data[i0:i1].astype(np.float64)
        if len(seg) < 8:
            raise ValueError("窗口超出记录范围")
        filtered = np.nan_to_num(self.filtered(entry)[i0:i1].astype(np.float64))
        bad = any(b["start"] < i1 / sfreq and b["end"] > i0 / sfreq for b in entry["bad_intervals"])
        phase, inside = None, False
        for tr in entry["trials"]:
            if tr["start"] <= i1 / sfreq <= tr["end"]:
                phase = tr["phase"]
                inside = tr["start"] - 1e-6 <= i0 / sfreq
                break
        return {
            "within_trial": inside,
            "subject": sid,
            "start": round(i0 / sfreq, 4),
            "end": round(i1 / sfreq, 4),
            "seconds": round(len(seg) / sfreq, 4),
            "sfreq": sfreq,
            "raw": seg.astype(np.float32),
            "filtered": filtered.astype(np.float32),
            "bands": band_powers(filtered, sfreq),
            "rms": float(np.sqrt(np.mean(filtered ** 2))),
            "ptp": float(np.ptp(filtered)),
            "bad": bad,
            "phase": phase,
        }


STORE: SubjectStore | None = None
SUBJECT_ROWS: list[dict] = []
PENDING: set[str] = set()
_PRED_LOCK = threading.Lock()


def start_pred(sid: str) -> None:
    """后台计算被试的方案输出（逐被试串行，GPU 推理在 model_eval 内加锁）。"""
    with _PRED_LOCK:
        if sid in PENDING:
            return
        PENDING.add(sid)

    def run():
        try:
            model_eval.load(sid, compute_missing=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[pred] {sid} 失败：{exc}", flush=True)
        finally:
            PENDING.discard(sid)

    threading.Thread(target=run, daemon=True).start()


class Handler(BaseHTTPRequestHandler):
    server_version = "CogStateViewer/1.0"

    def log_message(self, fmt, *args):
        print(f"{self.address_string()} {fmt % args}", flush=True)

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self):  # noqa: N802
        url = urlparse(self.path)
        path = url.path
        try:
            if path in ("/", "/index.html", "/viewer.html"):
                self._send(200, VIEWER.read_bytes(), "text/html; charset=utf-8")
                return
            if path.startswith("/api/envelope/"):
                sid = path.rsplit("/", 1)[-1]
                q = parse_qs(url.query)
                self._json(STORE.envelope(
                    sid, float(q["t0"][0]), float(q["t1"][0]),
                    int(q.get("bins", ["1500"])[0]), q.get("filt", ["0"])[0] == "1",
                ))
                return
            if path == "/api/schemes":
                self._json({"default": "current", "schemes": schemes.info()})
                return
            if path.startswith("/api/pred/"):
                sid = path.rsplit("/", 1)[-1]
                key = parse_qs(url.query).get("scheme", ["current"])[0]
                STORE.paths_ok(sid)
                d = schemes.load(key, sid)
                if d is None:
                    can = schemes.get(key)["compute"] is not None
                    if can:
                        start_pred(sid)
                    self._json({"subject": sid, "scheme": key, "ready": False, "computable": can})
                    return
                self._json({
                    "subject": sid, "scheme": key, "ready": True,
                    "t": np.round(d["t"], 3).tolist(),
                    "score": np.round(d["score"], 4).tolist(),
                    "y": d["y"].tolist(),
                    "metrics": schemes.subject_metrics(key, sid),
                })
                return
            if path == "/api/metrics":
                key = parse_qs(url.query).get("scheme", ["current"])[0]
                s = schemes.summary(key, STORE.subjects())
                s["computing"] = sorted(PENDING)
                self._json(s)
                return
            if path == "/api/subjects":
                if SUBJECT_ROWS:
                    self._json({"n_subjects": len(SUBJECT_ROWS), "subjects": SUBJECT_ROWS})
                    return
                rows = []
                for sid in STORE.subjects():
                    try:
                        e = STORE.get(sid)
                    except Exception as exc:  # noqa: BLE001
                        rows.append({"subject": sid, "error": str(exc)})
                        continue
                    rows.append({k: e[k] for k in (
                        "subject", "duration", "n_trials", "relax_seconds", "task_seconds",
                        "bad_seconds", "n_key_press", "n_task_failure", "file_start", "sfreq",
                    )})
                SUBJECT_ROWS[:] = rows
                self._json({"n_subjects": len(rows), "subjects": rows})
                return
            if path.startswith("/api/subject/"):
                sid = path.rsplit("/", 1)[-1]
                e = STORE.get(sid)
                self._json({k: v for k, v in e.items() if not k.startswith("_")})
                return
            if path.startswith("/api/signal/"):
                sid = path.rsplit("/", 1)[-1]
                q = parse_qs(url.query)
                chunk, real_start = STORE.signal(
                    sid, int(q.get("start", ["0"])[0]), int(q.get("count", ["25000"])[0])
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(chunk.nbytes))
                self.send_header("X-Start", str(real_start))
                self.send_header("X-Count", str(len(chunk)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(chunk.astype("<f4").tobytes())
                return
            if path.startswith("/api/window/"):
                sid = path.rsplit("/", 1)[-1]
                q = parse_qs(url.query)
                end = float(q.get("end", ["0"])[0])
                seconds = float(q.get("seconds", ["6"])[0])
                w = STORE.window(sid, end, seconds)
                self._json({
                    **{k: v for k, v in w.items() if k not in ("raw", "filtered")},
                    "raw": w["raw"].tolist(),
                    "filtered": w["filtered"].tolist(),
                })
                return
            self._json({"error": "not found"}, 404)
        except FileNotFoundError as exc:
            self._json({"error": str(exc)}, 404)
        except Exception as exc:  # noqa: BLE001
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)


def main():
    global STORE
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--cache", type=int, default=12, help="内存中缓存的被试数")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    STORE = SubjectStore(args.dataset, args.cache)
    url = f"http://{args.host}:{args.port}/"
    print(f"EEG 查看器：{url}  （数据集 {args.dataset}，Ctrl+C 退出）", flush=True)
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
