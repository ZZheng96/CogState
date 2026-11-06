"""后台任务（评估、合规检查、训练、导出）：线程执行，日志与进度可轮询。"""
from __future__ import annotations

import itertools
import threading
import time
import traceback
from typing import Callable

_ids = itertools.count(1)
JOBS: dict[str, "Job"] = {}


class Job:
    def __init__(self, kind: str, params: dict, fn: Callable[["Job"], dict]):
        self.id = f"j{next(_ids)}"
        self.kind, self.params = kind, params
        self.status, self.progress, self.result, self.error = "running", 0.0, None, None
        self.logs: list[str] = []
        self.t0, self.t1 = time.time(), None
        self.cancel = threading.Event()
        JOBS[self.id] = self
        threading.Thread(target=self._main, args=(fn,), daemon=True).start()

    def log(self, msg: str) -> None:
        self.logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def _main(self, fn) -> None:
        try:
            self.result = fn(self)
            self.status = "stopped" if self.cancel.is_set() else "done"
            self.progress = 1.0 if self.status == "done" else self.progress
        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            self.log(traceback.format_exc())
            self.status = "error"
        self.t1 = time.time()

    def info(self, logs_from: int = 0) -> dict:
        return {"id": self.id, "kind": self.kind, "params": self.params, "status": self.status,
                "progress": round(self.progress, 4), "error": self.error, "result": self.result,
                "elapsed": round((self.t1 or time.time()) - self.t0, 1),
                "logs": self.logs[logs_from:], "n_logs": len(self.logs)}

    def brief(self) -> dict:
        d = self.info(len(self.logs))
        d.pop("result"); d.pop("logs")
        return d
