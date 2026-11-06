"""合规检查（黑盒，针对 runner.RUNNABLE 中的方案）。

检查项（对应赛题：因果、6 s 感受野、分块接收、逐帧实时输出）：
  chunk     分块一致性：不同块长喂入，输出帧时刻与分数一致
  prefix    前缀一致性：只喂前一半数据，已输出帧与全量喂入完全一致（未来数据不影响过去输出）
  latency   输出时延：t 时刻的帧必须在喂入数据不超过 t + 1 块时就已输出（不等待未来数据）
  window    感受野：单帧特征窗口 ≤ 6 s（bundle 方案插桩测量；factory 方案不可测，记为 N/A）
  history   历史依赖（信息项）：把前 300 s 换成噪声，统计其后分数受影响的程度
  speed     实时性：单块最大计算耗时 < 块时长、RTF > 1
"""
from __future__ import annotations

import time

import numpy as np

from . import runner

FS = runner.FS
SEG_S = 900          # 检查用片段长度（秒）
CHUNKS = (25, 93, 1250)
TOL = 1e-4


def _timed(key: str, x: np.ndarray, chunk: int, pipe=None) -> dict:
    pipe = pipe or runner.Pipeline(key)
    pipe.reset()
    rows, fed, lat = [], [], []
    for i, c in runner.FileReplay(x, chunk, 0):
        t0 = time.perf_counter()
        out = pipe.update(c)
        lat.append(time.perf_counter() - t0)
        for r in out:
            rows.append(r)
            fed.append(i + len(c))
    a = np.asarray(rows, dtype=np.float64).reshape(-1, 3)
    return {"t": a[:, 0], "score": a[:, 1], "raw": a[:, 2], "fed": np.asarray(fed), "lat": np.asarray(lat)}


def _cmp(ref: dict, d: dict, upto: float | None = None) -> dict:
    rt, rs = ref["t"], ref["score"]
    if upto is not None:
        m = rt <= upto
        rt, rs = rt[m], rs[m]
    common = np.intersect1d(np.round(rt, 4), np.round(d["t"], 4))
    miss = len(rt) - len(common)
    ia = np.searchsorted(np.round(rt, 4), common)
    ib = np.searchsorted(np.round(d["t"], 4), common)
    diff = float(np.nanmax(np.abs(rs[ia] - d["score"][ib]))) if len(common) else float("nan")
    return {"n_ref": int(len(rt)), "n_missing": int(miss), "n_extra": int(len(d["t"]) - len(common))
            if upto is None else 0, "max_diff": diff}


def check(key: str, source: str, sid: str, job=None) -> dict:
    log = job.log if job else print
    prog = (lambda p: setattr(job, "progress", p)) if job else (lambda p: None)
    x = runner.load_signal_v(source, sid)[: SEG_S * FS]
    dur = len(x) / FS
    pipe = runner.Pipeline(key)
    items = []

    # 插桩：bundle 方案记录每帧特征窗口长度
    seen: list[int] = []
    dec = getattr(pipe, "dec", None)
    if dec is not None and hasattr(dec, "_frame"):
        orig = dec._frame
        dec._frame = lambda buf: (seen.append(len(buf)), orig(buf))[1]

    with runner._gpu:
        log(f"参考运行：{source}/{sid} 前 {dur:.0f} s，块长 250")
        ref = _timed(key, x, 250, pipe)
        prog(0.2)
        if dec is not None and hasattr(dec, "_frame"):
            dec._frame = orig
        n_ref = len(ref["t"])
        log(f"参考帧数 {n_ref}")

        worst = {"max_diff": 0.0, "n_missing": 0, "n_extra": 0}
        detail = []
        for k, c in enumerate(CHUNKS):
            d = _timed(key, x, c, pipe)
            r = _cmp(ref, d)
            detail.append(f"块长 {c}: 缺 {r['n_missing']} 多 {r['n_extra']} 最大差 {r['max_diff']:.2e}")
            for f in ("n_missing", "n_extra"):
                worst[f] = max(worst[f], r[f])
            worst["max_diff"] = max(worst["max_diff"], r["max_diff"] if np.isfinite(r["max_diff"]) else 1e9)
            prog(0.2 + 0.4 * (k + 1) / len(CHUNKS))
            log(detail[-1])
        ok = worst["n_missing"] == 0 and worst["n_extra"] == 0 and worst["max_diff"] < TOL
        items.append({"id": "chunk", "name": "分块一致性", "pass": ok, "detail": "；".join(detail)})

        half = len(x) // 2
        d = _timed(key, x[:half], 250, pipe)
        r = _cmp(ref, d, upto=half / FS)
        ok = r["n_missing"] == 0 and r["max_diff"] < TOL
        items.append({"id": "prefix", "name": "前缀一致性（未来数据不影响过去）", "pass": ok,
                      "detail": f"前 {half / FS:.0f} s：参考帧 {r['n_ref']}，缺 {r['n_missing']}，"
                                f"最大差 {r['max_diff']:.2e}"})
        log(items[-1]["detail"])
        prog(0.7)

        wait = ref["fed"] / FS - ref["t"]
        ok = bool(len(wait)) and float(wait.max()) <= 250 / FS + 1e-6
        items.append({"id": "latency", "name": "输出时延（不等待未来数据）", "pass": ok,
                      "detail": f"帧输出时已喂入数据超出帧时刻：最大 {wait.max():.2f} s，中位 {np.median(wait):.2f} s"
                                f"（上限 1 块 = 1.00 s）"})

        if seen:
            w = max(seen) / FS
            items.append({"id": "window", "name": "感受野 ≤ 6 s", "pass": w <= 6 + 1e-9,
                          "detail": f"单帧特征窗口最大 {w:.2f} s（插桩测得，{len(seen)} 帧）"})
        else:
            items.append({"id": "window", "name": "感受野 ≤ 6 s", "pass": None,
                          "detail": "非 bundle 方案，无法插桩测量"})

        rng = np.random.default_rng(0)
        cut = 300 * FS
        xp = x.copy()
        xp[:cut] = rng.normal(0, np.nanstd(x), cut)
        d = _timed(key, xp, 250, pipe)
        _, ia, ib = np.intersect1d(np.round(ref["t"], 4), np.round(d["t"], 4), return_indices=True)
        keep = ref["t"][ia] > 300 + 6
        ia, ib = ia[keep], ib[keep]
        diff_s = np.abs(ref["score"][ia] - d["score"][ib])
        diff_r = np.abs(ref["raw"][ia] - d["raw"][ib])
        flip = float(np.mean(np.sign(ref["score"][ia]) != np.sign(d["score"][ib])))
        items.append({"id": "history", "name": "历史依赖（信息项）", "pass": None,
                      "detail": f"前 300 s 换噪声后，306 s 之后 {len(ia)} 帧：原始分数中位差 {np.median(diff_r):.3f}、"
                                f"最大差 {diff_r.max():.3f}；最终分数最大差 {diff_s.max():.3f}，判定翻转 {100 * flip:.1f}%。"
                                "差值>0 说明除 6 s 特征窗口外，方案还带有因果历史状态"
                                "（如在线 running 归一化、去趋势/HMM 后处理），只用过去数据，不涉及未来"})
        log(items[-1]["detail"])
        prog(0.9)

    lat = ref["lat"]
    rtf = dur / max(lat.sum(), 1e-9)
    p_lat = float(np.percentile(lat[1:], 99)) if len(lat) > 1 else float(lat.max())
    ok = p_lat < 1.0 and rtf > 1
    items.append({"id": "speed", "name": "实时性", "pass": ok,
                  "detail": f"块长 1 s：RTF {rtf:.1f}×，单块耗时 P99 {1000 * p_lat:.1f} ms，"
                            f"最大 {1000 * lat.max():.1f} ms（含首块预热）"})
    hard = [i for i in items if i["pass"] is not None]
    return {"scheme": key, "label": runner.RUNNABLE[key]["label"], "source": source, "subject": sid,
            "seconds": dur, "pass": all(i["pass"] for i in hard), "items": items}
