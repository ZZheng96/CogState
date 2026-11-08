"""E76 conditioned_shrink 门控：在 57 人训练集内（in-sample）拟合。

做法（与 runner._CausalGate 在线计算完全一致）：
1. 每个内置被试用系统内 E17 bundle 从首样本起流式推理（chunk=250，与 runner 相同），逐帧记录
   (t, lr=s_lr/std_lr, mlp=s_m/std_mlp, raw=lr+mlp)，缓存到 system/cache/e17_components/builtin/<sid>.npz；
2. 每被试因果计算 9 维门控特征 [lr, mlp, |lr-mlp|, 最近 120 帧(含当前) 三者均值, 三者总体标准差]；
3. 标签 system.truth.labels，取 y>=0 的帧，合并拟合
   LogisticRegression(C=0.1, class_weight="balanced", solver="liblinear", max_iter=300, random_state=0)；
4. 写 system/algorithms/models/e76_gate.json。

参考实现（只读，未导入）：test/76_conditioned_fusion/nested_loso/nested_loso.py gate_features/causal_stats，
test/87_full_train_inference/e87.py 训练集内门控拟合。

用法：python -m system.algorithms.fit_e76_gate [--workers 4]
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .. import CACHE, dataset, truth

HERE = Path(__file__).resolve().parent
OUT = HERE / "models" / "e76_gate.json"
COMP = CACHE / "e17_components" / "builtin"
WINDOW, SHRINK, C = 120, 0.5, 0.1


def comp_path(sid: str) -> Path:
    return COMP / f"{sid}.npz"


def load_components(sid: str) -> dict | None:
    from .. import runner
    f = comp_path(sid)
    if not f.exists():
        return None
    z = np.load(f)
    if str(z["version"]) != runner.version("e17_raw"):
        return None
    return {k: z[k] for k in ("t", "lr", "mlp", "raw")}


def stream_components(sid: str) -> dict:
    """流式推理一个被试并缓存分量（与 runner.run_file 同一喂数方式）。"""
    from .. import runner
    d = load_components(sid)
    if d is not None:
        return d
    pipe = runner.E17Components()
    pipe.reset()
    rows = []
    for _, c in runner.FileReplay(runner.load_signal_v("builtin", sid), 250, 0):
        rows.extend(pipe.update(c))
    a = np.asarray(rows, dtype=np.float64).reshape(-1, 4)
    d = {"t": a[:, 0], "lr": a[:, 1], "mlp": a[:, 2], "raw": a[:, 3].astype(np.float32)}
    COMP.mkdir(parents=True, exist_ok=True)
    np.savez(comp_path(sid), version=runner.version("e17_raw"), **d)
    return d


def _worker(sid: str) -> str:
    t0 = time.time()
    d = stream_components(sid)
    return f"{sid} frames={len(d['t'])} {time.time() - t0:.0f}s"


def gate_features(lr: np.ndarray, mlp: np.ndarray, w: int = WINDOW) -> np.ndarray:
    """逐帧复现 runner._CausalGate 的特征（deque maxlen=w，np.mean/np.std 总体标准差）。"""
    from collections import deque
    q = deque(maxlen=w)
    X = np.empty((len(lr), 9))
    for i, (a, b) in enumerate(zip(lr, mlp)):
        d = abs(a - b)
        q.append((a, b, d))
        arr = np.asarray(q)
        X[i] = np.r_[a, b, d, arr.mean(0), arr.std(0)]
    return X


def fit(sids: list[str]) -> dict:
    from sklearn.linear_model import LogisticRegression
    Xs, ys = [], []
    for sid in sids:
        d = load_components(sid)
        y = truth.labels("builtin", sid, d["t"])
        X = gate_features(d["lr"], d["mlp"])
        Xs.append(X[y >= 0])
        ys.append(y[y >= 0])
    X, y = np.concatenate(Xs), np.concatenate(ys).astype(int)
    g = LogisticRegression(C=C, class_weight="balanced", max_iter=300, solver="liblinear", random_state=0).fit(X, y)
    spec = {"coef": g.coef_[0].tolist(), "intercept": float(g.intercept_[0]), "window": WINDOW, "shrink": SHRINK,
            "C": C, "class_weight": "balanced", "solver": "liblinear",
            "features": ["lr", "mlp", "d", "mean_lr", "mean_mlp", "mean_d", "std_lr", "std_mlp", "std_d"],
            "n_frames": int(len(y)), "n_subjects": len(sids), "fitted_on": "57 subjects in-sample",
            "base": "system/algorithms/models/e17_bundle.pt"}
    OUT.write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    sids = dataset.subjects("builtin")
    todo = [s for s in sids if load_components(s) is None]
    print(f"[e76] subjects={len(sids)} to_stream={len(todo)}", flush=True)
    if todo:
        with ProcessPoolExecutor(a.workers) as ex:
            for i, msg in enumerate(ex.map(_worker, todo), 1):
                print(f"[stream {i}/{len(todo)}] {msg}", flush=True)
    spec = fit(sids)
    print(f"[e76] fitted n_frames={spec['n_frames']} coef={np.round(spec['coef'], 4).tolist()} "
          f"b={spec['intercept']:.4f} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
