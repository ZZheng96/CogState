"""57 人全量训练（训练集内）指标重算：current / e17_raw / e76_gated。

三个方案共用同一 E17 流式推理，差别只在后处理，因此：
1. 读取 fit_e76_gate 产出的逐帧分量缓存 system/cache/e17_components/builtin/<sid>.npz；
2. 用 runner 中同一套 PostProcessor / _CausalGate 逐帧派生三个方案的 (t, score, raw)，写入 runner 运行缓存；
3. 抽检 --check 个被试：runner.run_file 真实流式重跑，与派生结果逐元素比较（要求完全一致）；
4. evaluate.subject / aggregate 汇总，写 system/cache/metrics_57.json。

--live：不派生，直接 runner.get_or_run（缓存缺失时流式重跑）。

用法：python -m system.recompute_metrics [--check 018] [--live]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from . import CACHE, dataset, evaluate, runner
from .algorithms import fit_e76_gate as G

KEYS = ("current", "e17_raw", "e76_gated")
OUT = CACHE / "metrics_57.json"


def derive(sid: str) -> dict[str, dict]:
    from .algorithms.postprocess import PostProcessor
    d = G.stream_components(sid)
    t, lr, mlp, raw = d["t"], d["lr"], d["mlp"], d["raw"]
    post, post76, gate = PostProcessor(1200, 300), PostProcessor(1200, 300), runner._CausalGate(runner.E76_GATE)
    cur, g, e76 = np.empty(len(t)), np.empty(len(t)), np.empty(len(t))
    for i in range(len(t)):
        r = float(raw[i])
        cur[i] = post.update(r)
        g[i] = gate.update(float(lr[i]), float(mlp[i]))
        e76[i] = post76.update(g[i])
    return {"current": {"t": t, "score": cur, "raw": raw},
            "e17_raw": {"t": t, "score": raw, "raw": raw},
            "e76_gated": {"t": t, "score": e76, "raw": g}}


def check(sid: str, der: dict[str, dict]) -> dict[str, bool]:
    ok = {}
    for k in KEYS:
        r = runner.run_file(k, "builtin", sid)
        ok[k] = all(np.array_equal(np.asarray(r[f]), np.asarray(der[k][f], dtype=np.asarray(r[f]).dtype))
                    for f in ("t", "score", "raw"))
        print(f"[check] {sid} {k} identical={ok[k]}", flush=True)
    return ok


def write_md(res: dict) -> None:
    agg, subs = res["aggregate"], res["subjects"]
    L = ["# 验证系统：57 人全量训练指标（训练集内）", "",
         "E17 bundle 与 E76 门控都用全部 57 人训练，再对同一批被试逐被试流式推理（chunk=250，阈值 0）。"
         "**这是训练集内指标，偏乐观，不代表对新被试的泛化能力。**", "",
         "代码与模型全部来自 `system/algorithms/`。复制后的特征、BDF 读取和流式分数与原实现逐元素一致"
         "（被试 018，见 `test/88_system_selfcontained/verify_equiv.py`）。", "",
         "## 宏平均", "", "|方案|AUC|F1|bACC|bACC≥0.9 人数|覆盖率|", "|---|---:|---:|---:|---:|---:|"]
    for k in KEYS:
        a = agg[k]
        L.append(f"|{k}|{a['auc']:.4f}|{a['f1']:.4f}|{a['bacc']:.4f}|{a['n_ge_09']}/{a['n_subjects']}|{a['coverage']:.4f}|")
    L += ["", "方案说明：", ""] + [f"- `{k}`：{res['labels'][k]}" for k in KEYS]
    L += ["", "## 一致性抽检", "", "派生缓存与 `runner.run_file` 真实流式重跑逐元素比较：", ""]
    L += [f"- {s}：" + "，".join(f"{k} {'一致' if v else '不一致'}" for k, v in c.items()) for s, c in res["checks"].items()]
    L += ["", "## 逐被试 bACC / AUC / F1", "", "|被试|" + "|".join(KEYS) + "|", "|---|" + "---:|" * len(KEYS)]
    for i, r in enumerate(subs[KEYS[0]]):
        cells = []
        for k in KEYS:
            q = subs[k][i]
            cells.append(f"{q['bacc']:.3f} / {q['auc']:.3f} / {q['f1']:.3f}" if "bacc" in q else "-")
        L.append(f"|{r['subject']}|" + "|".join(cells) + "|")
    L += ["", f"复现：`.venv\\Scripts\\python.exe -m system.recompute_metrics`（耗时 {res['elapsed_s']} s；"
          "分量缓存缺失时先运行 `python -m system.algorithms.fit_e76_gate`）。", ""]
    (CACHE.parent / "报告_57人全量训练指标.md").write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", nargs="*", default=["018"])
    ap.add_argument("--live", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    sids = dataset.subjects("builtin")
    rows = {k: [] for k in KEYS}
    checks = {}
    for i, sid in enumerate(sids, 1):
        if a.live:
            der = {k: runner.get_or_run(k, "builtin", sid)[0] for k in KEYS}
        else:
            der = derive(sid)
            for k in KEYS:
                runner.save_cache(k, "builtin", sid, der[k]["t"], der[k]["score"], der[k]["raw"])
            if sid in a.check:
                checks[sid] = check(sid, der)
        for k in KEYS:
            rows[k].append(evaluate.subject("builtin", sid, der[k]["t"], np.asarray(der[k]["score"], np.float32)))
        print(f"[{i}/{len(sids)}] {sid} " + " ".join(f"{k}={rows[k][-1].get('bacc', float('nan')):.4f}" for k in KEYS),
              flush=True)
    agg = {k: evaluate.aggregate(rows[k]) for k in KEYS}
    res = {"protocol": "57人全量训练（训练集内），逐被试流式推理", "aggregate": agg, "checks": checks,
           "subjects": rows, "labels": {k: runner.RUNNABLE[k]["label"] for k in KEYS},
           "versions": {k: runner.version(k) for k in KEYS}, "elapsed_s": round(time.time() - t0, 1)}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    write_md(res)
    for k in KEYS:
        print(k, json.dumps(agg[k]), flush=True)


if __name__ == "__main__":
    main()
