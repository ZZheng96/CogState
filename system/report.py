"""批量评估与 md 报告导出（reports/）。指标一律经 system.evaluate 统一口径计算。"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from . import ROOT, dataset, evaluate, runner

REPORTS = ROOT / "reports"
# 离线结果加载器（由服务端注入 system/web/schemes.py）：(key, sid) -> {"t","score"} 或 None
OfflineLoader = Callable[[str, str], "dict | None"]


def evaluate_schemes(job, keys: list[str], source: str, subjects: list[str] | None, mode: str,
                     offline: OfflineLoader | None, labels: dict[str, str]) -> dict:
    """mode: auto（缓存优先，缺失才在线计算）| live（全部重新流式计算）| cache（只用缓存）"""
    subjects = subjects or dataset.subjects(source)
    total, k = len(keys) * len(subjects), 0
    out = {"source": source, "mode": mode, "subjects": subjects, "schemes": {}}
    for key in keys:
        rows, origin = [], {"live": 0, "runner_cache": 0, "offline": 0, "missing": 0}
        for sid in subjects:
            if job.cancel.is_set():
                return out
            d, how = None, "missing"
            if mode != "live":
                d = runner.load_cache(key, source, sid) if key in runner.RUNNABLE else None
                how = "runner_cache" if d is not None else how
                if d is None and offline and source == "builtin":
                    d = offline(key, sid)
                    how = "offline" if d is not None else how
            if d is None and mode != "cache" and key in runner.RUNNABLE:
                t0 = time.time()
                d, _ = runner.get_or_run(key, source, sid, force=True)
                how = "live"
                job.log(f"{key} {sid} 在线计算 {time.time() - t0:.1f} s")
            origin[how] += 1
            if d is not None:
                r = evaluate.subject(source, sid, np.asarray(d["t"], float), np.asarray(d["score"], float))
                r["origin"] = how
                rows.append(r)
            k += 1
            job.progress = k / total
        out["schemes"][key] = {"label": labels.get(key, key), "rows": rows, "mean": evaluate.aggregate(rows),
                               "origin": origin}
        m = out["schemes"][key]["mean"]
        job.log(f"{key}: " + (f"bACC {m['bacc']:.4f} AUC {m['auc']:.4f} F1 {m['f1']:.4f}（{m['n_subjects']} 人）"
                              if m else "无结果"))
    out["report"] = write_eval_md(out)
    return out


def _f(v, n=4):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{n}f}"


def write_eval_md(res: dict) -> str:
    keys = list(res["schemes"])
    ts = time.strftime("%Y%m%d_%H%M%S")
    L = [f"# 评估报告 {time.strftime('%Y-%m-%d %H:%M:%S')}", "",
         f"- 数据源：`{res['source']}`，被试 {len(res['subjects'])} 名，取数模式 `{res['mode']}`",
         "- 口径：方案只提供逐帧 (t, score)；标签由 BDF+events 独立生成（6 s 窗口完全落在单个 trial 内且不含缺失段）；"
         "阈值 0；bACC / AUC / F1 逐被试计算后宏平均。", "", "## 汇总", "",
         "| 方案 | bACC | AUC | F1 | 覆盖率 | bACC≥0.9 人数 | 被试数 | 来源 |", "|---|---|---|---|---|---|---|---|"]
    for k in keys:
        s, m = res["schemes"][k], res["schemes"][k]["mean"] or {}
        o = "，".join(f"{a} {b}" for a, b in s["origin"].items() if b)
        L.append(f"| {s['label']} (`{k}`) | {_f(m.get('bacc'))} | {_f(m.get('auc'))} | {_f(m.get('f1'))} | "
                 f"{_f(m.get('coverage'), 3)} | {m.get('n_ge_09', '—')} | {m.get('n_subjects', 0)} | {o} |")
    L += ["", "## 逐被试 bACC / AUC / F1", ""]
    L.append("| 被试 | " + " | ".join(f"`{k}`" for k in keys) + " |")
    L.append("|---" * (len(keys) + 1) + "|")
    by = {k: {r["subject"]: r for r in res["schemes"][k]["rows"]} for k in keys}
    for sid in res["subjects"]:
        cells = []
        for k in keys:
            r = by[k].get(sid)
            cells.append("—" if not r or "bacc" not in r else f"{r['bacc']:.3f} / {r['auc']:.3f} / {r['f1']:.3f}")
        L.append(f"| {sid} | " + " | ".join(cells) + " |")
    f = REPORTS / f"eval_{ts}_{'_vs_'.join(keys)[:80]}.md"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("\n".join(L) + "\n", encoding="utf-8")
    return str(f.relative_to(ROOT))


def write_compliance_md(res: dict) -> str:
    mark = {True: "通过", False: "**不通过**", None: "信息"}
    L = [f"# 合规检查 `{res['scheme']}` {time.strftime('%Y-%m-%d %H:%M:%S')}", "",
         f"- 方案：{res['label']}",
         f"- 数据：`{res['source']}/{res['subject']}` 前 {res['seconds']:.0f} s",
         f"- 结论：{'全部通过' if res['pass'] else '**存在不通过项**'}", "",
         "| 检查项 | 结果 | 说明 |", "|---|---|---|"]
    for i in res["items"]:
        L.append(f"| {i['name']} | {mark[i['pass']]} | {i['detail']} |")
    if "metrics" in res and res["metrics"]:
        m = res["metrics"]
        L += ["", f"该片段指标：bACC {_f(m.get('bacc'))}，AUC {_f(m.get('auc'))}，F1 {_f(m.get('f1'))}"]
    f = REPORTS / f"compliance_{time.strftime('%Y%m%d_%H%M%S')}_{res['scheme']}.md"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("\n".join(L) + "\n", encoding="utf-8")
    return str(f.relative_to(ROOT))
