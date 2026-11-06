"""导出提交包 + 自检。

提交包保持仓库相对布局（src/infer.py 按自身位置找 test/17_compliant 等目录），无需改代码：
  exports/<名称>/  src/… test/…（推理依赖闭包，动态追踪 import 得到） test/17_compliant/model_bundle.pt
                   requirements.txt  运行说明.md  manifest.json  自检报告.md
  exports/<名称>.zip
自检：在包目录内用子进程运行 src/infer.py（不加载仓库代码），
  1) 依赖自足：子进程实际加载的项目文件全部位于包内；
  2) 一致性：包输出 CSV 与系统 runner 结果逐帧比对；
  3) 指标：由 CSV 的 (time_s, score_task) 按黑盒口径独立计算 bACC/AUC/F1；
  4) 合规：对该方案运行 compliance.check。
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import ROOT, compliance, dataset, evaluate, report, runner

EXPORTS = ROOT / "exports"
INFER_DEFAULT = {"detrend": 1200, "dwell": 300, "post": True}  # 与 src/infer.py 默认后处理一致

_TRACE = r"""
import sys, json, runpy
from pathlib import Path
root = Path(sys.argv[1]).resolve()
sys.argv = [str(root / "src" / "infer.py")] + sys.argv[2:]
sys.path.insert(0, str(root / "src"))
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
finally:
    fs = sorted({str(Path(m.__file__).resolve()) for m in list(sys.modules.values()) if getattr(m, "__file__", None)})
    Path(sys.argv[sys.argv.index("--output") + 1] + ".modules.json").write_text(json.dumps(fs))
"""


def _project_files(mods: list[str], root: Path) -> list[Path]:
    venv = (ROOT / ".venv").resolve()
    out = []
    for f in map(Path, mods):
        if f.is_relative_to(venv) or f.suffix != ".py":
            continue
        if f.is_relative_to(root) or f.is_relative_to(ROOT):
            out.append(f)
    return out


def _run_infer(root: Path, bdf: Path, out_csv: Path, log) -> tuple[pd.DataFrame, list[Path], float]:
    tracer = out_csv.with_suffix(".trace.py")
    tracer.write_text(_TRACE, encoding="utf-8")
    cmd = [sys.executable, "-E", "-s", str(tracer), str(root), "--input", str(bdf), "--output", str(out_csv),
           "--chunk-size", "250"]
    tic = time.perf_counter()
    p = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, encoding="utf-8", errors="replace")
    dt = time.perf_counter() - tic
    tracer.unlink(missing_ok=True)
    if p.returncode != 0:
        log(p.stderr[-2000:])
        raise RuntimeError(f"infer.py 运行失败（返回码 {p.returncode}）")
    mf = Path(str(out_csv) + ".modules.json")
    files = [f for f in _project_files(json.loads(mf.read_text()), root) if f != tracer.resolve()]
    mf.unlink(missing_ok=True)
    return pd.read_csv(out_csv, encoding="utf-8-sig"), files, dt


def _sha(f: Path) -> str:
    return hashlib.sha256(f.read_bytes()).hexdigest()[:16]


def build(job, name: str, scheme: str, source: str = "builtin", subjects: list[str] | None = None) -> dict:
    dataset._check(name)
    spec = runner.RUNNABLE.get(scheme)
    if not spec or not spec["bundle"]:
        raise ValueError(f"方案 {scheme} 不是 bundle 方案，无法导出")
    diff = {k: spec[k] for k in INFER_DEFAULT if spec[k] != INFER_DEFAULT[k]}
    if diff:
        raise ValueError(f"方案 {scheme} 的后处理 {diff} 与 src/infer.py 默认（{INFER_DEFAULT}）不一致，导出后结果会不同")
    subjects = subjects or ["018"]
    for s in subjects:
        dataset.paths(source, s)
    out = EXPORTS / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    log = job.log

    # 1) 依赖闭包：在仓库里跑一次（短片段即可，用首个被试），追踪实际加载的项目文件
    log("追踪推理依赖 …")
    work = EXPORTS / f".work_{name}"
    work.mkdir(parents=True, exist_ok=True)
    bdf0, _ = dataset.paths(source, subjects[0])
    _, deps, _ = _run_infer(ROOT, bdf0, work / "trace.csv", log)
    deps = sorted({f.relative_to(ROOT) for f in deps})
    for rel in deps:
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, out / rel)
    bundle_rel = Path("test/17_compliant/model_bundle.pt")
    shutil.copy2(spec["bundle"], out / bundle_rel)
    for f in ("requirements.txt", "运行说明.md"):
        if (ROOT / f).exists():
            shutil.copy2(ROOT / f, out / f)
    log(f"复制 {len(deps)} 个代码文件 + 模型 {runner._rel(spec['bundle'])}")
    job.progress = 0.15

    # 2) 包内自检
    rows, checks = [], []
    leaked: set[str] = set()
    for k, sid in enumerate(subjects):
        if job.cancel.is_set():
            break
        bdf, _ = dataset.paths(source, sid)
        log(f"包内运行 infer.py：{source}/{sid}")
        df, files, dt = _run_infer(out, bdf, work / f"{sid}.csv", log)
        leaked |= {str(f) for f in files if not f.is_relative_to(out.resolve())}
        t, s = df["time_s"].to_numpy(float), df["score_task"].to_numpy(float)
        ref, _ = runner.get_or_run(scheme, source, sid)
        n = min(len(t), len(ref["t"]))
        same_t = len(t) == len(ref["t"]) and np.allclose(t, ref["t"], atol=1e-4)
        md = float(np.max(np.abs(s[:n] - ref["score"][:n]))) if n else float("nan")
        r = evaluate.subject(source, sid, t, s)
        r.update({"frames": len(t), "ref_frames": len(ref["t"]), "same_t": bool(same_t), "max_diff": md,
                  "wall_s": round(dt, 1)})
        rows.append(r)
        log(f"  帧 {len(t)}（系统 {len(ref['t'])}），最大分数差 {md:.2e}，bACC {r.get('bacc', float('nan')):.4f} "
            f"AUC {r.get('auc', float('nan')):.4f} F1 {r.get('f1', float('nan')):.4f}，耗时 {dt:.0f} s")
        job.progress = 0.15 + 0.55 * (k + 1) / len(subjects)
    checks.append({"name": "依赖自足", "pass": not leaked,
                   "detail": "子进程只加载包内项目文件" if not leaked else "加载了包外文件：" + "；".join(sorted(leaked))})
    ok = all(r["same_t"] and r["max_diff"] < 1e-3 for r in rows)
    checks.append({"name": "与系统结果一致", "pass": ok,
                   "detail": "；".join(f"{r['subject']}: 帧 {r['frames']}/{r['ref_frames']}，最大差 {r['max_diff']:.2e}" for r in rows)})

    log("合规检查 …")
    comp = compliance.check(scheme, source, subjects[0])
    comp["report"] = report.write_compliance_md(comp)
    checks.append({"name": "合规检查", "pass": comp["pass"], "detail": f"{source}/{subjects[0]}，报告 {comp['report']}"})
    job.progress = 0.95

    mean = evaluate.aggregate(rows)
    for d in list(out.rglob("__pycache__")):
        shutil.rmtree(d, ignore_errors=True)
    manifest = {"name": name, "scheme": scheme, "label": spec["label"], "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "bundle_source": runner._rel(spec["bundle"]),
                "files": {str(p.relative_to(out)).replace("\\", "/"): _sha(p) for p in sorted(out.rglob("*")) if p.is_file()},
                "entry": "python src/infer.py --input <BDF> --output <CSV> [--chunk-size 250]"}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    passed = all(c["pass"] for c in checks)
    md = _write_md(out, manifest, checks, rows, mean, passed, source)
    shutil.copy2(out / "自检报告.md", ROOT / md)
    shutil.rmtree(work, ignore_errors=True)
    zp = shutil.make_archive(str(out), "zip", root_dir=out)
    log(f"完成：{'自检通过' if passed else '自检未通过'}，{Path(zp).relative_to(ROOT)}")
    return {"name": name, "scheme": scheme, "pass": passed, "checks": checks, "rows": rows, "mean": mean,
            "dir": str(out.relative_to(ROOT)), "zip": str(Path(zp).relative_to(ROOT)), "report": md,
            "n_files": len(manifest["files"])}


def _write_md(out: Path, man: dict, checks, rows, mean, passed, source) -> str:
    f4 = report._f
    L = [f"# 提交包自检 `{man['name']}` {man['created']}", "",
         f"- 方案：{man['label']}（`{man['scheme']}`），模型来源 `{man['bundle_source']}`",
         f"- 入口：`{man['entry']}`",
         f"- 结论：{'全部通过' if passed else '**存在不通过项**'}", "",
         "| 检查项 | 结果 | 说明 |", "|---|---|---|"]
    L += [f"| {c['name']} | {'通过' if c['pass'] else '**不通过**'} | {c['detail']} |" for c in checks]
    L += ["", f"## 包输出独立计算的指标（数据源 `{source}`，阈值 0）", "",
          "| 被试 | bACC | AUC | F1 | 覆盖率 | 帧数 | 耗时 s |", "|---|---|---|---|---|---|---|"]
    L += [f"| {r['subject']} | {f4(r.get('bacc'))} | {f4(r.get('auc'))} | {f4(r.get('f1'))} | "
          f"{f4(r.get('coverage'), 3)} | {r['frames']} | {r['wall_s']} |" for r in rows]
    if mean:
        L.append(f"| **宏平均** | {f4(mean['bacc'])} | {f4(mean['auc'])} | {f4(mean['f1'])} | {f4(mean['coverage'], 3)} | | |")
    L += ["", "## 文件清单（sha256 前 16 位）", ""] + [f"- `{k}` {v}" for k, v in man["files"].items()]
    (out / "自检报告.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    rel = Path("reports") / f"export_{time.strftime('%Y%m%d_%H%M%S')}_{man['name']}.md"
    return str(rel)
