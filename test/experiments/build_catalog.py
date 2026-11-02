"""重建实验文件目录：扫描磁盘产物，生成 test/experiments/catalog.json 与 实验文件目录.md。

只做索引，不移动、不删除、不修改任何实验文件；不挑选“最佳”结果，不从 AUC/bACC 推算 F1。
用法（项目根目录）：py -3.11 test/experiments/build_catalog.py
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

SCAN_DIRS = ["src", "test", "vis", "database"]
ROOT_DOCS = sorted(p.name for p in ROOT.iterdir() if p.is_file() and p.suffix.lower() == ".md")

KIND = {
    ".json": "result",
    ".log": "log",
    ".png": "figure",
    ".pt": "weights",
    ".joblib": "weights",
    ".ckpt": "weights",
    ".npz": "array",
    ".npy": "array",
    ".pkl": "array",
    ".err": "stderr",
    ".out": "stdout",
    ".txt": "text",
}

# (族名, 关键词)；按顺序匹配，命中即停止；仅用于检索分类，不代表合规认证
TEST_FAMILY_RULES = [
    ("回归测试", ("test_causal", "test_preprocessing", "test_kalman_deep", "test_highdim_deep", "test_nonstationary", "time_leak_test")),
    ("去位置化与跨被试方向", ("depositioned",)),
    ("历史上下文与时序", ("context", "history", "detrend", "temporal", "switch", "warmup", "position_leak", "protocol")),
    ("深度表征与迁移", ("window6", "kalman", "transfer", "pretrained", "resnet", "token", "timefreq", "spd", "singlem", "longtrain", "teacher", "consistency", "highdim", "nonlinear", "raw_waveform", "deep", "graph_note")),
    ("特征工程与窗口", ("feature", "emg", "broadband", "v6", "robust", "sanitize", "split6", "short_window", "window_ablation", "window_eval")),
    ("伪迹/边界/泄漏诊断", ("boundary", "artifact", "bottleneck", "separability", "seed_noise", "oracle", "position", "leak", "audit")),
    ("database 构建", ("database", "explore_database")),
    ("校准、融合与在线自适应", ("self_training", "unsup", "reliability", "calibration", "normalization_guard", "ensemble", "merge", "combo", "mixed_family", "ab_combine", "ref_", "b_branch", "b_final", "causal", "online", "hmm", "blend", "score")),
]

DIR_FAMILIES = {
    "src": "src 核心源码",
    "vis": "可视化（vis）",
    "test/00_baseline": "00 基线、对齐与回归测试",
    "test/01_features": "01 特征工程与窗口协议",
    "test/02_models": "02 传统模型、融合与状态映射",
    "test/03_deep": "03 深度表征、迁移与卡尔曼",
    "test/04_temporal": "04 因果时序积分",
    "test/05_history": "05 历史上下文与因果记忆",
    "test/06_adaptive": "06 校准、分支与在线自适应",
    "test/07_depositioned": "07 去位置化与跨被试方向",
    "test/08_diagnostics": "08 伪迹、边界、泄漏与上限诊断",
    "test/09_database": "09 database 构建与实验",
    "test/10_projects": "10 新假设实验",
    "test/memory": "因果记忆（test/memory）",
    "test/theory": "理论诊断（test/theory）",
    "test/realorder": "真实顺序管线（test/realorder）",
    "test/audit": "审计复现（test/audit）",
    "test/literature": "文献方法复现（test/literature）",
    "test/results": "结果汇总（test/results）",
    "test/_orphans": "无脚本历史产物（test/_orphans）",
    "test/reports": "实验报告（test/reports）",
    "test/cache": "共享数组缓存（test/cache）",
}

# 产物前缀与脚本名不同、但有明确生成关系的显式别名（按前缀长度降序匹配）
ALIASES = {
    "highdim_": "test/03_deep/highdim_deep_ablation.py",
    "separability": "test/01_features/feature_separability.py",
    "results_comparison": "test/00_baseline/plot_results.py",
    "kalman_deep_": "test/03_deep/train_kalman_deep.py",
    "warmup_sweep": "test/06_adaptive/warmup_sweep_db.py",
    "optimization_": "test/00_baseline/optimize_real.py",
    "nonstationary_": "test/08_diagnostics/nonstationary_ablation.py",
    "pretrained_transfer_": "test/03_deep/transfer_pretrained.py",
    "ref_braindecode_": "test/03_deep/ref_braindecode_eval.py",
    "theory_p0_p1_observation_state": "test/theory/p0_p1_observation_state.py",
    "theory_p2_direction_calibration": "test/theory/p2_direction_calibration.py",
}

MISSING_RE = re.compile(r"""["']([^"'\n{}]+\.(?:npz|npy|pt|joblib|ckpt))["']""")
METRIC_HINT = re.compile(r"auc|bacc|balancedaccuracy|balanced_acc", re.I)
F1_HINT = re.compile(r"(?:^|[._])f1(?:$|[._](?:score|macro|micro|weighted))", re.I)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def kind_of(name: str) -> str:
    return KIND.get(Path(name).suffix.lower(), "other")


def family_for(script: Path) -> str:
    rel = script.relative_to(ROOT).as_posix()
    for prefix, name in DIR_FAMILIES.items():
        if rel.startswith(prefix + "/"):
            return name
    stem = script.stem
    for name, keys in TEST_FAMILY_RULES:
        if any(k in stem for k in keys):
            return name
    return "其他基线"


def family_for_dir(dir_path: Path) -> str:
    rel = dir_path.relative_to(ROOT).as_posix()
    for prefix, name in DIR_FAMILIES.items():
        if rel == prefix or rel.startswith(prefix + "/"):
            return name
    if rel.startswith("test/pretrained_cache"):
        return "外部预训练权重"
    return "未归组产物"


def mdlink(text: str, target: str) -> str:
    """路径含空格或括号时用尖括号包裹，避免生成断链。"""
    return f"[{text}](<{target}>)" if re.search(r"[ ()\[\]]", target) else f"[{text}]({target})"


def rel_link(target: str) -> str:
    """从索引目录指向项目内文件。"""
    return os.path.relpath(ROOT / target, OUT).replace("\\", "/")


def script_summary(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    m = re.match(r'\s*(?:"""|\'\'\')(.*?)(?:"""|\'\'\')', text, re.S)
    if m:
        line = " ".join(x.strip() for x in m.group(1).splitlines() if x.strip())
        return line[:120]
    for line in text.splitlines():
        if line.strip().startswith("#"):
            return line.strip("# ").strip()[:120]
    return ""


def is_metric_key(key: str) -> bool:
    if METRIC_HINT.search(key):
        return True
    # 排除 s0_f1 这类“流/折编号”写法，避免把 fold 1 误当 F1
    return bool(F1_HINT.search(key)) and not re.search(r"s\d+_f1", key, re.I)


def walk_json(node, prefix=""):
    """产出 (key_path, numeric_value)，仅保留含 auc/f1/bacc 的键路径。"""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from walk_json(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk_json(v, f"{prefix}[{i}]")
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        if prefix and is_metric_key(prefix):
            yield prefix, float(node)


def scan_dir(dir_path: Path):
    files = sorted((p for p in dir_path.iterdir() if p.is_file()), key=lambda p: p.name)
    scripts = [p for p in files if p.suffix == ".py"]
    artifacts = [p for p in files if p.suffix != ".py"]
    stems = sorted(((p.stem, p) for p in scripts), key=lambda x: (-len(x[0]), x[0]))
    assigned = {p.name: [] for p in scripts}
    unassigned = []
    for a in artifacts:
        match = None
        for stem, script in stems:
            if a.stem == stem or a.stem.startswith(stem + "_") or a.stem.startswith(stem + "-"):
                match = script
                break
        if match is None:
            for prefix, script_rel in sorted(ALIASES.items(), key=lambda kv: -len(kv[0])):
                if a.stem.startswith(prefix):
                    candidate = ROOT / script_rel
                    if candidate.is_file():
                        match = candidate
                    break
        if match is None and a.name == "results.json" and (ROOT / "src" / "train.py").is_file():
            match = ROOT / "src" / "train.py"
        if match is not None and match.parent == dir_path and match.name in assigned:
            assigned[match.name].append(a)
        elif match is not None:
            # 跨目录显式关联（如 results.json -> src/train.py）
            unassigned.append((a, match.relative_to(ROOT).as_posix()))
        else:
            unassigned.append((a, None))

    entries = []
    for s in scripts:
        arts = sorted(assigned[s.name], key=lambda p: p.name)
        entries.append({
            "id": s.relative_to(ROOT).as_posix(),
            "path": s.relative_to(ROOT).as_posix(),
            "summary": script_summary(s),
            "artifacts": [{"path": a.relative_to(ROOT).as_posix(), "kind": kind_of(a.name),
                          "size": a.stat().st_size, "sha256": sha256(a)} for a in arts],
            "status": "has_result_json" if any(a.suffix == ".json" for a in arts) else (
                "artifacts_only" if arts else "script_only"),
        })
    loose = [{"path": a.relative_to(ROOT).as_posix(), "kind": kind_of(a.name),
              "size": a.stat().st_size, "sha256": sha256(a), "suggested_script": owner}
             for a, owner in unassigned]
    return entries, loose


def main():
    assert ROOT.joinpath("EEG_dataset").is_dir(), f"unexpected root: {ROOT}"
    families = {}
    all_files = []
    for d in SCAN_DIRS:
        base = ROOT / d
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                if p.relative_to(ROOT).as_posix().startswith("test/experiments/"):
                    continue
                all_files.append(p)

    # 全局脚本前缀表（长前缀优先），用于跨目录产物来源提示
    global_stems = sorted(
        ((p.stem, p.relative_to(ROOT).as_posix()) for p in all_files if p.suffix == ".py"),
        key=lambda x: (-len(x[0]), x[1]),
    )

    # 按脚本所在目录分组；database/windows 只做汇总，不逐文件建条目
    for p in sorted((q for q in all_files if q.suffix == ".py"), key=lambda q: q.relative_to(ROOT).as_posix()):
        fam = family_for(p)
        families.setdefault(fam, {"name": fam, "dirs": set(), "scripts": [], "loose": []})
        families[fam]["dirs"].add(p.parent.relative_to(ROOT).as_posix())

    scanned_dirs = sorted({p.parent for p in all_files}, key=lambda q: q.relative_to(ROOT).as_posix())
    for d in scanned_dirs:
        if "database" in d.parts:
            continue
        entries, loose = scan_dir(d)
        for e in entries:
            fam = family_for(ROOT / e["path"])
            families[fam]["scripts"].append(e)
        for item in loose:
            if not item["suggested_script"]:
                # 跨目录前缀提示：脚本被移动/复制后仍能找回来源，不做物理关联
                name = Path(item["path"]).stem
                best = None
                for stem, sp in global_stems:
                    if name == stem or name.startswith(stem + "_") or name.startswith(stem + "-"):
                        best = sp
                        break
                item["suggested_script"] = best
            if item["suggested_script"]:
                fam = family_for(ROOT / item["suggested_script"])
            else:
                fam = family_for_dir(d)
            families.setdefault(fam, {"name": fam, "dirs": set(), "scripts": [], "loose": []})
            families[fam]["dirs"].add(d.relative_to(ROOT).as_posix())
            families[fam]["loose"].append(item)

    # 指标按文件截断：优先保留键路径较浅的汇总值；深层逐被试/自助法条目留在原始 JSON，
    # 只记录总数，避免索引体积超过实验产物本身。
    MAX_METRICS_PER_FILE = 40
    metrics = []
    metric_totals = {}
    for p in sorted((q for q in all_files if q.suffix == ".json"), key=lambda q: q.relative_to(ROOT).as_posix()):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rel = p.relative_to(ROOT).as_posix()
        entries = sorted(walk_json(data), key=lambda kv: (kv[0].count(".") + kv[0].count("["), kv[0]))
        metric_totals[rel] = len(entries)
        for key, val in entries[:MAX_METRICS_PER_FILE]:
            metrics.append({"file": rel, "key": key, "value": val})

    missing = []
    for p in sorted((q for q in all_files if q.suffix == ".py"), key=lambda q: q.relative_to(ROOT).as_posix()):
        text = p.read_text(encoding="utf-8", errors="replace")
        rel = p.relative_to(ROOT).as_posix()
        for ref in sorted(set(MISSING_RE.findall(text))):
            if "://" in ref:
                continue
            candidates = [p.parent / ref, ROOT / ref, ROOT / "test" / ref, ROOT / "test" / "cache" / ref, ROOT / "src" / ref]
            if not any(c.is_file() for c in candidates):
                missing.append({"script": rel, "reference": ref})

    docs = []
    doc_paths = [ROOT / name for name in ROOT_DOCS]
    doc_paths += sorted((ROOT / "test" / "reports").glob("*.md"))
    index_md = OUT / "实验方法索引.md"
    if index_md.is_file():
        doc_paths.append(index_md)
    for p in sorted(set(doc_paths), key=lambda q: q.relative_to(ROOT).as_posix()):
        docs.append({"path": p.relative_to(ROOT).as_posix(), "size": p.stat().st_size, "sha256": sha256(p)})

    catalog = {
        "schema_version": 1,
        "notes": [
            "本目录是磁盘产物的检索索引，不是统一口径排行榜。",
            "不同窗口、划分、顺序、历史自适应与 oracle 结果不可直接比较。",
            "smoke 结果仅用于运行检查；未保存的 F1 记为未记录，不由 AUC/bACC 推算。",
            "缺失依赖为静态字符串提示，非完备依赖图。",
            "metrics 每文件最多保留 40 条、优先浅层汇总键；深层逐被试/自助法数值请查原始 JSON；metric_totals 给出各文件实际条目数。",
        ],
        "scan_scope": SCAN_DIRS,
        "documents": docs,
        "families": [
            {
                "name": fam["name"],
                "dirs": sorted(fam["dirs"]),
                "scripts": sorted(fam["scripts"], key=lambda e: e["id"]),
                "unassigned": sorted(fam["loose"], key=lambda e: e["path"]),
            }
            for fam in sorted(families.values(), key=lambda f: f["name"])
        ],
        "metrics": sorted(metrics, key=lambda m: (m["file"], m["key"])),
        "metric_totals": dict(sorted(metric_totals.items())),
        "missing_references": sorted(missing, key=lambda m: (m["script"], m["reference"])),
        "summary": {
            "files_scanned": len(all_files),
            "py_scripts": sum(1 for p in all_files if p.suffix == ".py"),
            "json_results": sum(1 for p in all_files if p.suffix == ".json"),
            "metrics_entries": len(metrics),
            "missing_reference_count": len(missing),
            "database_window_files": len(list((ROOT / "database" / "windows").rglob("*"))) if (ROOT / "database" / "windows").is_dir() else 0,
        },
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# 实验文件目录（自动生成）",
        "",
        "由 [build_catalog.py](build_catalog.py) 扫描磁盘生成，请勿手工编辑。",
        "",
        "**用途与边界**：本页只回答“脚本在哪里、产物有哪些、是否缺文件”，不是统一口径排行榜。不同窗口、划分、顺序、历史自适应与 oracle 结果不可横排比较；smoke 仅作运行检查；未保存的 F1 记为未记录，不能由 AUC/bACC 推算。完整指标数值见 [catalog.json](catalog.json) 的 `metrics` 字段（附原始 JSON 键路径）。",
        "",
        f"扫描文件 {catalog['summary']['files_scanned']} 个（脚本 {catalog['summary']['py_scripts']}、JSON {catalog['summary']['json_results']}）；`database/windows` 汇总 {catalog['summary']['database_window_files']} 项，不逐条列出。指标条目共 {catalog['summary']['metrics_entries']} 条（每文件最多 40 条，优先浅层汇总值；逐被试与自助法明细见原始 JSON）。",
        "",
        "## 文档入口",
        "",
        "| 文档 | 说明 |",
        "|---|---|",
    ]
    for d in docs:
        lines.append(f"| {mdlink(d['path'], rel_link(d['path']))} | 项目文档 |")
    lines.append("")

    for fam in catalog["families"]:
        lines += [f"## {fam['name']}", "", f"目录：{', '.join(f'`{x}`' for x in fam['dirs']) or '—'}", ""]
        if fam["scripts"]:
            lines += ["| 脚本 | 脚本说明 | 结果/日志/权重（同前缀） | 状态 |", "|---|---|---|---|"]
            for s in fam["scripts"]:
                links = "<br>".join(
                    mdlink(Path(a["path"]).name, rel_link(a["path"])) for a in s["artifacts"]
                ) or "—"
                desc = s["summary"].replace("|", "\\|")
                lines.append(f"| {mdlink(Path(s['path']).name, rel_link(s['path']))} | {desc} | {links} | `{s['status']}` |")
            lines.append("")
            no_res = [s["path"] for s in fam["scripts"] if s["status"] == "script_only"]
            if no_res:
                lines += ["无任何同前缀产物（可能未运行、输出已删或另行命名）：", ""]
                lines += [f"- `{x}`" for x in no_res]
                lines.append("")
        if fam["unassigned"]:
            lines += ["未匹配到同目录脚本的产物：", "", "| 文件 | 建议关联 |", "|---|---|"]
            for a in fam["unassigned"]:
                lines.append(f"| {mdlink(Path(a['path']).name, rel_link(a['path']))} | {a['suggested_script'] or '未关联'} |")
            lines.append("")

    lines += ["## 脚本中引用但磁盘缺失的数组/权重", ""]
    if catalog["missing_references"]:
        lines += ["静态字符串提示，仅覆盖显式路径；`.npz/.npy` 被 `.gitignore` 忽略，缺失可能只是本机未保留缓存。", "",
                  "| 脚本 | 引用 |", "|---|---|"]
        for m in catalog["missing_references"]:
            lines.append(f"| `{m['script']}` | `{m['reference']}` |")
    else:
        lines.append("未发现。")
    lines.append("")
    (OUT / "实验文件目录.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(catalog["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()

