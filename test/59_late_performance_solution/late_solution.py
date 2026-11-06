"""E59: 后段性能的严格因果双时间尺度软适应筛选。

只读取 E53 LOSO 分数；block/标签仅用于离线评价，不进入后处理。
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TESTS = ROOT / "test"
SCORES = TESTS / "53_e17_post_loso" / "e53_scores.npz"
OUT_JSON = HERE / "late_solution.json"
OUT_CSV = HERE / "late_solution.csv"
OUT_MD = HERE / "报告_E59_后段性能软适应实验.md"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
REV = ("004", "009", "010", "050")
WATCH = ("004", "009", "010", "018", "050")

sys.path.insert(0, str(TESTS / "17_compliant"))
import loso  # noqa: E402


def imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E44 = imp("e44_e59", TESTS / "44_late_drift" / "e44.py")


def metrics(y, s):
    from sklearn.metrics import f1_score, roc_auc_score
    p = s > 0
    return {"bacc": float(0.5 * (p[y == 1].mean() + (1 - p[y == 0]).mean())),
            "auc": float(roc_auc_score(y, s)), "f1": float(f1_score(y, p))}


def groups():
    with (TESTS / "43_group_analysis" / "group_members.csv").open(encoding="utf-8-sig") as f:
        return {r["subject"]: r["group"] for r in csv.DictReader(f)}


def ema(x, hl):
    a = 1.0 - 0.5 ** (1.0 / hl)
    out = np.empty_like(x, dtype=float)
    m = float(x[0])
    out[0] = m
    for i in range(1, len(x)):
        m += a * (float(x[i]) - m)
        out[i] = m
    return out


def forward_hmm(x, dwell=300):
    stay = math.log1p(-1.0 / dwell)
    switch = -math.log(dwell)
    lp = np.array([-math.log(2.0), -math.log(2.0)], dtype=float)
    out = np.empty(len(x), dtype=float)
    for i, z in enumerate(x):
        em = np.array([-np.logaddexp(0.0, z), -np.logaddexp(0.0, -z)])
        if i:
            lp = np.array([np.logaddexp(lp[0] + stay, lp[1] + switch),
                           np.logaddexp(lp[0] + switch, lp[1] + stay)])
        u = lp + em
        lp = u - np.logaddexp(u[0], u[1])
        out[i] = lp[1] - lp[0]
    return out


def soft_score(raw, short_hl, weight, cap, threshold=None, persistence=0):
    """返回处理分数及严格因果诊断。统计量更新包含当前帧。"""
    long = ema(raw, 1200)
    short = ema(raw, short_hl)
    gap = short - long
    if threshold is None:
        weights = np.full(len(raw), weight, dtype=float)
        fire = np.zeros(len(raw), dtype=bool)
    else:
        scale = np.maximum(ema(raw * raw, 600) - ema(raw, 600) ** 2, 1e-6) ** 0.5
        active = np.abs(gap) / scale > threshold
        run = 0
        fire = np.zeros(len(raw), dtype=bool)
        for i, a in enumerate(active):
            run = run + 1 if a else 0
            fire[i] = run >= persistence
        weights = np.zeros(len(raw), dtype=float)
        target = 0.0
        up, down = 1.0 / 60.0, 1.0 / 120.0
        for i in range(len(raw)):
            target = weight if fire[i] else 0.0
            if i == 0:
                weights[i] = min(target, up)
                continue
            rate = up if target > weights[i - 1] else down
            weights[i] = weights[i - 1] + rate * (target - weights[i - 1])
    correction = np.clip(gap, -cap, cap)
    z = raw - long - weights * correction
    return z, {"trigger_rate": float(fire.mean()), "active_rate": float(fire.mean()),
               "mean_weight": float(weights.mean()), "max_weight": float(weights.max()),
               "max_correction": float(np.max(np.abs(weights * correction))),
               "raw_trigger_rate": float(fire.mean())}


def build_candidates(subs, scores):
    out = {"raw": {}, "hmm_d300": {}, "dt1200_hmm": {}}
    diag = {k: {} for k in out}
    for q in subs:
        raw = np.asarray(scores[q], dtype=float)
        out["raw"][q] = raw
        out["hmm_d300"][q] = forward_hmm(raw)
        z = raw - ema(raw, 1200)
        out["dt1200_hmm"][q] = forward_hmm(z)
        diag["raw"][q] = diag["hmm_d300"][q] = diag["dt1200_hmm"][q] = {
            "trigger_rate": 0.0, "mean_weight": 0.0, "max_correction": 0.0
        }
        for hl in (60, 120, 300, 600):
            for w in (0.10, 0.25, 0.50):
                for cap in (0.10, 0.25, 0.50):
                    name = f"dual_h{hl}_w{int(w*100):02d}_c{int(cap*100):02d}"
                    z, d = soft_score(raw, hl, w, cap)
                    out.setdefault(name, {})[q] = forward_hmm(z)
                    diag.setdefault(name, {})[q] = d
        for hl in (120, 300, 600):
            for w in (0.25, 0.50):
                for thr in (1.0, 1.5, 2.0):
                    for cap in (0.10, 0.25, 0.50):
                        name = f"trigger_h{hl}_w{int(w*100):02d}_t{thr:g}_p60_c{int(cap*100):02d}"
                        z, d = soft_score(raw, hl, w, cap, thr, 60)
                        out.setdefault(name, {})[q] = forward_hmm(z)
                        diag.setdefault(name, {})[q] = d
    return out, diag


def evaluate(subs, D, candidates, diag, grp):
    result = {}
    for name, scores in candidates.items():
        per = {}
        for q in subs:
            y, b = D[q][1], D[q][2]
            per[q] = {"full": metrics(y, scores[q]), "late45": metrics(y[b >= 4], scores[q][b >= 4]),
                      "trigger": diag[name][q]}
        def agg(qs, part):
            return {k: float(np.mean([per[q][part][k] for q in qs])) for k in ("bacc", "auc", "f1")}
        sections = {}
        for part in ("full", "late45"):
            sections[part] = {g: agg([q for q in subs if g == "all" or grp[q] == g], part)
                              for g in ("all", "high", "low")}
        sections["watch"] = {q: {"full": per[q]["full"], "late45": per[q]["late45"],
                                  "delta_late_bacc": per[q]["late45"]["bacc"]}
                              for q in WATCH if q in per}
        late_sorted = sorted(((q, per[q]["late45"]["bacc"]) for q in subs), key=lambda x: x[1])
        result[name] = {"metrics": sections,
                        "diagnostics": {"mean_trigger_rate": float(np.mean([per[q]["trigger"]["trigger_rate"] for q in subs])),
                                        "mean_weight": float(np.mean([per[q]["trigger"]["mean_weight"] for q in subs])),
                                        "max_correction": float(max(per[q]["trigger"]["max_correction"] for q in subs))},
                        "worst5": [{"subject": q, "late45_bacc": v,
                                    "delta_vs_baseline": None} for q, v in late_sorted[:5]],
                        "per_subject": {q: {"full": per[q]["full"], "late45": per[q]["late45"]} for q in subs}}
    base = result["dt1200_hmm"]
    base_late = base["metrics"]["late45"]["all"]["bacc"]
    base_full = base["metrics"]["full"]["all"]
    for item in result.values():
        for w in item["worst5"]:
            w["delta_vs_baseline"] = w["late45_bacc"] - base["per_subject"][w["subject"]]["late45"]["bacc"]
        m = item["metrics"]
        item["retain"] = bool(m["full"]["all"]["bacc"] >= 0.7774 and m["full"]["all"]["auc"] >= 0.8265 and
                                m["full"]["all"]["f1"] >= 0.7692 and m["late45"]["all"]["bacc"] >= base_late + 0.01 and
                                m["full"]["all"]["bacc"] >= base_full["bacc"] - 0.005)
    return result


def nested_select(subs, result):
    names = list(result)
    scores = np.array([[0.5 * (result[n]["per_subject"][q]["full"]["bacc"] +
                               result[n]["per_subject"][q]["late45"]["bacc"])
                        for q in subs] for n in names])
    chosen = []
    held = []
    for i, q in enumerate(subs):
        j = int(np.argmax(np.delete(scores, i, axis=1).mean(axis=1)))
        name = names[j]
        chosen.append(name)
        held.append((q, result[name]["per_subject"][q]))
    def agg(part, key):
        return {k: float(np.mean([r[1][part][key] for r in held])) for k in ("bacc", "auc", "f1")}
    return {"full": agg("full", "bacc"), "late45": agg("late45", "bacc"),
            "chosen_counts": {n: chosen.count(n) for n in names if n in chosen},
            "heldout_full": {k: float(np.mean([r[1]["full"][k] for r in held])) for k in ("bacc", "auc", "f1")},
            "heldout_late45": {k: float(np.mean([r[1]["late45"][k] for r in held])) for k in ("bacc", "auc", "f1")}}


def write_csv(result):
    fields = ["candidate", "scope", "part", "bacc", "auc", "f1", "trigger_rate", "mean_weight", "max_correction", "retain"]
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for name, r in result.items():
            for scope in ("all", "high", "low"):
                for part in ("full", "late45"):
                    w.writerow({"candidate": name, "scope": scope, "part": part, **r["metrics"][part][scope],
                                 "trigger_rate": r["diagnostics"]["mean_trigger_rate"],
                                 "mean_weight": r["diagnostics"]["mean_weight"],
                                 "max_correction": r["diagnostics"]["max_correction"],
                                 "retain": r["retain"]})


def write_report(result):
    base = result["dt1200_hmm"]["metrics"]
    lines = ["# E59 后段性能软适应实验", "", f"数据源：`{SCORES}`；设备：`{DEV}`。所有候选逐被试逐帧因果处理，HMM dwell=300。", "", "## 当前基线", "", "|范围|bACC|AUC|F1|", "|---|---:|---:|---:|"]
    for p, label in (("full", "全段"), ("late45", "块4–5")):
        x = base[p]["all"]; lines.append(f"|{label}|{x['bacc']:.4f}|{x['auc']:.4f}|{x['f1']:.4f}|")
    ranked = sorted(result.items(), key=lambda kv: kv[1]["metrics"]["late45"]["all"]["bacc"], reverse=True)
    lines += ["", "## 候选结果（按块4–5 bACC排序）", "", "|候选|全段 bACC/AUC/F1|块4–5 bACC/AUC/F1|低组块4–5 bACC/AUC/F1|触发率|判定|", "|---|---|---|---|---:|---|"]
    for name, r in ranked:
        f, l, ll = r["metrics"]["full"]["all"], r["metrics"]["late45"]["all"], r["metrics"]["late45"]["low"]
        fmt = lambda x: f"{x['bacc']:.4f}/{x['auc']:.4f}/{x['f1']:.4f}"
        lines.append(f"|{name}|{fmt(f)}|{fmt(l)}|{fmt(ll)}|{r['diagnostics']['mean_trigger_rate']:.4f}|{'可保留' if r['retain'] else '放弃'}|")
    lines += ["", "## 指定个体（最佳候选）", "", "|候选|004 late bACC|009 late bACC|010 late bACC|018 late bACC|050 late bACC|", "|---|---:|---:|---:|---:|---:|"]
    for name, r in ranked[:10]:
        vals = [r["per_subject"].get(q, {}).get("late45", {}).get("bacc", float("nan")) for q in WATCH]
        lines.append("|" + name + "|" + "|".join(f"{v:.4f}" for v in vals) + "|")
    lines += ["", "## 嵌套 LOSO 选择", "", "候选选择排除当前被试，选择分数为其余被试全段与块4–5 bACC 的平均；目标被试只用于最终汇总。详见 `late_solution.json` 的 `nested_select`。", "", "## 约束与结论", "", "- 门控只读取当前及历史分数；不读取标签、block 或未来数据。", "- block 4–5、分组和个体结果仅用于离线评价。", "- 只有同时满足全段保护线、块4–5提升至少0.01、低组后段改善和个体保护的候选才可进入真实 BDF 核验。", "- 触发率、平均权重和最大修正幅度已写入 JSON/CSV。"]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    subs, D = loso.load()
    data = np.load(SCORES)
    scores = {q: data[q].astype(float) for q in subs}
    candidates, diag = build_candidates(subs, scores)
    result = evaluate(subs, D, candidates, diag, groups())
    nested = nested_select(subs, result)
    OUT_JSON.write_text(json.dumps({"device": DEV, "causal": True, "hmm_dwell": 300,
                                    "nested_select": nested, "results": result}, ensure_ascii=False, indent=1), encoding="utf-8")
    write_csv(result)
    write_report(result)
    ranked = sorted(result.items(), key=lambda kv: kv[1]["metrics"]["late45"]["all"]["bacc"], reverse=True)
    for name, r in ranked[:12]:
        f, l = r["metrics"]["full"]["all"], r["metrics"]["late45"]["all"]
        print(f"{name:38s} full {f['bacc']:.4f}/{f['auc']:.4f}/{f['f1']:.4f} late {l['bacc']:.4f}/{l['auc']:.4f}/{l['f1']:.4f} retain={r['retain']}")
    print(f"candidates={len(result)} device={DEV} outputs={OUT_JSON.name},{OUT_CSV.name},{OUT_MD.name}")


if __name__ == "__main__":
    main()
