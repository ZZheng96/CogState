"""E60 cached-score attribution; exploratory, not fully nested validation."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location("e59_e60", ROOT / "test/59_late_performance_solution/late_solution.py")
e59 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e59)
KEYS = ("auc", "f1", "bacc")
PARTS = ("full", "early12", "late45")
SEED = 20261107
LIMITATION = ("Cached exploratory analysis only: excluding target subject from threshold/scale "
              "selection does not remove it from the models that produced other subjects' cached "
              "scores. Not fully nested or strict training-fold calibration. Repeated exploration "
              "on these same subjects is additionally selection-biased.")


def save(name, obj):
    (HERE / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def csvsave(name, rows):
    with (HERE / name).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def masks(block):
    return {"full": np.ones(len(block), dtype=bool), "early12": block <= 2, "late45": block >= 4}


def evaluate(scores, labels):
    per = {}
    for q, s in scores.items():
        y, b = labels[q]
        per[q] = {p: e59.metrics(y[m], s[m]) for p, m in masks(b).items() if len(np.unique(y[m])) == 2}
    macro = {p: {k: float(np.mean([v[p][k] for v in per.values() if p in v])) for k in KEYS} for p in PARTS}
    return {"macro": macro, "per_subject": per}


def paired(values):
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(SEED)
    bs = v[rng.integers(0, len(v), (5000, len(v)))].mean(axis=1)
    return {"mean_delta": float(v.mean()), "ci95": np.quantile(bs, [.025, .975]).tolist(),
            "improved": int((v > 1e-12).sum()), "worsened": int((v < -1e-12).sum()),
            "unchanged": int((np.abs(v) <= 1e-12).sum())}


def compare(a, b):
    out = {}
    for p in PARTS:
        out[p] = {}
        for k in KEYS:
            delta = {q: b["per_subject"][q][p][k] - v[p][k] for q, v in a["per_subject"].items() if p in v}
            out[p][k] = {**paired(list(delta.values())), "per_subject_delta": delta,
                          "worst5": sorted(delta.items(), key=lambda x: x[1])[:5]}
    return out


def warm_ema(x, startup):
    """n<=S: alpha=1/n; S<n<2S: alpha=(1-u)/n+u*a; n>=2S: alpha=a."""
    a = 1 - .5 ** (1 / 1200)
    out = np.empty(len(x))
    m = float(x[0])
    for i, value in enumerate(x):
        n = i + 1
        u = np.clip((n - startup) / startup, 0, 1)
        alpha = (1 - u) / n + u * a
        m += alpha * (value - m)
        out[i] = m
    return out


def main():
    # Match loso.load ordering without loading unused feature tensors into GPU memory.
    subs = sorted(p.stem[4:] for p in e59.loso.SLID.glob("sub-*.npz"))
    labels = {}
    for q in subs:
        with np.load(e59.loso.SLID / f"sub-{q}.npz") as d:
            order = np.argsort(d["t"])
            labels[q] = (d["y"][order].astype(int), d["blk"][order].astype(int))
    with np.load(e59.SCORES) as d:
        raw = {q: d[q].astype(float) for q in subs}
    assert len(subs) == 57
    for q in subs:
        assert len(raw[q]) == len(labels[q][0]) and np.isfinite(raw[q]).all()
        for m in masks(labels[q][1]).values():
            assert not m.any() or set(np.unique(labels[q][0][m])) == {0, 1}
    scores = {"raw": raw, "detrend": {q: x - e59.ema(x, 1200) for q, x in raw.items()}}
    scores["dt1200_hmm"] = {q: e59.forward_hmm(x) for q, x in scores["detrend"].items()}
    results = {n: evaluate(s, labels) for n, s in scores.items()}
    centers = []
    for name, sc in scores.items():
        for q, x in sc.items():
            y, b = labels[q]
            for p, m in masks(b).items():
                if not m.any():
                    continue
                x0, x1 = x[m & (y == 0)], x[m & (y == 1)]
                sd0, sd1 = x0.std(), x1.std()
                gap = x1.mean() - x0.mean()
                centers.append({"candidate": name, "subject": q, "part": p,
                                "n0": len(x0), "n1": len(x1), "mean0": float(x0.mean()), "mean1": float(x1.mean()),
                                "sd0": float(sd0), "sd1": float(sd1), "gap": float(gap),
                                "standardized_gap": float(gap / max(np.sqrt((sd0 ** 2 + sd1 ** 2) / 2), 1e-12)),
                                **results[name]["per_subject"][q][p]})
    transitions = {"raw_to_detrend": compare(results["raw"], results["detrend"]),
                   "detrend_to_hmm": compare(results["detrend"], results["dt1200_hmm"])}
    temporal = {n: {k: paired([v["late45"][k] - v["early12"][k] for v in r["per_subject"].values() if "early12" in v])
                    for k in KEYS} for n, r in results.items()}
    hmm_harmed = [q for q in subs if any(results["dt1200_hmm"]["per_subject"][q]["late45"][k] <
                                        results["detrend"]["per_subject"][q]["late45"][k] - 1e-12 for k in ("auc", "f1"))]
    raw_drop = temporal["raw"]["auc"]
    attribution = {"results": results.copy(), "transitions": transitions, "late_minus_early": temporal,
                   "hmm_harmed_late_subjects_auc_or_f1": hmm_harmed,
                   "raw_auc_drop_ci_below_zero": raw_drop["ci95"][1] < 0,
                   "limitations": LIMITATION, "missing_early12": [q for q in subs if "early12" not in results["raw"]["per_subject"][q]], "scope_n": {"full": 57, "early12": 56, "late45": 57, "paired_temporal": 56}, "bootstrap": {"seed": SEED, "replicates": 5000, "unit": "paired subject", "ci": "percentile, unadjusted"}}
    save("attribution.json", attribution)
    csvsave("attribution_subjects.csv", centers)
    print("ATTRIBUTION", json.dumps({n: r["macro"] for n, r in results.items()}), flush=True)
    print("TEMPORAL", json.dumps(temporal), "HMM_HARMED", len(hmm_harmed), hmm_harmed, flush=True)

    for name, hl, w, cap in (("dual_h120_w50_c50", 120, .5, .5), ("dual_h300_w25_c25", 300, .25, .25)):
        scores[name] = {q: e59.forward_hmm(e59.soft_score(x, hl, w, cap)[0]) for q, x in raw.items()}
        results[name] = evaluate(scores[name], labels)
    calibration, threshold_rows = {}, []
    grid = np.linspace(-12, 12, 2401)
    for name in ("dt1200_hmm", "dual_h120_w50_c50", "dual_h300_w25_c25"):
        matrix = []
        for q in subs:
            y = labels[q][0]
            x = scores[name][q]
            neg, pos = np.sort(x[y == 0]), np.sort(x[y == 1])
            matrix.append(.5 * (np.searchsorted(neg, grid, side="right") / len(neg) +
                                    1 - np.searchsorted(pos, grid, side="right") / len(pos)))
        matrix = np.asarray(matrix)
        shifted = {}
        for i, q in enumerate(subs):
            obj = np.delete(matrix, i, axis=0).mean(axis=0)
            tied = np.flatnonzero(obj >= obj.max() - 1e-12)
            j = tied[np.argmin(np.abs(grid[tied]))]
            threshold = float(grid[j])
            shifted[q] = scores[name][q] - threshold
            threshold_rows.append({"candidate": name, "subject": q, "threshold": threshold,
                                   "other_subject_macro_bacc": float(obj[j]), "grid_boundary": bool(j in (0, len(grid)-1))})
        calibration[name] = {"evaluated": evaluate(shifted, labels),
                             "delta_vs_zero": compare(results[name], evaluate(shifted, labels))}
    csvsave("thresholds.csv", threshold_rows)

    initialization = {}
    for startup in (60, 120):
        name = f"startup{startup}_hmm"
        scores[name] = {q: e59.forward_hmm(x - warm_ema(x, startup)) for q, x in raw.items()}
        results[name] = evaluate(scores[name], labels)
        initialization[name] = compare(results["dt1200_hmm"], results[name])
    print("CALIBRATION", json.dumps({n: r["evaluated"]["macro"] for n, r in calibration.items()}), flush=True)
    print("INITIALIZATION", json.dumps({n: results[n]["macro"] for n in initialization}), flush=True)

    fusion, scale_rows = {}, []
    if hmm_harmed:
        # Subject-balanced RMS scales, fitted excluding target; no target statistics/labels.
        ms = {n: np.array([np.mean(scores[n][q] ** 2) for q in subs]) for n in ("detrend", "dt1200_hmm")}
        for w in (.05, .1):
            name = f"fusion_w{int(w*100):02d}"
            scores[name] = {}
            for i, q in enumerate(subs):
                ds = float(np.sqrt(np.delete(ms["detrend"], i).mean()))
                hs = float(np.sqrt(np.delete(ms["dt1200_hmm"], i).mean()))
                scores[name][q] = (1-w) * scores["dt1200_hmm"][q] / hs + w * scores["detrend"][q] / ds
                scale_rows.append({"candidate": name, "subject": q, "detrend_scale": ds, "hmm_scale": hs, "weight": w})
            results[name] = evaluate(scores[name], labels)
            fusion[name] = compare(results["dt1200_hmm"], results[name])
        csvsave("fusion_scales.csv", scale_rows)
        # Select only between the two requested ratios on other subjects' full macro bACC.
        names = list(fusion)
        selected_per, selected_rows = {}, []
        for q in subs:
            chosen = max(names, key=lambda n: np.mean([results[n]["per_subject"][p]["full"]["bacc"] for p in subs if p != q]))
            selected_per[q] = results[chosen]["per_subject"][q]
            selected_rows.append({"subject": q, "candidate": chosen})
        selected = {"per_subject": selected_per, "macro": {p: {k: float(np.mean([v[p][k] for v in selected_per.values() if p in v])) for k in KEYS} for p in PARTS}}
        fusion["selected_other_subject_full_bacc"] = {"results": selected, "choices": selected_rows, "delta_vs_baseline": compare(results["dt1200_hmm"], selected)}
        print("FUSION", json.dumps({n: results[n]["macro"] for n in names}), flush=True)
    rows = []
    for name, result in results.items():
        for q, per in result["per_subject"].items():
            for p, m in per.items():
                rows.append({"candidate": name, "subject": q, "part": p, "threshold": 0., **m})
    for name, item in calibration.items():
        thresholds = {r["subject"]: r["threshold"] for r in threshold_rows if r["candidate"] == name}
        for q, per in item["evaluated"]["per_subject"].items():
            for p, m in per.items():
                rows.append({"candidate": name + "_calibrated", "subject": q, "part": p, "threshold": thresholds[q], **m})
    csvsave("metrics_subjects.csv", rows)
    save("results.json", {"limitations": LIMITATION, "subject_count": len(subs), "device": "CPU: small cached 1-D scores and metrics; no training",
                          "score_sha256": hashlib.sha256(e59.SCORES.read_bytes()).hexdigest(),
                          "versions": {"python": sys.version, "numpy": np.__version__},
                          "label_order": "Same SLID and np.argsort(t) as loso.load; unused feature tensors not loaded",
                          "threshold_grid": {"min": -12, "max": 12, "step": .01, "objective": "other-subject full macro bACC", "tie": "closest zero, then lower"},
                          "initialization_formula": "n starts at 1; a=1-0.5**(1/1200); u=clip((n-S)/S,0,1); alpha_n=(1-u)/n+u*a; m_n=m_(n-1)+alpha_n*(x_n-m_(n-1)); m_1=x_1; output x_n-m_n from first frame. S=60 or 120; no block reset.",
                          "fusion_scale": "sqrt(mean_other_subject(mean_full_score_squared)); no centering; ratios 0.05/0.1; selected using other-subject full macro bACC",
                          "results": results, "calibration": calibration, "initialization_vs_first_frame": initialization, "fusion": fusion,
                          "deferred": ["True training-fold fixed-center initialization", "Time-balanced model retraining", "Fully nested validation", "BDF stream validation"]})
    print("DONE", str(HERE), flush=True)


if __name__ == "__main__":
    main()
