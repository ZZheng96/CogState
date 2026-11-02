"""Legal P2 direction calibration stress test on the strict current-window OOF scores.

For each held-out subject, only the first m labeled windows of each class are
reserved as an explicit calibration set. The remaining windows are evaluated.
Calibration estimates only a sign and intercept for the frozen score; it never
uses later labels. This is a protocol experiment, not a claim that the
competition grants these labels.
"""
# --- test layout bootstrap (generated 2026-11-02): locate test/ root, shared modules and cache ---
import sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_TEST = _HERE
while _TEST.name != "test" and _TEST.parent != _TEST:
    _TEST = _TEST.parent
CACHE_DIR = _TEST / "cache"
if str(_TEST) not in _sys.path:
    _sys.path.append(str(_TEST))
for _d in sorted(_TEST.iterdir()):
    if _d.is_dir() and _d.name not in {"cache", "results", "pretrained_cache", "reports", "experiments", "_orphans", "__pycache__"}:
        if str(_d) not in _sys.path:
            _sys.path.append(str(_d))
# --- end bootstrap ---

import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

OUT = Path(__file__).resolve().parent
CACHE = CACHE_DIR / 'broadband_shape_window6_oof.npz'
RES = OUT / "theory_p2_direction_calibration_results.json"
MS = (1, 3, 5, 10, 20)
START = time.perf_counter()


def log(msg):
    print(f"[{time.perf_counter() - START:7.1f}s] {msg}", flush=True)


def metrics(y, g, p, mask):
    vals = []
    for s in np.unique(g[mask]):
        q = mask & (g == s)
        if np.unique(y[q]).size == 2:
            vals.append(balanced_accuracy_score(y[q], p[q] >= .5))
    classes = np.unique(y[mask])
    auc = float(roc_auc_score(y[mask], p[mask])) if len(classes) == 2 else None
    return {"auc": auc, "bacc": float(balanced_accuracy_score(y[mask], p[mask] >= .5)) if mask.any() else None, "subject_macro_bacc": float(np.mean(vals)) if vals else None, "n_eval": int(mask.sum())}


def main():
    d = np.load(CACHE, allow_pickle=False)
    y, g = d["y"].astype(int), d["groups"]
    fold = d["fold_id"].astype(int)
    rt_path = OUT / "audit" / "literature_protocol" / "real_time.npz"
    if not rt_path.exists():
        raise FileNotFoundError(f"required chronology cache missing: {rt_path}")
    rt = np.load(rt_path, allow_pickle=False)
    rt_groups = np.array([str(s).replace("sub-", "") for s in rt["groups"]])
    assert np.array_equal(np.sort(np.unique(g)), np.sort(np.unique(rt_groups)))
    t_end = np.full(len(y), np.nan)
    for s in np.unique(g):
        for c in (0, 1):
            hi = np.flatnonzero((g == s) & (y == c))
            ri = np.flatnonzero((rt_groups == s) & (rt["y"] == c))
            assert len(hi) == len(ri)
            perm = np.random.default_rng(0).permutation(len(ri))
            t_end[hi] = rt["t_end"][ri[perm]]
    assert np.isfinite(t_end).all() and len(np.unique(t_end)) == len(t_end)
    names = [k for k in d.files if k.startswith("logreg__") or k.startswith("hgb__")]
    names = [k for k in names if k not in ("logreg__orig856", "hgb__orig856")]
    result = {"protocol": {"calibration": "per subject, time-sorted first m class-0 and first m class-1 windows", "evaluation": "t_end strictly greater than calibration maximum; same-mask frozen baseline", "score": "frozen strict current-window OOF score", "labels_after_calibration": False, "m_values": list(MS), "real_time": True, "alignment": "real_time.npz y/groups exact equality asserted"}, "variants": {}}
    for name in names:
        base = d[name].astype(float)
        result["variants"][name] = {}
        for m in MS:
            p = base.copy(); eval_mask = np.zeros(len(y), bool); same_mask = np.zeros(len(y), bool); ncal = []
            for s in np.unique(g):
                ix = np.flatnonzero(g == s)
                order = ix[np.argsort(t_end[ix])] if t_end is not None else ix
                cal = []
                for c in (0, 1):
                    hit = order[y[order] == c]
                    cal.extend(hit[:m].tolist())
                cal = np.asarray(sorted(cal), dtype=int)
                assert len(cal) > 0
                eval_ix = ix[t_end[ix] > np.max(t_end[cal])]
                eval_mask[eval_ix] = True
                same_mask[eval_ix] = True
                ncal.append(len(cal))
                if len(cal) >= 2 and np.unique(y[cal]).size == 2:
                    z = np.log(np.clip(base[cal], 1e-5, 1 - 1e-5) / np.clip(1 - base[cal], 1e-5, 1 - 1e-5))
                    # Compare the two possible orientations; intercept is the midpoint
                    # of the two calibration class medians after applying that orientation.
                    a = np.median(z[y[cal] == 1]) - np.median(z[y[cal] == 0])
                    sign = 1.0 if a >= 0 else -1.0
                    mu0, mu1 = np.median(z[y[cal] == 0]), np.median(z[y[cal] == 1])
                    threshold = sign * (mu0 + mu1) / 2.0
                    zi = np.log(np.clip(base[ix], 1e-5, 1 - 1e-5) / np.clip(1 - base[ix], 1e-5, 1 - 1e-5))
                    p[ix] = 1.0 / (1.0 + np.exp(-(sign * zi - threshold)))
            result["variants"][name][f"m{m}"] = metrics(y, g, p, eval_mask)
            result["variants"][name][f"m{m}"]["same_mask_frozen_baseline"] = metrics(y, g, base, same_mask)
            result["variants"][name][f"m{m}"]["mean_calibration_windows"] = float(np.mean(ncal))
            log(f"{name} m={m} {result['variants'][name][f'm{m}']}")
    result["elapsed_seconds"] = time.perf_counter() - START
    RES.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"COMPLETE {RES.name}")


if __name__ == "__main__":
    main()
