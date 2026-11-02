"""Shared helpers for test/memory/* (real recording order, causal subject memory).

Input: test/memory/realorder_oof.npz written by `python test/realorder/realorder_v2.py --orders real --save-oof`:
  oof__<name>   raw-tier outer-OOF logits (real order) of every realorder_v2 output (ana_bc, a, bc_sel, c, ...)
  inner<k>__<n> inner SGKF(4) OOF logits on the TRAINING subjects of outer fold k (a, bc_sel, c, ana_bc), nan elsewhere
  pos           0-based position of each row in its subject's real order;  t  recording time (s, window end)
Window step = 6 s (non-overlapping 6 s windows; median diff of t_end), so half-life h windows = 6 h s.
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

import json, sys, time, warnings
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

HERE = Path(__file__).resolve().parent
TEST = HERE.parent
sys.path.insert(0, str(TEST / 'audit')); sys.path.insert(0, str(TEST))
from common import GROUPS, Tee, boot, load_all, metrics  # noqa: E402,F401

STEP_S = 6.0
BLOCK_S = 300.0  # R/T alternate in 5-min blocks
T0 = time.perf_counter()


def el():
    return f'elapsed={time.perf_counter() - T0:.1f}s'


def open_log(stem):
    warnings.filterwarnings('ignore')
    tee = Tee(sys.stdout, HERE / (stem + '.log')); sys.stdout = tee
    return tee


def save(stem, obj, tee):
    obj['seconds'] = time.perf_counter() - T0
    (HERE / (stem + '.json')).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float), encoding='utf-8')
    print(f'[done] {el()} -> {stem}.json', flush=True)
    sys.stdout = tee.stream; tee.f.close()


def hl_time(h):
    s = h * STEP_S
    return f'{h}win={s:.0f}s={s / 60:.1f}min'


def load_oof(smoke=False):
    O = np.load(HERE / ('realorder_oof' + ('_smoke' if smoke else '') + '.npz'))
    O = {k: O[k] for k in O.files}
    D = load_all()
    assert np.array_equal(O['y'], D['y']) and np.array_equal(O['g'], D['g']) and np.array_equal(O['fold'], D['fold'])
    O['subs'], O['grp_of'], O['D'] = D['subs'], D['grp_of'], D
    O['order'] = {s: np.flatnonzero(O['g'] == s)[np.argsort(O['pos'][O['g'] == s], kind='stable')] for s in D['subs']}
    O['ev'] = np.isin(O['fold'], O['folds'])
    st = np.concatenate([np.diff(np.sort(O['t'][O['g'] == s])) for s in D['subs']])
    print(f'[load] rows={len(O["y"])} folds={list(O["folds"])} median step={np.median(st):.2f}s '
          f'(assumed {STEP_S}s) {el()}', flush=True)
    assert abs(np.median(st) - STEP_S) < 1e-6
    return O


# ---------------------------------------------------------------- causal memory primitives (one subject, real order)
def ema(l, h):
    """causal EMA, half-life h windows, initialised at l[0] (no future); h<=0 -> identity."""
    if h <= 0: return np.asarray(l, float).copy()
    a = 1. - 2. ** (-1. / h)
    return lfilter([a], [1., -(1. - a)], l, zi=[(1. - a) * l[0]])[0]


def per_subject(fn, l, order):
    """apply fn (1-D causal) per subject along its real order (subjects with any nan are skipped -> nan)."""
    out = np.full(len(l), np.nan)
    for s, loc in order.items():
        if not np.isfinite(l[loc]).all(): continue
        out[loc] = fn(l[loc])
    return out


HS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256)  # 0 = raw (no smoothing)


def ema_stack(l, hs=HS):
    return np.column_stack([ema(l, h) for h in hs])


def _bacc(yy, yh):
    return .5 * (yh[yy == 1].mean() + (~yh[yy == 0]).mean())


def stack_lr():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=1., max_iter=2000)


def select_smooth(L, locs, y, hs=HS):
    """Inner selection on TRAINING subjects only. L: inner-OOF logits (finite on locs), locs: list of row-index arrays
    (one per training subject, real order). -> dict: h (single half-life, ties -> smaller), inner macro per h,
    multi-scale LR stack (fitted on all locs) and its nested GroupKFold(4)-over-subjects inner macro, mode."""
    from sklearn.model_selection import GroupKFold
    sc = {}
    for h in hs:
        sc[h] = float(np.mean([_bacc(y[loc], ema(L[loc], h) >= 0) for loc in locs]))
    hbest = max(hs, key=lambda h: (sc[h] + 1e-12 * 0, -h)); hbest = min([h for h in hs if sc[h] >= sc[hbest] - 1e-12])
    X = np.vstack([ema_stack(L[loc], hs) for loc in locs]); yy = np.concatenate([y[loc] for loc in locs])
    gg = np.concatenate([np.full(len(loc), j) for j, loc in enumerate(locs)])
    nested = []
    for a_, b_ in GroupKFold(4).split(X, yy, gg):
        m = stack_lr().fit(X[a_], yy[a_]); d = m.decision_function(X[b_])
        for j in np.unique(gg[b_]):
            q = gg[b_] == j; nested.append(_bacc(yy[b_][q], d[q] >= 0))
    st = stack_lr().fit(X, yy)
    sm = float(np.mean(nested))
    return {'h': int(hbest), 'inner_macro': {int(h): v for h, v in sc.items()}, 'stack': st, 'stack_inner_macro': sm,
            'stack_coef': st.coef_.ravel().tolist(), 'stack_intercept': float(st.intercept_[0]),
            'mode': 'stack' if sm > sc[hbest] else 'single'}


def apply_smooth(sel, l):
    """one subject, real order -> dict(single, stack, auto) logits."""
    s1 = ema(l, sel['h']); s2 = sel['stack'].decision_function(ema_stack(l))
    return {'single': s1, 'stack': s2, 'auto': s2 if sel['mode'] == 'stack' else s1}


def sig(z):
    return 1. / (1. + np.exp(-z))


# ---------------------------------------------------------------- evaluation
def per_subject_bacc(y, g, l, mask, thr=0.):
    per = {}
    for s in np.unique(g[mask & np.isfinite(l)]):
        q = mask & (g == s); yh = l[q] >= thr; yy = y[q]
        if np.unique(yy).size == 2: per[s] = .5 * (yh[yy == 1].mean() + (~yh[yy == 0]).mean())
    return per


def macro(y, g, l, mask):
    p = per_subject_bacc(y, g, l, mask); return float(np.mean(list(p.values())))


def evaluate(y, g, l, ev, grp_of):
    m = metrics(y, g, sig(l), ev); per = per_subject_bacc(y, g, l, ev)
    m['groups'] = {G: float(np.mean([v for s, v in per.items() if grp_of[s] == G])) for G in GROUPS
                   if any(grp_of[s] == G for s in per)}
    m['n_ge_0.9'] = int(sum(v >= .9 for v in per.values())); m['n_subj'] = len(per)
    return m


def fmt(m):
    return (f'AUC={m["AUC"]:.4f} mac={m["subject_macro_bACC"]:.4f} ' +
            ' '.join(f'{G}={m["groups"].get(G, float("nan")):.4f}' for G in GROUPS) + f' n9={m["n_ge_0.9"]}')


def boot_vs(a, b, y, g, ev, nb):
    bb = boot(sig(a), sig(b), y, g, ev, nb)
    return bb


def fmt_boot(bb):
    return (f'dmac={bb["dmacro_bACC"]["mean"]:+.4f}[{bb["dmacro_bACC"]["p2.5"]:+.4f},{bb["dmacro_bACC"]["p97.5"]:+.4f}] '
            f'dAUC={bb["dAUC"]["mean"]:+.4f}[{bb["dAUC"]["p2.5"]:+.4f},{bb["dAUC"]["p97.5"]:+.4f}]')


def label_shuffle_metric(y, g, l, ev, fold, seed=0):
    """test labels permuted within each test subject's rows (fold-wise pooled permutation) -> metrics ~ .5."""
    rng = np.random.default_rng(seed); y2 = y.copy()
    for s in np.unique(g[ev]):
        q = np.flatnonzero(g == s); y2[q] = rng.permutation(y2[q])
    return y2
