"""Shared utilities for test/literature/classic/* (replications of classic / transfer methods on our single-channel data).

Protocols
  STRICT: subject-wise outer folds = StratifiedGroupKFold(5, shuffle, rs=0) (== depositioned_aperiodic_emg_oof.npz
          'fold'); any test-subject adaptation uses only PAST unlabelled windows in stream order (stream seeds 0,1,2);
          no test labels.
  LIT:    window-level KFold(5, shuffle, rs=0) (subjects mixed train/test) and/or whole-session per-subject
          normalisation / alignment (uses future unlabelled windows of the same subject).
Baseline for all paired bootstraps: plain LR(C=.01) on 889 features (858 + 31), STRICT folds, threshold .5.
"""
import json, sys, time, warnings
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch, hilbert
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedGroupKFold
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
TEST = HERE.parents[1]
sys.path.insert(0, str(TEST))
from depositioned_history_delta import arrange, load_db, boot  # noqa: E402
from depositioned_resnet_history_delta import load_raw, stream_order, logit, sig  # noqa: E402
from depositioned_symmetric_contrast import impute_scale  # noqa: E402
from causal_kernel_state_mapping import metrics  # noqa: E402
from switch_dual_memory import Tee  # noqa: E402

FS = 250.
STREAM_SEEDS = (0, 1, 2)
C_BASE = .01
BANDS = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}
FBANK = [(1, 4), (4, 6), (6, 8), (8, 10), (10, 13), (13, 20), (20, 30), (30, 45)]
T0 = time.perf_counter()
REF_PIPELINE = {'AUC': .7152, 'subject_macro_bACC': .6684, 'source': 'depositioned_within_subject_adapt_v2.json '
                'fuse_pseudo_lr_full_thr'}


def el():
    return f'elapsed={time.perf_counter() - T0:.1f}s'


def start_log(method, smoke):
    d = HERE / method; d.mkdir(exist_ok=True)
    stem = method + ('_smoke' if smoke else '')
    tee = Tee(sys.stdout, d / (stem + '.log')); sys.stdout = tee
    warnings.filterwarnings('ignore')
    return d, stem, tee


def finish(d, stem, tee, result):
    result['seconds'] = time.perf_counter() - T0
    (d / (stem + '.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False, default=float), encoding='utf-8')
    print(f'[done] {el()} -> {d / (stem + ".json")}', flush=True)
    sys.stdout = tee.stream; tee.f.close()


def setup():
    """Load 858 + 31 features, labels, groups, strict folds (asserted == npz) and the LIT window-level folds."""
    man, subs, data = load_db()
    Fc = np.vstack([data[s][0] for s in subs]); y = np.concatenate([data[s][1] for s in subs])
    g = np.concatenate([np.full(len(data[s][1]), s) for s in subs])
    A_ = np.load(TEST / 'depositioned_aperiodic_emg_oof.npz')
    assert np.array_equal(A_['y'], y) and np.array_equal(A_['groups'], g)
    Fn = A_['new_features']; fold = A_['fold']
    Xa, ya, ga, _ = arrange(subs, data, 0, 'independent'); fold_of = {}
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Xa, ya, ga)):
        for s in np.unique(ga[te]): fold_of[s] = k
    assert np.array_equal(np.array([fold_of[s] for s in g]), fold), 'strict fold derivation != npz'
    lit_fold = np.full(len(y), -1)
    for k, (_, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(y)): lit_fold[te] = k
    print(f'[setup] rows={len(y)} subjects={len(subs)} strict folds == npz OK {el()}', flush=True)
    return dict(subs=list(subs), Fc=Fc, Fn=Fn, y=y, g=g, fold=fold, lit_fold=lit_fold)


def Z889(D, tr):
    tfo, tfn = impute_scale(D['Fc'], tr), impute_scale(D['Fn'], tr)
    return np.hstack([tfo(D['Fc']), tfn(D['Fn'])])


# ---------------------------------------------------------------- raw-derived features (cached)
def _bp(lo, hi):
    return butter(4, [lo, min(hi, 124.)], btype='band', fs=FS, output='sos')


def raw_features(D):
    path = HERE / '_cache_rawfeat.npz'
    if path.exists():
        c = np.load(path)
        assert np.array_equal(c['y'], D['y']) and np.array_equal(c['g'], D['g'])
        print(f'[raw] loaded cache {path.name} {el()}', flush=True)
        return {k: c[k] for k in c.files}
    X, F, y, g = load_raw(D['subs'])
    assert np.array_equal(F, D['Fc']) and np.array_equal(y, D['y']) and np.array_equal(g, D['g'])
    n = len(y); out = {'y': y, 'g': g}
    nb = len(FBANK)
    out['cov'] = np.zeros((n, nb, nb)); out['envcorr'] = np.zeros((n, nb * (nb - 1) // 2))
    out['band'] = np.zeros((n, 2 * len(BANDS) + 1)); out['psd'] = np.zeros((n, 44)); out['theta'] = np.zeros((n, 3))
    sos_fb = [_bp(*b) for b in FBANK]; sos_th = _bp(4, 8); sos_bb = _bp(1, 45)
    iu = np.triu_indices(nb, 1)
    for a in range(0, n, 2000):
        xb = X[a:a + 2000].astype(np.float64) * 1e6  # V -> uV
        xb = xb - xb.mean(1, keepdims=True)
        f, P = welch(xb, fs=FS, nperseg=500, noverlap=250, axis=1)
        bp = np.stack([P[:, (f >= lo) & (f < hi)].sum(1) for lo, hi in BANDS.values()], 1) + 1e-12
        tot = P[:, (f >= 1) & (f < 45)].sum(1) + 1e-12
        out['band'][a:a + 2000] = np.c_[np.log(bp), np.log(bp / tot[:, None]), np.log(tot)]
        sel = (f >= 1) & (f < 45); fs_ = f[sel]; Ps = P[:, sel]
        out['psd'][a:a + 2000] = np.log(np.stack([Ps[:, (fs_ >= j) & (fs_ < j + 1)].sum(1) for j in range(1, 45)], 1) + 1e-12)
        fb = np.stack([sosfiltfilt(s, xb, axis=1) for s in sos_fb], 1)  # (b, nb, T)
        C = np.einsum('bit,bjt->bij', fb, fb) / fb.shape[2]
        tr_ = np.trace(C, axis1=1, axis2=2)
        out['cov'][a:a + 2000] = C + 1e-6 * tr_[:, None, None] * np.eye(nb)[None]
        env = np.abs(hilbert(fb, axis=2)); env = env - env.mean(2, keepdims=True)
        env = env / (np.linalg.norm(env, axis=2, keepdims=True) + 1e-12)
        R = np.einsum('bit,bjt->bij', env, env)
        out['envcorr'][a:a + 2000] = np.arctanh(np.clip(R[:, iu[0], iu[1]], -.999, .999))
        th = sosfiltfilt(sos_th, xb, axis=1); bb = sosfiltfilt(sos_bb, xb, axis=1)
        for i in range(len(xb)):
            v = th[i]; mn = np.flatnonzero((v[1:-1] < v[:-2]) & (v[1:-1] <= v[2:]) & (v[1:-1] < 0)) + 1
            amp = -v[mn] if mn.size else np.array([1e-6])
            out['theta'][a + i] = [np.log(amp.mean() + 1e-9), np.log(np.median(amp) + 1e-9),
                                   np.log(amp.mean() / (bb[i].std() + 1e-9) + 1e-9)]
        print(f'[raw] {min(a + 2000, n)}/{n} windows {el()}', flush=True)
    np.savez_compressed(path, **out)
    return out


def covs(R):
    """Filter-bank covariances + absolute ridge (flat std=0 windows otherwise give singular matrices)."""
    return R['cov'] + 1e-2 * np.eye(R['cov'].shape[1])[None]


# ---------------------------------------------------------------- Riemannian helpers (no pyriemann dependency)
def _fn(C, f):
    w, V = np.linalg.eigh(C)
    return (V * f(np.maximum(w, 1e-12))[..., None, :]) @ np.swapaxes(V, -1, -2)


def logm(C): return _fn(C, np.log)
def expm_s(C):
    w, V = np.linalg.eigh(C); return (V * np.exp(w)[..., None, :]) @ np.swapaxes(V, -1, -2)
def sqrtm(C): return _fn(C, np.sqrt)
def invsqrtm(C): return _fn(C, lambda w: 1 / np.sqrt(w))


def riem_mean(Cs, it=50, tol=1e-8):
    M = Cs.mean(0)
    for _ in range(it):
        s, si = sqrtm(M), invsqrtm(M)
        T = logm(si @ Cs @ si).mean(0)
        M = s @ expm_s(T) @ s
        if np.linalg.norm(T) < tol: break
    return M


def geodesic(A, B, a):
    s, si = sqrtm(A), invsqrtm(A)
    return s @ _fn(si @ B @ si, lambda w: w ** a) @ s


def tangent(Cs, M):
    """Upper-triangular vectorisation of logm(M^-1/2 C M^-1/2) with sqrt(2) off-diagonal weights."""
    si = invsqrtm(M); L = logm(si @ Cs @ si); n = L.shape[-1]
    iu = np.triu_indices(n); w = np.where(iu[0] == iu[1], 1., np.sqrt(2))
    return L[..., iu[0], iu[1]] * w


# ---------------------------------------------------------------- baselines, reporting, checks
def strict_baseline(D, folds):
    """Plain LR(C=.01) on 889 features, strict subject folds -> probability."""
    p = np.full(len(D['y']), np.nan)
    for k in folds:
        tr, te = np.flatnonzero(D['fold'] != k), np.flatnonzero(D['fold'] == k)
        Z = Z889(D, tr)
        with threadpool_limits(limits=8):
            p[te] = LogisticRegression(C=C_BASE, max_iter=2000).fit(Z[tr], D['y'][tr]).predict_proba(Z[te])[:, 1]
    return p


def lit_baseline(D, folds):
    p = np.full(len(D['y']), np.nan)
    for k in folds:
        tr, te = np.flatnonzero(D['lit_fold'] != k), np.flatnonzero(D['lit_fold'] == k)
        Z = Z889(D, tr)
        with threadpool_limits(limits=8):
            p[te] = LogisticRegression(C=C_BASE, max_iter=2000).fit(Z[tr], D['y'][tr]).predict_proba(Z[te])[:, 1]
    return p


def report(name, p, D, mask, base, nb, tag=''):
    y, g = D['y'], D['g']
    m = metrics(y, g, p, mask)
    ba = [v for s in np.unique(g[mask]) if (v := metrics(y, g, p, mask & (g == s))['subject_macro_bACC']) is not None]
    m['n_subj_bacc_ge_0.9'] = int(np.sum(np.array(ba) >= .9))
    m['boot_vs_strict_LR889'] = boot(p, base, y, g, mask, nb)
    b = m['boot_vs_strict_LR889']
    print(f'[result{tag}] {name:34s} AUC={m["AUC"]:.4f} mac={m["subject_macro_bACC"]:.4f} >=.9:{m["n_subj_bacc_ge_0.9"]:2d} '
          f'dAUC={b["dAUC"]["mean"]:+.4f}[{b["dAUC"]["p2.5"]:+.4f},{b["dAUC"]["p97.5"]:+.4f}] '
          f'dmac={b["dmacro_bACC"]["mean"]:+.4f}[{b["dmacro_bACC"]["p2.5"]:+.4f},{b["dmacro_bACC"]["p97.5"]:+.4f}]',
          flush=True)
    return m


def check_label_shuffle(fn, D, k):
    """fn(y_used) -> test scores of fold k; permuting the test-fold labels must leave them bit-identical."""
    te = D['fold'] == k
    y2 = D['y'].copy(); y2[te] = np.random.default_rng(k + 11).permutation(y2[te])
    a, b = fn(D['y']), fn(y2)
    assert np.array_equal(a, b), 'test-label shuffle changed strict scores'
    return 'ok'


def check_prefix(fn_subject, Xs, cut=None):
    """fn_subject(rows in stream order) -> scores; perturbing rows[cut:] must leave scores[:cut] identical."""
    cut = len(Xs) // 2 if cut is None else cut
    a = fn_subject(Xs)
    Xp = Xs.copy(); Xp[cut:] = Xp[cut:][::-1] * 1.7 + .3 if Xp.ndim == 2 else Xp[cut:][::-1] * 1.7
    b = fn_subject(Xp)
    d_pre, d_fut = float(np.abs(a[:cut] - b[:cut]).max()), float(np.abs(a[cut:] - b[cut:]).max())
    assert d_pre == 0. and d_fut > 0., ('prefix invariance', d_pre, d_fut)
    return {'max_abs_prefix_change': d_pre, 'max_abs_future_change': d_fut}
