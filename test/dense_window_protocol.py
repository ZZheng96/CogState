"""Dense-window training protocol (user-approved):
training may use any-length data (dense overlapping windows, long-context teacher targets);
inference uses only the newest 6 s, optionally with 1 s delay (7 s input [t-6, t+1]).

Test set per outer fold = canonical non-overlapping 6 s windows of test subjects (identical to
build_database / highdim_deep_features order). Outer CV: StratifiedGroupKFold(5, shuffle, rs=0).
Features: 86-dim broadband_shape compute_features + 7 log band powers, per window only.

Usage:
  python test/dense_window_protocol.py            # feature conditions (a)-(d), Logistic + HGB
  python test/dense_window_protocol.py --resnet 0 1   # ResNet1d (a) vs (b), seeds 0 1 (appends to results)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import welch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'test'
sys.path[:0] = [str(ROOT / 'src'), str(OUT)]
from cogstate import load_trials, bad_intervals, valid_window  # noqa: E402
from broadband_shape_window6 import compute_features  # noqa: E402

FS = 250.
RES = OUT / 'dense_window_protocol_results.json'
OOF = OUT / 'dense_window_protocol_oof.npz'
CACHE = OUT / 'dense_window_protocol_cache.npz'
RAW_CACHE = OUT / 'dense_window_protocol_raw6.npy'
STRIDE = 1.0
CTX = 30.0
C_GRID = (0.001, 0.01, 0.1, 1.0)
HGB_SEEDS = (0, 1, 2)
BOOT_SEED, N_BOOT = 20261029, 1000
T0 = time.time()


def log(msg):
    print(f'[{time.time() - T0:7.1f}s] {msg}', flush=True)


# ---------------- features ----------------
BANDS7 = (('delta', 1, 4), ('theta', 4, 8), ('alpha', 8, 13), ('beta1', 13, 20),
          ('beta2', 20, 30), ('gamma', 30, 45), ('hf', 52, 100))


def window_features(raw):
    """raw: (n, L) float. Returns (n, 93) and names. Per-window only."""
    raw = raw.astype('float64')
    f, P = welch(raw, fs=FS, nperseg=500, noverlap=250, axis=1, detrend='constant')
    fam = compute_features(raw, f, P)
    X = np.hstack([fam[k][0] for k in fam])
    names = sum([fam[k][1] for k in fam], [])
    df = f[1] - f[0]
    lbp = np.stack([np.log(P[:, (f >= lo) & (f < hi)].sum(1) * df + 1e-30) for _, lo, hi in BANDS7], 1)
    X = np.hstack([X, lbp])
    names += [f'logbp:{n}' for n, _, _ in BANDS7]
    X[~np.isfinite(X)] = 0.
    return X.astype(np.float32), names


def extract():
    paths = sorted((ROOT / 'EEG_dataset').glob('*/*_events.csv'))
    F6, F7, meta, raws, names = [], [], [], [], None
    for i, csv in enumerate(paths, 1):
        bdf = next(csv.parent.glob('*.bdf'))
        import mne
        raw = mne.io.read_raw_bdf(str(bdf), preload=True, verbose='ERROR')
        fs = float(raw.info['sfreq']); assert np.isclose(fs, FS)
        data = raw.get_data()[0]; bad = bad_intervals(raw); rec_end = raw.n_times / fs
        sid = csv.parent.name
        x6, x7, m = [], [], []
        for tr in load_trials(str(csv), str(bdf)):
            k = 0
            while True:
                t = tr.start + 6 + k * STRIDE
                if t > tr.end + 1e-6:
                    break
                s = round((t - 6) * fs)
                if valid_window(t - 6, t, bad, rec_end):
                    x = data[s:s + 1500]
                    if len(x) == 1500 and np.isfinite(x).all():
                        has7 = False
                        if t + 1 <= tr.end + 1e-6 and valid_window(t - 6, t + 1, bad, rec_end):
                            z = data[s:s + 1750]
                            has7 = len(z) == 1750 and np.isfinite(z).all()
                        x6.append(x)
                        x7.append(data[s:s + 1750] if has7 else None)
                        m.append((sid, tr.trial_id, tr.label, t - tr.start, k % 6 == 0, has7))
                k += 1
        raw.close()
        A6 = np.array(x6)
        f6, names = window_features(A6)
        f7 = np.full_like(f6, np.nan)
        h7 = np.array([q[5] for q in m])
        if h7.any():
            f7[h7] = window_features(np.array([z for z in x7 if z is not None]))[0]
        F6.append(f6); F7.append(f7); meta += m; raws.append(A6.astype(np.float32))
        log(f'[extract {i}/{len(paths)}] sub-{sid} dense={len(m)} canon={sum(q[4] for q in m)} '
            f'has7={int(h7.sum())}')
    F6, F7 = np.vstack(F6), np.vstack(F7)
    meta = {'subject': np.array([q[0] for q in meta]), 'trial': np.array([q[1] for q in meta]),
            'label': np.array([q[2] for q in meta], np.int8), 'trel': np.array([q[3] for q in meta]),
            'canon': np.array([q[4] for q in meta]), 'has7': np.array([q[5] for q in meta])}
    np.save(RAW_CACHE, np.vstack(raws))
    np.savez_compressed(CACHE, F6=F6, F7=F7, names=np.array(names), **meta)
    return F6, F7, meta, names


def teacher_targets(F6, meta):
    """Label-free long-context targets (computed per subject; only training rows are ever used for fitting)."""
    ctx = np.zeros_like(F6, dtype=np.float64)
    snorm = np.zeros_like(F6, dtype=np.float64)
    keys = np.char.add(np.char.add(meta['subject'], '_'), meta['trial'].astype(str))
    for key in np.unique(keys):
        idx = np.where(keys == key)[0]
        t = meta['trel'][idx]; o = np.argsort(t); idx, t = idx[o], t[o]
        cs = np.vstack([np.zeros(F6.shape[1]), np.cumsum(F6[idx].astype(np.float64), 0)])
        lo = np.searchsorted(t, t - CTX - 1e-6, 'left'); hi = np.searchsorted(t, t + CTX + 1e-6, 'right')
        ctx[idx] = (cs[hi] - cs[lo]) / (hi - lo)[:, None]
    for s in np.unique(meta['subject']):
        idx = meta['subject'] == s
        mu, sd = F6[idx].mean(0), F6[idx].std(0)
        snorm[idx] = (F6[idx] - mu) / np.where(sd > 1e-12, sd, 1.)
    return ctx.astype(np.float32), snorm.astype(np.float32)


# ---------------- classifiers ----------------
def youden(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


class Logit:
    def __init__(self, C=None):
        self.C = C

    def fit(self, X, y, g):
        self.keep = X.std(0) > 1e-12
        X = X[:, self.keep]
        inner_oof = None
        if self.C is None:
            best = -1
            for C in C_GRID:
                oof = np.zeros(len(y))
                for itr, iva in GroupKFold(3).split(X, y, g):
                    sc = StandardScaler().fit(X[itr])
                    m = LogisticRegression(C=C, max_iter=2000, class_weight='balanced').fit(sc.transform(X[itr]), y[itr])
                    oof[iva] = m.decision_function(sc.transform(X[iva]))
                a = np.mean([roc_auc_score(y[iva], oof[iva]) for _, iva in GroupKFold(3).split(X, y, g)])
                if a > best:
                    best, self.C, inner_oof = a, C, oof
        self.sc = StandardScaler().fit(X)
        self.m = LogisticRegression(C=self.C, max_iter=2000, class_weight='balanced').fit(self.sc.transform(X), y)
        return inner_oof

    def score(self, X):
        return self.m.decision_function(self.sc.transform(X[:, self.keep]))


class HGB:
    C = None

    def __init__(self, seeds=HGB_SEEDS):
        self.seeds = seeds

    @staticmethod
    def _m(s):
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                                              l2_regularization=1.0, max_features=0.5, early_stopping=False, random_state=s)

    def fit(self, X, y, g, inner=True):
        inner_oof = None
        if inner:
            inner_oof = np.zeros(len(y))
            for itr, iva in GroupKFold(3).split(X, y, g):
                inner_oof[iva] = self._m(0).fit(X[itr], y[itr]).predict_proba(X[iva])[:, 1]
        self.ms = [self._m(s).fit(X, y) for s in self.seeds]
        return inner_oof

    def score(self, X):
        return np.mean([m.predict_proba(X)[:, 1] for m in self.ms], 0)


def make_clf(kind, C=None):
    return Logit(C) if kind == 'logreg' else HGB()


class Teacher:
    """Ridge student: short-window features -> long-context targets; classifier trained on targets."""
    def __init__(self, kind, alpha=10.):
        self.kind, self.alpha = kind, alpha

    def _fit_ridge(self, X, T):
        sx = StandardScaler().fit(X); st = StandardScaler().fit(T)
        r = Ridge(alpha=self.alpha).fit(sx.transform(X), st.transform(T))
        return lambda Z: st.inverse_transform(r.predict(sx.transform(Z)))

    def fit(self, X, T, y, g):
        self.ridge = self._fit_ridge(X, T)
        self.clf = make_clf(self.kind)
        self.clf.fit(T, y, g, **({} if self.kind == 'logreg' else {'inner': False}))
        inner_oof = np.zeros(len(y))
        for itr, iva in GroupKFold(3).split(X, y, g):
            rd = self._fit_ridge(X[itr], T[itr])
            c = Logit(self.clf.C) if self.kind == 'logreg' else HGB(seeds=(0,))
            c.fit(T[itr], y[itr], g[itr], **({} if self.kind == 'logreg' else {'inner': False}))
            inner_oof[iva] = c.score(rd(X[iva]))
        return inner_oof

    def score(self, X):
        return self.clf.score(self.ridge(X))


# ---------------- metrics ----------------
def subject_macro_auc(y, p, g):
    a = [roc_auc_score(y[g == s], p[g == s]) for s in np.unique(g) if len(np.unique(y[g == s])) == 2]
    return float(np.mean(a)), len(a)


def boot_delta(y, g, a, b):
    """ΔAUC = b − a, subject-level bootstrap."""
    rng = np.random.default_rng(BOOT_SEED)
    subs = np.unique(g); idx = {s: np.where(g == s)[0] for s in subs}; d = []
    for _ in range(N_BOOT):
        ii = np.concatenate([idx[s] for s in rng.choice(subs, len(subs), replace=True)])
        if len(np.unique(y[ii])) == 2:
            d.append(roc_auc_score(y[ii], b[ii]) - roc_auc_score(y[ii], a[ii]))
    d = np.array(d)
    return {'delta_auc': float(roc_auc_score(y, b) - roc_auc_score(y, a)),
            'ci95': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            'p_delta_gt0': float((d > 0).mean()), 'n_boot': int(len(d))}


def summarize(y, g, score, hard, mask):
    y, g, score, hard = y[mask], g[mask], score[mask], hard[mask]
    macro, nsub = subject_macro_auc(y, score, g)
    return {'n_test': int(mask.sum()), 'auc': float(roc_auc_score(y, score)),
            'bacc': float(balanced_accuracy_score(y, hard)), 'subject_macro_auc': macro, 'n_subjects_macro': nsub}


# ---------------- main (feature conditions) ----------------
def load_or_extract():
    if CACHE.exists() and RAW_CACHE.exists():
        d = np.load(CACHE)
        meta = {k: d[k] for k in ('subject', 'trial', 'label', 'trel', 'canon', 'has7')}
        log(f'loaded cache {CACHE.name}')
        return d['F6'], d['F7'], meta, list(d['names'])
    return extract()


def check_canonical(meta):
    ref = np.load(OUT / 'highdim_deep_features.npz')
    c = np.where(meta['canon'])[0]
    ok = (len(c) == len(ref['y']) and np.array_equal(meta['label'][c], ref['y'])
          and np.array_equal(meta['subject'][c], ref['groups']))
    if ok:
        raw6 = np.load(RAW_CACHE, mmap_mode='r')
        sel = np.linspace(0, len(c) - 1, 50).astype(int)
        ok = bool(np.allclose(raw6[c[sel]], ref['raw'][sel], atol=1e-9))
    assert ok, 'canonical windows do not match highdim_deep_features.npz'
    return c


def main_features():
    F6, F7, meta, names = load_or_extract()
    log(f'dense rows={len(F6)} dim={F6.shape[1]} canon={int(meta["canon"].sum())} has7={int(meta["has7"].sum())}')
    cidx = check_canonical(meta)
    log('canonical windows identical to highdim_deep_features.npz (order, labels, subjects, raw)')
    ctx, snorm = teacher_targets(F6, meta)
    y_all, g_all = meta['label'].astype(int), meta['subject']
    yc, gc = y_all[cidx], g_all[cidx]
    has7_c = meta['has7'][cidx]
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(len(yc)), yc, gc))
    fold_id = np.zeros(len(yc), int)
    for k, (_, te) in enumerate(folds):
        fold_id[te] = k + 1
    log(f'canonical test windows: {len(yc)}; with 7 s available: {int(has7_c.sum())} (dropped {int((~has7_c).sum())})')

    # condition: (train-row selector, X_train source, X_test source, teacher target, test mask)
    dense, dense7 = np.ones(len(F6), bool), meta['has7'].copy()
    canon = meta['canon'].copy()
    conds = {
        'a_canon6': (canon, F6, F6[cidx], None, np.ones(len(yc), bool)),
        'b_dense6': (dense, F6, F6[cidx], None, np.ones(len(yc), bool)),
        'c_dense7': (dense7, F7, F7[cidx], None, has7_c),
        'c_canon7': (canon & dense7, F7, F7[cidx], None, has7_c),
        'd_teacher_ctx30_6s': (dense, F6, F6[cidx], ctx, np.ones(len(yc), bool)),
        'd_teacher_subjnorm_6s': (dense, F6, F6[cidx], snorm, np.ones(len(yc), bool)),
        'd_teacher_ctx30_7s': (dense7, F7, F7[cidx], ctx, has7_c),
    }
    results = {'protocol': {
        'train_data': 'any length allowed; dense = 6 s windows at 1 s stride ending at trial.start+6+k (valid_window, finite)',
        'inference': '6 s [t-6,t] or 7 s [t-6,t+1] (1 s delay, needs t+1<=trial.end and valid); per-window features only',
        'test': 'canonical non-overlapping 6 s windows of test subjects (== highdim_deep_features.npz order)',
        'features': '86-dim broadband_shape compute_features + 7 log band powers (Welch 2 s, hann, 50% overlap), raw data (no extra preprocessing, same as build_database)',
        'outer_cv': 'StratifiedGroupKFold(5, shuffle=True, random_state=0) on canonical windows, groups=subject; training rows = all rows of training subjects',
        'logreg': 'StandardScaler + L2 LR class_weight=balanced; C in {0.001,0.01,0.1,1} by inner GroupKFold(3) AUC',
        'hgb': 'max_iter=200 lr=0.05 leaves=15 min_leaf=40 l2=1 max_features=0.5, seeds 0/1/2 averaged',
        'threshold': 'Youden on inner GroupKFold(3) OOF scores of training subjects (never test)',
        'teacher': f'targets (training subjects only): ctx30 = mean of dense 6 s features within ±{CTX:.0f} s in same trial; '
                   'subjnorm = per-subject z-scored 6 s features. Ridge(alpha=10, standardized) student maps 6/7 s features -> target; '
                   'classifier trained on true targets, applied to predicted targets. Threshold from inner-CV full pipeline.',
        'bootstrap': f'{N_BOOT} subject resamples, seed {BOOT_SEED}; ΔAUC = condition − a_canon6 (same classifier, same test windows)',
        'n_dense_rows': int(len(F6)), 'n_canonical': int(len(yc)), 'n_canonical_with_7s': int(has7_c.sum()),
        'n_dropped_for_7s': int((~has7_c).sum()), 'feature_dim': int(F6.shape[1]),
    }, 'conditions': {}}
    scores, hards = {}, {}
    for kind in ('logreg', 'hgb'):
        for cname, (sel, Xsrc, Xte_all, T, tmask) in conds.items():
            key = f'{kind}|{cname}'
            sc, hd = np.full(len(yc), np.nan), np.zeros(len(yc), bool)
            tick, per_fold = time.time(), []
            for k, (_, te) in enumerate(folds):
                trs = set(gc[te]) ^ set(gc)  # training subjects
                tr = np.where(sel & np.isin(g_all, list(trs)))[0]
                te = te[tmask[te]]
                assert not set(g_all[tr]) & set(gc[te])
                if T is None:
                    model = make_clf(kind)
                    inner = model.fit(Xsrc[tr], y_all[tr], g_all[tr])
                else:
                    model = Teacher(kind)
                    inner = model.fit(Xsrc[tr], T[tr], y_all[tr], g_all[tr])
                thr = youden(y_all[tr], inner)
                sc[te] = model.score(Xte_all[te]); hd[te] = sc[te] >= thr
                fa = roc_auc_score(yc[te], sc[te])
                extra = {}
                if T is not None:
                    pt = model.ridge(Xte_all[te]); tt = T[cidx][te]
                    extra['student_r2_test_mean'] = float(np.mean(1 - ((pt - tt) ** 2).sum(0) / np.maximum(((tt - tt.mean(0)) ** 2).sum(0), 1e-12)))
                per_fold.append({'auc': float(fa), 'bacc': float(balanced_accuracy_score(yc[te], hd[te])), 'thr': thr,
                                 'n_train': int(len(tr)), 'n_test': int(len(te)), 'C': getattr(model, 'C', None) if T is None else model.clf.C, **extra})
                log(f'  {key} fold {k + 1}/5 ntr={len(tr)} nte={len(te)} AUC={fa:.4f} '
                    f'bACC={per_fold[-1]["bacc"]:.4f} {extra} ({time.time() - tick:.0f}s)')
            r = summarize(yc, gc, sc, hd, tmask)
            r.update({'per_fold': per_fold, 'seconds': time.time() - tick})
            if cname != 'a_canon6':
                a = f'{kind}|a_canon6'
                r['delta_vs_a'] = boot_delta(yc[tmask], gc[tmask], scores[a][tmask], sc[tmask])
                r['a_on_same_test'] = summarize(yc, gc, scores[a], hards[a], tmask)
                if cname.startswith('c_') or cname.endswith('_7s'):
                    b = f'{kind}|b_dense6'
                    r['delta_vs_b_same_subset'] = boot_delta(yc[tmask], gc[tmask], scores[b][tmask], sc[tmask])
            scores[key], hards[key] = sc, hd
            results['conditions'][key] = r
            dv = r.get('delta_vs_a')
            log(f'== {key}: AUC={r["auc"]:.4f} bACC={r["bacc"]:.4f} macroAUC={r["subject_macro_auc"]:.4f} '
                + (f'ΔAUC vs a={dv["delta_auc"]:+.4f} CI[{dv["ci95"][0]:+.4f},{dv["ci95"][1]:+.4f}]' if dv else ''))
            results['elapsed_seconds'] = time.time() - T0
            RES.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding='utf-8')
    np.savez_compressed(OOF, y=yc, groups=gc, fold_id=fold_id, has7=has7_c,
                        **{k.replace('|', '__'): v.astype('float32') for k, v in scores.items()})
    log(f'done features part, total {time.time() - T0:.0f}s')


# ---------------- ResNet1d (a) vs (b) ----------------
def main_resnet(seeds):
    import torch
    from torch import nn
    import raw_waveform_window6 as rw
    d = np.load(CACHE)
    meta = {k: d[k] for k in ('subject', 'label', 'canon')}
    cidx = np.where(meta['canon'])[0]
    raw6 = np.load(RAW_CACHE)
    X, _ = rw.normalize(raw6); del raw6
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Xg = torch.from_numpy(X).to(device); del X
    y_all, g_all = meta['label'].astype(int), meta['subject']
    yg = torch.tensor(y_all, dtype=torch.float32, device=device)
    yc, gc = y_all[cidx], g_all[cidx]
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(len(yc)), yc, gc))
    MAX_EP, PAT = 40, 8
    # separate files so it can run concurrently with the feature part; merged by --merge
    RES, OOF = OUT / 'dense_window_protocol_resnet_part.json', OUT / 'dense_window_protocol_resnet_part_oof.npz'
    res = {'conditions': {}}
    oof = {}
    log(f'resnet device={device} dense={len(y_all)} canon={len(cidx)} seeds={seeds}')
    for cname in ('a_canon6', 'b_dense6'):
        for seed in seeds:
            key = f'resnet1d_s{seed}|{cname}'
            sc, hd = np.full(len(yc), np.nan), np.zeros(len(yc), bool); tick = time.time(); per_fold = []
            for k, (trc, te) in enumerate(folds, 1):
                a, b = next(StratifiedGroupKFold(4, shuffle=True, random_state=k).split(trc, yc[trc], gc[trc]))
                itr_s, iva = set(gc[trc[a]]), cidx[trc[b]]
                pool = np.where((meta['canon'] if cname == 'a_canon6' else np.ones(len(y_all), bool))
                                & np.isin(g_all, list(itr_s)))[0]
                n_per_ep = len(a)  # same gradient steps per epoch as canonical training
                torch.manual_seed(1000 * seed + k); rng = np.random.default_rng(1000 * seed + k)
                gen = torch.Generator(device=device); gen.manual_seed(1000 * seed + k)
                net = rw.ResNet1d().to(device)
                opt = torch.optim.AdamW(net.parameters(), lr=rw.LR, weight_decay=rw.WD)
                crit = nn.BCEWithLogitsLoss(); best, state, stale, best_ep = -1., None, 0, 0
                for ep in range(1, MAX_EP + 1):
                    net.train(); perm = rng.permutation(pool)[:n_per_ep] if len(pool) > n_per_ep else rng.permutation(pool)
                    for bi in range(0, len(perm), rw.BS):
                        ids = perm[bi:bi + rw.BS]
                        if len(ids) < 2:
                            continue
                        idt = torch.as_tensor(ids, device=device)
                        loss = crit(net(rw.augment(Xg[idt], gen)).ravel(), yg[idt])
                        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.); opt.step()
                    auc = roc_auc_score(y_all[iva], rw.predict(net, Xg, iva))
                    if auc > best + 1e-4:
                        best, state, stale, best_ep = auc, {kk: v.clone() for kk, v in net.state_dict().items()}, 0, ep
                    else:
                        stale += 1
                    print(f'    [{key} f{k}] ep{ep:02d} valAUC={auc:.4f} best={best:.4f}@{best_ep} t={time.time() - tick:.0f}s', flush=True)
                    if stale >= PAT:
                        break
                net.load_state_dict(state)
                thr = youden(y_all[iva], rw.predict(net, Xg, iva))
                sc[te] = rw.predict(net, Xg, cidx[te]); hd[te] = sc[te] >= thr
                per_fold.append({'auc': float(roc_auc_score(yc[te], sc[te])), 'best_epoch': best_ep, 'thr': thr,
                                 'n_pool': int(len(pool)), 'best_val_auc': float(best)})
                log(f'  {key} fold {k}/5 AUC={per_fold[-1]["auc"]:.4f} pool={len(pool)} best_ep={best_ep}')
            r = summarize(yc, gc, sc, hd, np.ones(len(yc), bool)); r.update({'per_fold': per_fold, 'seconds': time.time() - tick})
            if cname == 'b_dense6':
                r['delta_vs_a'] = boot_delta(yc, gc, oof[f'resnet1d_s{seed}__a_canon6'], sc)
            oof[key.replace('|', '__')] = sc.astype('float32')
            res['conditions'][key] = r
            log(f'== {key}: AUC={r["auc"]:.4f} bACC={r["bacc"]:.4f} macro={r["subject_macro_auc"]:.4f}'
                + (f' Δ={r["delta_vs_a"]["delta_auc"]:+.4f} CI{r["delta_vs_a"]["ci95"]}' if 'delta_vs_a' in r else ''))
            res.setdefault('protocol', {})['resnet'] = (
                'raw_waveform_window6 ResNet1d, per-window z-score, random 4 s crop aug, 5-crop TTA; inner-val = canonical windows '
                'of 1/4 training subjects (early stop val AUC, patience 8, max 40 epochs, Youden threshold); '
                'each epoch draws n_canonical_inner_train samples from pool (canonical or dense 1 s stride)')
            RES.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
            np.savez_compressed(OOF, **oof)
    if len(seeds) > 1:
        ea = np.mean([oof[f'resnet1d_s{s}__a_canon6'] for s in seeds], 0)
        eb = np.mean([oof[f'resnet1d_s{s}__b_dense6'] for s in seeds], 0)
        res['conditions']['resnet1d_seedmean'] = {'a_auc': float(roc_auc_score(yc, ea)), 'b_auc': float(roc_auc_score(yc, eb)),
                                                  'delta_vs_a': boot_delta(yc, gc, ea, eb)}
        log(f'seed-mean resnet a={res["conditions"]["resnet1d_seedmean"]["a_auc"]:.4f} '
            f'b={res["conditions"]["resnet1d_seedmean"]["b_auc"]:.4f}')
        RES.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    log(f'done resnet part, total {time.time() - T0:.0f}s')


def merge():
    res = json.loads(RES.read_text(encoding='utf-8'))
    part = json.loads((OUT / 'dense_window_protocol_resnet_part.json').read_text(encoding='utf-8'))
    res['conditions'].update(part['conditions']); res['protocol']['resnet'] = part['protocol']['resnet']
    RES.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    oof = dict(np.load(OOF)); oof.update(dict(np.load(OUT / 'dense_window_protocol_resnet_part_oof.npz')))
    np.savez_compressed(OOF, **oof)
    (OUT / 'dense_window_protocol_resnet_part.json').unlink(); (OUT / 'dense_window_protocol_resnet_part_oof.npz').unlink()
    log('merged resnet part into results/oof')


if __name__ == '__main__':
    if '--merge' in sys.argv:
        merge()
    elif '--resnet' in sys.argv:
        main_resnet([int(s) for s in sys.argv[sys.argv.index('--resnet') + 1:]] or [0])
    else:
        main_features()
