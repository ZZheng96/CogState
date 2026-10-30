"""Switch-lag diagnosis + causal dual-memory change-point (CPD) state model.

Pre-registered main method (cpd_dual): per subject, causal stream over train-standardized 16-dim Welch features.
  Detector: at t, recent = last W=4 windows (incl. current), old = current segment before them (>= MIN_OLD=8);
    stat = W*n/(W+n) * mean_sq(mean_recent - mean_old); alarm if stat > h -> new segment starts at t-W+1.
    h chosen per outer fold on TRAINING subjects only (max detection F1, hit = alarm within [c, c+10] of a true switch).
  Features: [x, x - short(past<=8 in segment), x - long(prior-weighted expanding past mean),
             seg_mean - prev_seg_mean, min(t - seg_start, 60)]; HGB trained on train-fold streams built identically.
Ablations: no_reset (x, x-short8, x-long; never resets), current_only (x).
Labels/true switch positions are used only for training-fold h selection, HGB training and reporting.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from causal_kernel_state_mapping import features, metrics  # noqa: E402

W, MIN_OLD, SHORT, PRIOR, TSC_CAP, HIT = 4, 8, 8, 32., 60, 10
PRE, POST = 10, 20
FNAMES = ([f'logbp_{b}' for b in ('delta', 'theta', 'alpha', 'beta1', 'beta2', 'gamma1', 'hgamma')]
          + [f'rel_{b}' for b in ('delta', 'theta', 'alpha', 'beta1', 'beta2', 'gamma1', 'hgamma')]
          + ['spec_entropy', 'flatness'])


class Tee:
    def __init__(self, stream, path):
        self.stream, self.f = stream, open(path, 'w', encoding='utf-8')

    def write(self, s):
        self.stream.write(s); self.f.write(s)

    def flush(self):
        self.stream.flush(); self.f.flush()


def hgb():
    return HistGradientBoostingClassifier(max_iter=150, max_depth=3, learning_rate=.06,
                                          l2_regularization=1., random_state=0)


def cpd_stats(z):
    """Detector statistic sequence for one subject given a fixed (non-resetting) segment start rule is
    path-dependent, so stream() computes it online; this helper returns stats without resets for h grid."""
    out = np.zeros(len(z))
    for t in range(W + MIN_OLD - 1, len(z)):
        old, rec = z[:t - W + 1], z[t - W + 1:t + 1]
        n = len(old)
        out[t] = W * n / (W + n) * np.mean((rec.mean(0) - old.mean(0)) ** 2)
    return out


def stream(z, h, reset=True):
    """Causal per-subject stream. Row t depends only on z[:t+1]."""
    n, d = z.shape
    F = np.zeros((n, 4 * d + 1)); alarms = []
    seg0, prev_mean = 0, None
    csum = np.vstack([np.zeros(d), np.cumsum(z, 0)])
    for t in range(n):
        if reset:
            nold = t - W + 1 - seg0
            if nold >= MIN_OLD:
                old = (csum[t - W + 1] - csum[seg0]) / nold
                rec = (csum[t + 1] - csum[t - W + 1]) / W
                if W * nold / (W + nold) * np.mean((rec - old) ** 2) > h:
                    prev_mean = (csum[t - W + 1] - csum[seg0]) / nold
                    seg0 = t - W + 1; alarms.append(t)
        long = csum[t] / (t + PRIOR)  # prior mean 0 (train-standardized), current excluded
        lo = max(seg0, t - SHORT)
        short = (csum[t] - csum[lo]) / (t - lo) if t > lo else long
        seg = (csum[t + 1] - csum[seg0]) / (t + 1 - seg0)
        dseg = seg - prev_mean if prev_mean is not None else np.zeros(d)
        F[t] = np.r_[z[t], z[t] - short, z[t] - long, dseg, min(t - seg0, TSC_CAP)]
    return F, alarms


def build(z, g, h, variant):
    d = z.shape[1]
    F = np.zeros((len(z), 4 * d + 1)); n_alarm = 0
    for s in np.unique(g):
        ii = np.flatnonzero(g == s)
        F[ii], a = stream(z[ii], h, reset=(variant == 'cpd_dual')); n_alarm += len(a)
    if variant == 'cpd_dual':
        return F, n_alarm
    if variant == 'no_reset':
        return F[:, :3 * d], 0
    return z.copy(), 0


def switch_cuts(y_s):
    return np.flatnonzero(y_s[1:] != y_s[:-1]) + 1


def choose_h(z, y, g):
    """Grid of h from no-reset stat quantiles; pick max detection F1 on training subjects."""
    st = np.concatenate([cpd_stats(z[g == s]) for s in np.unique(g)])
    grid = np.unique(np.quantile(st[st > 0], np.linspace(.80, .995, 14)))
    best = None
    for h in grid:
        tp = fp = nsw = 0
        for s in np.unique(g):
            ii = np.flatnonzero(g == s)
            _, al = stream(z[ii], h); cuts = switch_cuts(y[ii]); nsw += len(cuts)
            used = set()
            for a in al:
                m = [c for c in cuts if c <= a <= c + HIT and c not in used]
                if m: used.add(m[0]); tp += 1
                else: fp += 1
        prec = tp / max(tp + fp, 1); rec = tp / max(nsw, 1); f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        if best is None or f1 > best[0]:
            best = (f1, float(h), prec, rec)
    return {'h': best[1], 'train_F1': best[0], 'train_precision': best[2], 'train_recall': best[3],
            'grid': [float(v) for v in grid]}


def masks(y, g):
    sw = np.zeros(len(y), bool); hist = np.zeros(len(y), bool)
    for s in np.unique(g):
        ii = np.flatnonzero(g == s)
        for c in switch_cuts(y[ii]): sw[ii[c:c + 5]] = True
        hist[ii[120:]] = True
    return {'all': np.ones(len(y), bool), 'switching': sw, 'steady': ~sw, 'history_comparable': hist}


def align(v, y, g):
    """Return per relative position rel in [-PRE, POST): mean score by direction, bACC, n."""
    rows = {'R->T': {r: [] for r in range(-PRE, POST)}, 'T->R': {r: [] for r in range(-PRE, POST)}}
    lab = {r: [] for r in range(-PRE, POST)}; val = {r: [] for r in range(-PRE, POST)}
    for s in np.unique(g):
        ii = np.flatnonzero(g == s); cuts = switch_cuts(y[ii]); bounds = np.r_[0, cuts, len(ii)]
        for k, c in enumerate(cuts):
            lo, hi = bounds[k], bounds[k + 2]
            key = 'R->T' if y[ii][c] == 1 else 'T->R'
            for r in range(-PRE, POST):
                t = c + r
                if lo <= t < hi and np.isfinite(v[ii[t]]):
                    rows[key][r].append(v[ii[t]]); lab[r].append(y[ii[t]]); val[r].append(v[ii[t]])
    out = {'rel': list(range(-PRE, POST))}
    for key in rows:
        out[key] = [float(np.mean(rows[key][r])) if rows[key][r] else None for r in out['rel']]
    out['bACC'] = [float(balanced_accuracy_score(lab[r], np.asarray(val[r]) >= .5)) for r in out['rel']]
    out['n'] = [len(lab[r]) for r in out['rel']]
    post = np.asarray(out['bACC'][PRE:])
    out['lag_first_bACC_gt_0.5'] = int(np.argmax(post > .5)) if (post > .5).any() else None
    out['lag_bACC_ge_0.9xsteady'] = None
    steady = float(np.mean(post[10:]))
    hit = np.flatnonzero(post >= .9 * steady)
    out['lag_bACC_ge_0.9xsteady'] = int(hit[0]) if len(hit) else None
    for key in ('R->T', 'T->R'):
        tr = np.asarray(out[key], float); a, b = np.nanmean(tr[:PRE]), np.nanmean(tr[PRE + 10:])
        frac = (tr[PRE:] - a) / (b - a)
        out[f'{key}_halfway_lag'] = int(np.argmax(frac >= .5)) if (frac >= .5).any() else None
    return out


def feature_steps(x, y, g):
    """Within-subject z-scored features; per-switch step = mean(post 3) - mean(pre 3), sign by direction R->T."""
    zs = np.empty_like(x)
    for s in np.unique(g):
        ii = g == s; zs[ii] = (x[ii] - x[ii].mean(0)) / (x[ii].std(0) + 1e-12)
    res = {}; traj = {f: {r: [] for r in range(-PRE, POST)} for f in range(x.shape[1])}
    steps = []; diff_sw = []; diff_ns = []
    for s in np.unique(g):
        ii = np.flatnonzero(g == s); cuts = switch_cuts(y[ii]); bounds = np.r_[0, cuts, len(ii)]
        dz = np.r_[np.nan, np.sqrt(np.mean(np.diff(zs[ii], axis=0) ** 2, 1))]
        cs = set(cuts.tolist())
        for t in range(1, len(ii)):
            (diff_sw if t in cs else diff_ns).append(dz[t])
        for k, c in enumerate(cuts):
            lo, hi = bounds[k], bounds[k + 2]
            if c - 3 < lo or c + 3 > hi: continue
            sign = 1 if y[ii][c] == 1 else -1
            steps.append(sign * (zs[ii][c:c + 3].mean(0) - zs[ii][c - 3:c].mean(0)))
            for r in range(-PRE, POST):
                if lo <= c + r < hi:
                    for f in range(x.shape[1]): traj[f][r].append(sign * zs[ii][c + r, f])
    steps = np.asarray(steps)
    for f, name in enumerate(FNAMES):
        tr = np.array([np.mean(traj[f][r]) for r in range(-PRE, POST)])
        a, b = tr[:PRE].mean(), tr[PRE + 10:].mean()
        frac = (tr[PRE:] - a) / (b - a) if abs(b - a) > 1e-9 else np.zeros(POST)
        res[name] = {'step_mean_z': float(steps[:, f].mean()), 'step_d': float(steps[:, f].mean() / steps[:, f].std()),
                     'pre_level': float(a), 'post_level': float(b), 'halfway_lag': int(np.argmax(frac >= .5)) if (frac >= .5).any() else None,
                     'aligned_traj_R->T_sign': [float(v) for v in tr]}
    det_auc = float(roc_auc_score(np.r_[np.ones(len(diff_sw)), np.zeros(len(diff_ns))], np.r_[diff_sw, diff_ns]))
    return {'per_feature': res, 'n_switches': int(len(steps)),
            'single_window_jump_AUC_switch_vs_nonswitch': det_auc}


def boot(a, b, y, g, mask, n=1000):
    subs = np.unique(g[mask]); pools = [np.flatnonzero((g == s) & mask) for s in subs]
    rng = np.random.default_rng(0); d = []
    for _ in range(n):
        pick = rng.integers(0, len(subs), len(subs)); ix = np.concatenate([pools[j] for j in pick])
        per = []
        for j in pick:
            q = pools[j]
            if np.unique(y[q]).size == 2:
                per.append(balanced_accuracy_score(y[q], a[q] >= .5) - balanced_accuracy_score(y[q], b[q] >= .5))
        d.append((roc_auc_score(y[ix], a[ix]) - roc_auc_score(y[ix], b[ix]), np.mean(per),
                  balanced_accuracy_score(y[ix], a[ix] >= .5) - balanced_accuracy_score(y[ix], b[ix] >= .5)))
    d = np.asarray(d)
    return {k: {'mean': float(d[:, j].mean()), 'p2.5': float(np.percentile(d[:, j], 2.5)),
                'p97.5': float(np.percentile(d[:, j], 97.5))} for j, k in enumerate(('dAUC', 'dmacro_bACC', 'dmicro_bACC'))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); args = ap.parse_args()
    stem = 'switch_dual_memory' + ('_smoke' if args.smoke else '')
    if not args.smoke:
        for suf in ('.log', '_results.json', '_oof.npz'):
            if (OUT / (stem + suf)).exists():
                raise SystemExit(f'refuse to overwrite formal artifact {stem + suf}')
    tee = Tee(sys.stdout, OUT / (stem + '.log')); sys.stdout = tee
    start = time.perf_counter()
    d = np.load(OUT / 'highdim_deep_features.npz'); y, g = d['y'].astype(int), d['groups']
    x = features(d['raw']); M = masks(y, g)
    mb = np.load(OUT / 'merge_best_window6_oof.npz'); assert np.array_equal(mb['y'], y) and np.array_equal(mb['groups'], g)
    ref = {'merge_best': expit(mb['combo__C4_bb_resnet_eq'])}
    hc = np.load(OUT / 'context_score_calibration_oof.npz')
    sys.path.insert(0, str(OUT)); from context_fixed_histgb import smooth
    ref['hist_best_d100_200_400_s5'] = smooth(np.mean([hc[f'd{k}_mean_std'] for k in (100, 200, 400)], 0), g, 5)
    print(f'[start] rows={len(y)} subjects={len(np.unique(g))} smoke={args.smoke}', flush=True)
    result = {'meta': {'W': W, 'MIN_OLD': MIN_OLD, 'SHORT': SHORT, 'PRIOR': PRIOR, 'TSC_CAP': TSC_CAP, 'HIT': HIT,
                       'threshold': .5, 'hist_best_note': 'context_score_calibration_oof d100/200/400_mean_std prob mean + causal smooth5; '
                       'long history (>6s), uses old float-cumsum causal_mean_std; reference only'},
              'diagnosis': {}, 'folds': [], 'checks': [], 'methods': {}, 'bootstrap': {}}
    # ---- diagnosis
    for name, v in ref.items():
        result['diagnosis'][name] = align(v, y, g)
        a = result['diagnosis'][name]
        print(f'[diag] {name} lag(bACC>0.5)={a["lag_first_bACC_gt_0.5"]} lag(>=0.9 steady)={a["lag_bACC_ge_0.9xsteady"]} '
              f'halfway R->T={a["R->T_halfway_lag"]} T->R={a["T->R_halfway_lag"]} bACC rel0..9='
              + ' '.join(f'{b:.2f}' for b in a['bACC'][PRE:PRE + 10]), flush=True)
    fs = feature_steps(x, y, g); result['diagnosis']['feature_steps'] = fs
    print(f'[diag] single-window jump AUC (switch vs non-switch) = {fs["single_window_jump_AUC_switch_vs_nonswitch"]:.4f}', flush=True)
    for nm, r in sorted(fs['per_feature'].items(), key=lambda kv: -abs(kv[1]['step_d'])):
        print(f'[diag] {nm:14s} step_d={r["step_d"]:+.3f} pre={r["pre_level"]:+.3f} post={r["post_level"]:+.3f} halfway_lag={r["halfway_lag"]}', flush=True)
    # ---- model
    variants = ('cpd_dual', 'no_reset', 'current_only')
    scores = {v: np.full(len(y), np.nan) for v in variants}
    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(x, y, g))
    with threadpool_limits(limits=4):
        for fold, (tr, te) in enumerate(splits):
            assert not set(g[tr]) & set(g[te])
            if args.smoke:
                tr = tr[np.isin(g[tr], np.unique(g[tr])[:6])]; te = te[np.isin(g[te], np.unique(g[te])[:2])]
            sc = StandardScaler().fit(x[tr]); ztr, zte = sc.transform(x[tr]), sc.transform(x[te])
            hsel = choose_h(ztr, y[tr], g[tr])
            print(f'[fold {fold+1}/5] h={hsel["h"]:.4f} trainF1={hsel["train_F1"]:.3f} P={hsel["train_precision"]:.3f} '
                  f'R={hsel["train_recall"]:.3f} elapsed={time.perf_counter()-start:.1f}s', flush=True)
            info = {'fold': fold, **hsel}
            for v in variants:
                t0 = time.perf_counter()
                Ftr, _ = build(ztr, g[tr], hsel['h'], v); Fte, na = build(zte, g[te], hsel['h'], v)
                if v == 'cpd_dual':
                    info['test_alarms'] = na; info['test_switches'] = int(sum(len(switch_cuts(y[te][g[te] == s])) for s in np.unique(g[te])))
                    # prefix invariance + subject isolation
                    for s in np.unique(g[te]):
                        ii = np.flatnonzero(g[te] == s); cut = len(ii) // 2
                        alt = zte[ii].copy(); alt[cut:] = np.random.default_rng(fold).normal(size=alt[cut:].shape) * 5
                        Fa, _ = stream(alt, hsel['h']); Fp, _ = stream(zte[ii][:cut], hsel['h'])
                        assert np.array_equal(Fa[:cut], Fte[ii][:cut]) and np.array_equal(Fp, Fte[ii][:cut])
                        assert np.array_equal(stream(zte[ii], hsel['h'])[0], Fte[ii])  # isolation: alone == in batch
                    result['checks'].append({'fold': fold, 'prefix_invariance': True, 'subject_isolation': True})
                model = hgb().fit(Ftr, y[tr]); p = model.predict_proba(Fte)[:, 1]
                if v == 'cpd_dual':
                    s = np.unique(g[te])[0]; ii = np.flatnonzero(g[te] == s); cut = len(ii) // 2
                    alt = zte.copy(); alt[ii[cut:]] += 3.
                    Falt, _ = build(alt, g[te], hsel['h'], v); palt = model.predict_proba(Falt)[:, 1]
                    other = g[te] != s
                    assert np.array_equal(palt[ii[:cut]], p[ii[:cut]]) and np.array_equal(palt[other], p[other])
                scores[v][te] = p
                print(f'[fold {fold+1}/5] method={v} dim={Ftr.shape[1]} fold_s={time.perf_counter()-t0:.1f} '
                      f'elapsed={time.perf_counter()-start:.1f}s', flush=True)
            result['folds'].append(info)
            if args.smoke: break
    ev = np.isfinite(scores['cpd_dual'])
    allscores = {**scores, **ref}
    for name, v in allscores.items():
        result['methods'][name] = {k: metrics(y, g, v, ev & m) for k, m in M.items()}
        r = result['methods'][name]
        print(f'[result] {name:28s} ' + ' | '.join(f'{k}: AUC={r[k]["AUC"] or float("nan"):.4f} mac={r[k]["subject_macro_bACC"] or float("nan"):.4f} '
                                            f'mic={r[k]["micro_bACC"]:.4f}' for k in ('all', 'switching', 'steady', 'history_comparable')), flush=True)
    for name in ('cpd_dual', 'no_reset'):
        result['diagnosis'][name] = align(np.where(ev, scores[name], np.nan), y, g)
        a = result['diagnosis'][name]
        print(f'[diag] {name} lag(bACC>0.5)={a["lag_first_bACC_gt_0.5"]} bACC rel0..9=' + ' '.join(f'{b:.2f}' for b in a['bACC'][PRE:PRE + 10]), flush=True)
    for b in ('merge_best', 'hist_best_d100_200_400_s5', 'no_reset', 'current_only'):
        for seg in ('all', 'switching', 'steady', 'history_comparable'):
            m = ev & M[seg] & np.isfinite(allscores[b])
            key = f'cpd_dual_vs_{b}__{seg}'; result['bootstrap'][key] = bb = boot(scores['cpd_dual'], allscores[b], y, g, m, 200 if args.smoke else 1000)
            print(f'[boot] {key} dAUC={bb["dAUC"]["mean"]:+.4f}[{bb["dAUC"]["p2.5"]:+.4f},{bb["dAUC"]["p97.5"]:+.4f}] '
                  f'dmac={bb["dmacro_bACC"]["mean"]:+.4f}[{bb["dmacro_bACC"]["p2.5"]:+.4f},{bb["dmacro_bACC"]["p97.5"]:+.4f}]', flush=True)
    result['seconds'] = time.perf_counter() - start
    np.savez_compressed(OUT / (stem + '_oof.npz'), y=y, groups=g, eval_mask=ev, **{k: v for k, v in scores.items()},
                        **{f'mask_{k}': v for k, v in M.items()})
    (OUT / (stem + '_results.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[done] elapsed={result["seconds"]:.1f}s', flush=True)
    sys.stdout = tee.stream; tee.f.close()


if __name__ == '__main__':
    main()
