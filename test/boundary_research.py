"""真实 EEG 交接边界研究；时间元数据只用于构造边界与审计，不进入模型输入。

``--raw`` 执行原始 BDF、事件对齐实验；旧版 JSON/PNG 保留不覆盖。
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch, sosfilt, sosfilt_zi, butter, iirnotch, tf2sos
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT/'test'
sys.path.insert(0, str(ROOT/'src')); sys.path.insert(0, str(OUT))
from causal_history import rebuild_meta
from cogstate import load_trials, bad_intervals, valid_window
from preprocessing import causal_filter, PreprocessingConfig

SEED = 20261029
FS = 250.0
BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))


def transform_fit(X, y, k=300):
    ok = np.isfinite(X).all(0) & (X.var(0) > 1e-20)
    ids = np.flatnonzero(ok)
    sel = SelectKBest(f_classif, k=min(k, len(ids))).fit(X[:, ids], y)
    ids = ids[sel.get_support()]
    sc = StandardScaler().fit(X[:, ids])
    return ids, sc


def report(y, p, threshold=.5):
    return {'ROC_AUC': float(roc_auc_score(y, p)), 'AP': float(average_precision_score(y, p)),
            'bACC': float(balanced_accuracy_score(y, p >= threshold)),
            'baseline_prevalence': float(np.mean(y)), 'fixed_threshold': threshold,
            'threshold_positive_rate': float(np.mean(p >= threshold))}


def subject_bootstrap(y, p, groups, n_boot=5000):
    rng = np.random.default_rng(SEED); subjects = np.unique(groups); vals = []
    for _ in range(n_boot):
        chosen = rng.choice(subjects, len(subjects), replace=True)
        ix = np.concatenate([np.flatnonzero(groups == s) for s in chosen])
        vals.append([roc_auc_score(y[ix], p[ix]), average_precision_score(y[ix], p[ix]),
                     balanced_accuracy_score(y[ix], p[ix] >= .5)])
    vals = np.asarray(vals)
    return {'n_subjects': int(len(subjects)), 'AUC_ci95': [float(np.quantile(vals[:, 0], .025)), float(np.quantile(vals[:, 0], .975))],
            'AP_ci95': [float(np.quantile(vals[:, 1], .025)), float(np.quantile(vals[:, 1], .975))],
            'bACC_ci95': [float(np.quantile(vals[:, 2], .025)), float(np.quantile(vals[:, 2], .975))]}


def band_features(x, fs=FS):
    x = np.asarray(x, dtype=float).ravel()
    if len(x) == 0 or not np.isfinite(x).all():
        raise ValueError('invalid window')
    # Window-local de-bias and steady-state initialization prevent a DC step
    # at the left edge from becoming a filter transient; no outside samples used.
    x = x - x[0]
    cfg = PreprocessingConfig(sfreq=fs, low_hz=1., high_hz=45., mains_hz=50., filter_order=4)
    sos = np.concatenate([butter(cfg.filter_order, [cfg.low_hz, cfg.high_hz], btype='bandpass', fs=fs, output='sos'),
                          tf2sos(*iirnotch(cfg.mains_hz, cfg.mains_q, fs=fs))])
    zi = sosfilt_zi(sos) * x[0]
    y, _ = sosfilt(sos, x, zi=zi)
    parts = [y, y[:len(y)//2], y[len(y)//2:]]
    feats = []
    for z in parts:
        f, p = welch(z, fs=fs, nperseg=min(len(z), int(2*fs)), noverlap=0, detrend='constant')
        bp = np.asarray([np.trapezoid(p[(f >= lo) & (f < hi)], f[(f >= lo) & (f < hi)]) for lo, hi in BANDS])
        total = float(np.sum(bp) + 1e-12)
        feats.extend(np.log(bp + 1e-12)); feats.extend(bp / total)
        q = p[(f >= 1) & (f < 45)] + 1e-12; q = q / q.sum()
        feats.append(float(-np.sum(q*np.log(q))))
        feats.extend([float(np.sqrt(np.mean(z*z))), float(np.ptp(z)), float(np.mean(np.diff(z)**2)),
                      float(np.sum(np.abs(np.diff(z))))])
    feats.extend([float(np.log((np.mean(y[:len(y)//2]**2)+1e-12)/(np.mean(y[len(y)//2:]**2)+1e-12))),
                  float(np.mean(np.abs(y[:len(y)//2]))/(np.mean(np.abs(y[len(y)//2:]))+1e-12))])
    return np.asarray(feats, dtype=np.float32)


def raw_events():
    import mne
    events = []; audits = []
    paths = sorted((ROOT/'EEG_dataset').glob('*/*_events.csv'))
    for ci, csv in enumerate(paths, 1):
        sid = csv.parent.name; bdf = next(csv.parent.glob('*.bdf'))
        raw = mne.io.read_raw_bdf(str(bdf), preload=True, verbose='ERROR')
        fs = float(raw.info['sfreq']); record_end = raw.n_times/fs; bad = bad_intervals(raw)
        trials = load_trials(str(csv), str(bdf))
        print(f'[raw {ci}/{len(paths)}] subject {sid}: trials={len(trials)} fs={fs:g} record={record_end:.2f}s', flush=True)
        for a, b in zip(trials[:-1], trials[1:]):
            if a.label == b.label or b.start < a.end - .05: continue
            boundary = float(b.start)
            events.append({'subject': sid, 'kind': 'real', 'source_label': a.label, 'target_label': b.label,
                           'boundary': boundary, 'trial_a': a.trial_id, 'trial_b': b.trial_id})
            audits.append({'subject': sid, 'a_end': a.end, 'b_start': b.start, 'gap': b.start-a.end})
        # Candidate pseudo-boundaries are centers of target-label blocks; +6s stays in-block.
        for tr in trials:
            if tr.end - tr.start >= 12.:
                boundary = tr.start + (tr.end - tr.start - 6.) / 2.
                events.append({'subject': sid, 'kind': 'pseudo', 'source_label': tr.label,
                                'target_label': tr.label, 'boundary': float(boundary),
                                'trial_a': tr.trial_id, 'trial_b': tr.trial_id})
        raw.close()
    return events, audits


def extract_raw(events):
    import mne
    by_sub = {}
    for csv in sorted((ROOT/'EEG_dataset').glob('*/*_events.csv')):
        sid = csv.parent.name; bdf = next(csv.parent.glob('*.bdf'))
        raw = mne.io.read_raw_bdf(str(bdf), preload=True, verbose='ERROR'); by_sub[sid] = raw
    offsets = (0., 3., 6.)
    rows = []
    for ei, ev in enumerate(events, 1):
        raw = by_sub[ev['subject']]; fs = float(raw.info['sfreq']); bad = bad_intervals(raw); rend = raw.n_times/fs
        for off in offsets:
            end = ev['boundary'] + off; start = end - 6.
            if not valid_window(start, end, bad, rend): continue
            s, z = int(np.ceil(start*fs - 1e-9)), int(np.floor(end*fs + 1e-9)); assert z / fs <= end + 1e-9
            x = raw.get_data(start=s, stop=z)[0]
            if len(x) != z-s or not np.isfinite(x).all(): continue
            rows.append({**ev, 'offset': off, 'x': band_features(x, fs)})
        if ei % 25 == 0: print(f'[extract] events {ei}/{len(events)}', flush=True)
    for raw in by_sub.values(): raw.close()
    # A pair is usable at an offset only when both real and pseudo windows pass.
    keep = {(r['pair_id'], r['offset']) for r in rows if r['kind'] == 'real'} & {(r['pair_id'], r['offset']) for r in rows if r['kind'] == 'pseudo'}
    return [r for r in rows if (r['pair_id'], r['offset']) in keep]


def raw_main():
    tick = time.perf_counter(); print('[raw] loading BDF events and constructing aligned windows', flush=True)
    events, audits = raw_events()
    rng = np.random.default_rng(SEED)
    real = [e for e in events if e['kind'] == 'real']; pseudos = [e for e in events if e['kind'] == 'pseudo']
    # Primary +6s compares the target-label block; pair without replacement within subject.
    selected = []; used = set()
    for e in real:
        pool = [p for p in pseudos if p['subject'] == e['subject'] and
                p['target_label'] == e['target_label'] and id(p) not in used]
        if pool:
            p = pool[int(rng.integers(len(pool)))]; used.add(id(p)); selected.append((e, p))
    real = [{**e, 'pair_id': i + 1} for i, (e, _) in enumerate(selected)]
    matched = [{**p, 'pair_id': i + 1} for i, (_, p) in enumerate(selected)]
    events = real + matched
    rows = extract_raw(events)
    offsets = (0., 3., 6.)
    data = {}
    for off in offsets:
        rr = [r for r in rows if r['offset'] == off]
        if not rr: continue
        data[off] = (np.stack([r['x'] for r in rr]), np.asarray([r['kind']=='real' for r in rr], int),
                     np.asarray([r['subject'] for r in rr]))
    results = {'method': {'mode': 'raw_bdf_event_aligned', 'window_seconds': 6,
        'window_end_offsets_seconds': list(offsets), 'primary_offsets_seconds': [6],
        'exploratory_offsets_seconds': [3],
        'offset_0_note': 'window ends at boundary and contains no future samples; descriptive only',
        'offset_3_note': 'exploratory cross-boundary window; state mixture cannot be fully matched',
        'offset_6_note': 'pre-specified primary: first complete target-label 6s window after event',
        'input': 'only raw EEG samples in the current 6s window; event times/labels/groups only for construction, supervision and evaluation',
        'filter': 'window-local causal 1-45Hz SOS bandpass plus 50Hz notch; no samples outside the 6s window read by feature extraction',
        'features': 'compact window-local log/relative five-band power, front/back 3s contrasts, RMS, peak-to-peak, derivative energy, line length and spectral entropy',
        'bad_segments': 'any window overlapping BAD_ACQ_SKIP or BDF bounds excluded as a whole pair', 'matching': 'same subject and same target label block; one fixed-seed pseudo per real boundary without replacement',
        'cv': '5-fold StratifiedGroupKFold by subject, random_state=0; fixed LogisticRegression C=1 and HistGradientBoosting parameters; SelectKBest and StandardScaler fit in training fold only',
        'sampling': 'balanced real/pseudo case-control sample; AP is not a real-stream prevalence estimate', 'seed': SEED},
        'counts': {'real_adjacent_transitions': len(real), 'matched_pseudo': len(matched), 'subjects': len(set(e['subject'] for e in real)),
                   'raw_rows_by_offset': {str(k): int(len(v[1])) for k,v in data.items()},
                   'bad_or_bounds_excluded_rows': int(len(events)*len(offsets)-sum(len(v[1]) for v in data.values()))},
        'alignment_audit': {'n_adjacent': len(audits), 'gap_seconds': [float(a['gap']) for a in audits],
                            'max_abs_gap_seconds': float(max([abs(a['gap']) for a in audits], default=0.))}}
    results['offset_descriptive'] = {}
    for off, (X, y, groups) in data.items():
        real_x = X[y == 1]; pseudo_x = X[y == 0]
        pooled_sd = np.sqrt((real_x.var(0, ddof=1) + pseudo_x.var(0, ddof=1)) / 2.0) + 1e-12
        results['offset_descriptive'][str(off)] = {'n': int(len(y)), 'real': int(y.sum()), 'pseudo': int((y==0).sum()),
            'standardized_paired_effect_by_feature': ((real_x.mean(0) - pseudo_x.mean(0)) / pooled_sd).tolist(),
            'interpretation': 'feature-wise standardized descriptive effects only; no cross-unit feature mean'}
    for off in (3., 6.):
        if off not in data: continue
        X, y, groups = data[off]; out = {}
        splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, groups))
        for model_name in ('LogisticRegression_C1', 'HistGradientBoosting_fixed'):
            p = np.full(len(y), np.nan)
            for fold, (tr, te) in enumerate(splits, 1):
                ids, sc = transform_fit(X[tr], y[tr], k=min(40, X.shape[1]))
                if model_name.startswith('Logistic'):
                    model = LogisticRegression(C=1, max_iter=1500, class_weight='balanced', random_state=0)
                    model.fit(sc.transform(X[tr][:, ids]), y[tr]); p[te] = model.predict_proba(sc.transform(X[te][:, ids]))[:,1]
                else:
                    model = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=15, learning_rate=.05, l2_regularization=1., random_state=0)
                    model.fit(sc.transform(X[tr][:, ids]), y[tr]); p[te] = model.predict_proba(sc.transform(X[te][:, ids]))[:,1]
                print(f'[raw][offset={off:g}][{model_name}] fold={fold}/5 ntrain={len(tr)} ntest={len(te)}', flush=True)
            out[model_name] = {**report(y, p), 'subject_bootstrap': subject_bootstrap(y, p, groups)}
        # Direction is an evaluation/descriptive split, never a model input.
        out['real_transition_direction'] = {}
        real_rows = [r for r in rows if r['offset']==off and r['kind']=='real']
        for name, pair in (('R_to_T',(0,1)), ('T_to_R',(1,0))):
            q = [r for r in real_rows if (r['source_label'], r['target_label']) == pair]
            out['real_transition_direction'][name] = {'n': len(q), 'subjects': len(set(r['subject'] for r in q)),
                'feature_mean': np.mean(np.stack([r['x'] for r in q]),0).tolist() if q else []}
        results['detector_offset_'+str(int(off))] = out
    # Plot only the two pre-specified primary offsets and both fixed models.
    fig, ax = plt.subplots(figsize=(7,4), dpi=150); names=[]; vals=[]
    for off in (3.,6.):
        key='detector_offset_'+str(int(off))
        if key in results:
            for model in ('LogisticRegression_C1','HistGradientBoosting_fixed'):
                names.append(f'+{int(off)}s\n{model[:4]}'); vals.append(results[key][model]['ROC_AUC'])
    ax.bar(np.arange(len(vals)), vals, color=['#4472c4','#ed7d31']*(len(vals)//2)); ax.axhline(.5,color='k',ls='--',lw=.8)
    ax.set_ylim(0,1); ax.set_ylabel('ROC AUC (subject OOF)'); ax.set_title('Raw BDF boundary detector (pre-specified +3s/+6s)'); ax.set_xticks(np.arange(len(vals)),names); fig.tight_layout()
    fig.savefig(OUT/'boundary_research_raw.png'); plt.close(fig)
    results['elapsed_seconds'] = time.perf_counter()-tick
    (OUT/'boundary_research_raw_results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


def legacy_main():
    # Preserve the previous cached-result path for existing users without rerunning raw BDF.
    path = OUT / 'boundary_research_results.json'
    if not path.exists():
        raise FileNotFoundError(f'cached legacy result not found: {path}')
    print(f'[cached] using legacy result: {path}', flush=True)
    print(path.read_text(encoding='utf-8'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--raw', action='store_true'); args = parser.parse_args()
    with threadpool_limits(limits=4):
        raw_main() if args.raw else legacy_main()
