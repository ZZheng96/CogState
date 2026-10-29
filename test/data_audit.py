"""数据/标签审计：BDF 头、事件对齐、标签平移扫描、信号质量、按键肌电瞬态。
输出 test/data_audit_results.json。
"""
import json, sys, time
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import mne
from scipy.signal import welch, butter, sosfiltfilt
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT/'test'
sys.path.insert(0, str(ROOT/'src'))
from cogstate import load_trials, bad_intervals, valid_window, _file_start

WIN, STRIDE = 6., 3.
DELTAS = [-60, -30, -20, -15, -10, -8, -6, -4, -2, -1, 0, 1, 2, 4, 6, 8, 10, 15, 20, 30, 60]
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45), (45, 100)]
FEAT_NAMES = [f'logbp_{a}_{b}' for a, b in BANDS]
LOW_AUC = {'029': .354, '024': .441, '004': .467, '009': .472}


def parse_header(path):
    with open(path, 'rb') as fh:
        h = fh.read(256)
        ns = int(h[252:256].decode('ascii', 'replace').strip())
        s = fh.read(256*ns)
    def field(off, w):
        return [s[off+i*w: off+(i+1)*w].decode('ascii', 'replace').strip() for i in range(ns)]
    o = 0; sig = {}
    for name, w in [('label', 16), ('transducer', 80), ('physdim', 8), ('physmin', 8), ('physmax', 8),
                    ('digmin', 8), ('digmax', 8), ('prefilter', 80), ('nsamples', 8), ('reserved', 32)]:
        sig[name] = field(o, w); o += w*ns
    return {'version': h[1:8].decode('ascii', 'replace').strip(),
            'patient': h[8:88].decode('ascii', 'replace').strip(),
            'recording': h[88:168].decode('ascii', 'replace').strip(),
            'startdate': h[168:176].decode('ascii', 'replace'), 'starttime': h[176:184].decode('ascii', 'replace'),
            'reserved': h[192:236].decode('ascii', 'replace').strip(),
            'n_records': int(h[236:244].decode().strip()), 'record_dur': float(h[244:252].decode().strip()),
            'n_signals': ns, 'signals': sig}


def bp_feats(x, fs):
    f, p = welch(x, fs=fs, nperseg=int(2*fs), noverlap=int(fs), detrend='linear')
    return np.array([np.log(np.trapezoid(p[(f >= a) & (f < b)], f[(f >= a) & (f < b)]) + 1e-30) for a, b in BANDS])


def label_at(trials, t0, t1, d):
    for tr in trials:
        if tr.start+d <= t0 and t1 <= tr.end+d:
            return tr.label
    return -1


def main():
    tick = time.perf_counter()
    paths = sorted((ROOT/'EEG_dataset').glob('*/*_events.csv'))
    res = {'header': {}, 'alignment': {}, 'quality': {}, 'keypress': {}}
    headers, rows = {}, []
    win_feat, win_sub, win_lab = [], [], {d: [] for d in DELTAS}
    kp_profiles = []
    lags = np.arange(-5, 5.01, .25)

    for i, csv in enumerate(paths, 1):
        sub = csv.parent.name
        bdf = next(csv.parent.glob('*.bdf'))
        hdr = parse_header(bdf)
        raw = mne.io.read_raw_bdf(str(bdf), preload=True, verbose='ERROR')
        fs = float(raw.info['sfreq']); n = raw.n_times; dur = n/fs
        ch = raw.info['chs'][0]
        headers[sub] = {'ch_names': raw.ch_names, 'ch_type': raw.get_channel_types(), 'unit_code': int(ch['unit']),
                        'loc_nonzero': bool(np.any(np.nan_to_num(ch['loc']) != 0)),
                        'highpass': raw.info['highpass'], 'lowpass': raw.info['lowpass'], 'sfreq': fs,
                        'meas_date': str(raw.info['meas_date']), 'dig': None if raw.info['dig'] is None else len(raw.info['dig']),
                        **{k: hdr[k] for k in ('version', 'patient', 'recording', 'startdate', 'starttime', 'reserved', 'n_records', 'record_dur', 'n_signals')},
                        **{f'sig_{k}': v for k, v in hdr['signals'].items()}}
        x = raw.get_data()[0]
        bad = bad_intervals(raw)
        trials = load_trials(str(csv), str(bdf))
        ev = pd.read_csv(csv)
        ts = pd.to_datetime(ev.local_time, utc=True).astype('int64').to_numpy()/1e9 - _file_start(str(bdf))

        # ---- 2. alignment
        hdr_t = pd.Timestamp(f"20{hdr['startdate'][6:8]}-{hdr['startdate'][3:5]}-{hdr['startdate'][0:2]} "
                             f"{hdr['starttime'].replace('.', ':')}").timestamp() - 8*3600
        blocks = [{'label': t.label, 'start': round(t.start, 3), 'end': round(t.end, 3), 'dur': round(t.end-t.start, 3)} for t in trials]
        gaps = [round(trials[k+1].start-trials[k].end, 4) for k in range(len(trials)-1)]
        anomalies = []
        if abs(trials[0].start) > 2: anomalies.append(f'first trial_start offset {trials[0].start:.2f}s')
        if any(abs(b['dur']-300) > 5 for b in blocks): anomalies.append('block duration !=300s')
        if [b['label'] for b in blocks] != [0, 1, 0, 1, 0, 1]: anomalies.append(f"order {[b['label'] for b in blocks]}")
        if trials[-1].end > dur: anomalies.append(f'last trial_end {trials[-1].end:.2f}s > rec {dur:.2f}s')
        if any(abs(g) > 1 for g in gaps): anomalies.append(f'gaps {gaps}')
        if abs(hdr_t - _file_start(str(bdf))) > 1.5: anomalies.append(f'header start differs from filename by {hdr_t-_file_start(str(bdf)):.2f}s')
        if abs(n/fs - hdr['n_records']*hdr['record_dur']) > 1: anomalies.append('n_times vs header records mismatch')
        if dur - trials[-1].end > 30: anomalies.append(f'recording extends {dur-trials[-1].end:.1f}s after last trial_end')
        res['alignment'][sub] = {'first_start': round(trials[0].start, 3), 'last_end': round(trials[-1].end, 3),
                                 'rec_dur': round(dur, 3), 'tail_after_last_end': round(dur-trials[-1].end, 3),
                                 'header_minus_filename_s': round(hdr_t-_file_start(str(bdf)), 3),
                                 'blocks': blocks, 'gaps': gaps, 'n_keypress': int((ev.event == 'key_press').sum()),
                                 'n_task_failure': int((ev.event == 'task_failure').sum()), 'anomalies': anomalies}

        # ---- 3. windows on a fixed grid
        t0 = 0.
        while t0+WIN <= dur:
            t1 = t0+WIN
            if valid_window(t0, t1, bad, dur):
                labs = {d: label_at(trials, t0, t1, d) for d in DELTAS}
                if any(v >= 0 for v in labs.values()):
                    seg = x[int(round(t0*fs)): int(round(t1*fs))]
                    win_feat.append(bp_feats(seg, fs)); win_sub.append(sub)
                    for d in DELTAS: win_lab[d].append(labs[d])
            t0 += STRIDE

        # ---- 4. quality
        uv = x*1e6
        pmin, pmax = float(hdr['signals']['physmin'][0]), float(hdr['signals']['physmax'][0])
        rng = pmax-pmin; xs = uv if hdr['signals']['physdim'][0].lower().startswith('u') else x
        f, p = welch(x, fs=fs, nperseg=int(4*fs))
        tot = np.trapezoid(p[(f >= 1) & (f < 100)], f[(f >= 1) & (f < 100)])
        p50 = np.trapezoid(p[(f >= 49) & (f <= 51)], f[(f >= 49) & (f <= 51)])
        hp = sosfiltfilt(butter(4, 1, 'highpass', fs=fs, output='sos'), x)
        def rms_lab(lab):
            m = np.zeros(n, bool)
            for t in trials:
                if t.label == lab: m[int(t.start*fs): int(t.end*fs)] = True
            for a, b in bad: m[int(a*fs): int(b*fs)] = False
            v = hp[m]; return float(np.sqrt(np.median(v**2)))  # median-based robust RMS
        bad_frac = sum(b-a for a, b in bad)/dur
        res['quality'][sub] = {'amp_pct_uV': {q: float(np.percentile(uv, q)) for q in (0.1, 1, 5, 50, 95, 99, 99.9)},
                               'hp_std_uV': float(hp.std()*1e6), 'flat_frac': float(np.mean(np.diff(x) == 0)),
                               'sat_frac': float(np.mean((xs <= pmin+.001*rng) | (xs >= pmax-.001*rng))),
                               'line50_ratio': float(p50/tot), 'log_emg_45_100': float(np.log(np.trapezoid(p[(f >= 45) & (f < 100)], f[(f >= 45) & (f < 100)]))),
                               'bad_frac': float(bad_frac), 'n_bad': len(bad),
                               'log_rms_task_over_relax': float(np.log(rms_lab(1)/rms_lab(0)))}

        # ---- 5. keypress EMG transients
        emg = sosfiltfilt(butter(4, [45, 100], 'bandpass', fs=fs, output='sos'), x)**2
        task_mask = np.zeros(n, bool)
        for t in trials:
            if t.label == 1: task_mask[int(t.start*fs): int(min(t.end, dur)*fs)] = True
        for a, b in bad: task_mask[int(a*fs): int(b*fs)] = False
        kt = ts[ev.event.isin(['key_press', 'task_failure']).to_numpy()]
        near = np.zeros(n, bool)
        for k in kt:
            near[max(0, int((k-2)*fs)): int((k+2)*fs)] = True
        a_near, a_far = emg[near & task_mask], emg[~near & task_mask]
        prof = []
        for k in kt:
            row = []
            for L in lags:
                s0 = int((k+L-.125)*fs); s1 = int((k+L+.125)*fs)
                row.append(np.median(emg[s0:s1]) if 0 <= s0 and s1 <= n and task_mask[s0:s1].all() else np.nan)
            prof.append(row)
        prof = np.array(prof)
        if len(prof):
            base = np.nanmedian(prof)
            sp = np.log(np.nanmedian(prof, 0)/base) if np.isfinite(base) and base > 0 else np.full(len(lags), np.nan)
            kp_profiles.append(sp)
        else:
            sp = np.full(len(lags), np.nan)
        res['keypress'][sub] = {'n_events': int(len(kt)),
                                'log_ratio_near_vs_other': float(np.log(np.median(a_near)/np.median(a_far))) if len(a_near) and len(a_far) else None,
                                'peak_lag_s': float(lags[np.nanargmax(sp)]) if np.isfinite(sp).any() else None,
                                'peak_log_rel': float(np.nanmax(sp)) if np.isfinite(sp).any() else None}
        raw.close()
        print(f'[{i}/{len(paths)}] sub-{sub} dur={dur:.0f}s first={trials[0].start:.2f}s windows={len(win_sub)} '
              f'anom={len(anomalies)} elapsed={time.perf_counter()-tick:.0f}s', flush=True)

    # ---- 1. header summary
    summ = {}
    for k in next(iter(headers.values())):
        vals = Counter(json.dumps(h[k]) for h in headers.values())
        summ[k] = vals.most_common(10) if k not in ('startdate', 'starttime', 'meas_date', 'n_records') else f'{len(vals)} unique'
    res['header'] = {'unique_values': summ}

    # ---- 3. label shift scan
    X = np.array(win_feat); G = np.array(win_sub)
    shift = {}
    sub_auc0 = {}
    for d in DELTAS:
        y = np.array(win_lab[d]); m = y >= 0
        Xd, yd, gd = X[m], y[m], G[m]
        oof = np.zeros(len(yd))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Xd, yd, gd):
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xd[tr], yd[tr])
            oof[te] = clf.predict_proba(Xd[te])[:, 1]
        sa, fe = {}, []
        for s in np.unique(gd):
            mm = gd == s
            if len(np.unique(yd[mm])) == 2:
                sa[s] = roc_auc_score(yd[mm], oof[mm])
                fe.append([abs(roc_auc_score(yd[mm], Xd[mm, j])-.5) for j in range(X.shape[1])])
        fe = np.array(fe)
        shift[d] = {'n': int(m.sum()), 'pooled_auc': float(roc_auc_score(yd, oof)), 'mean_subj_auc': float(np.mean(list(sa.values()))),
                    'within_feat_absdev': float(fe.mean()), 'within_feat_absdev_per_feat': dict(zip(FEAT_NAMES, fe.mean(0).round(4).tolist())),
                    'within_best_feat_absdev': float(fe.max(1).mean())}
        if d == 0: sub_auc0 = sa
        print(f'[shift {d:+d}] n={m.sum()} pooled={shift[d]["pooled_auc"]:.3f} subj={shift[d]["mean_subj_auc"]:.3f} '
              f'within={shift[d]["within_feat_absdev"]:.4f} elapsed={time.perf_counter()-tick:.0f}s', flush=True)
    res['label_shift'] = shift
    res['subject_auc_delta0'] = {k: float(v) for k, v in sub_auc0.items()}

    # ---- 4. correlations & outliers
    subs = sorted(sub_auc0)
    qkeys = ['hp_std_uV', 'flat_frac', 'sat_frac', 'line50_ratio', 'log_emg_45_100', 'bad_frac', 'log_rms_task_over_relax']
    corr = {}
    for k in qkeys:
        v = [res['quality'][s][k] for s in subs]
        r, pv = spearmanr([sub_auc0[s] for s in subs], v)
        corr[k] = {'rho': float(r), 'p': float(pv)}
    ks = [s for s in subs if res['keypress'][s]['log_ratio_near_vs_other'] is not None]
    r, pv = spearmanr([sub_auc0[s] for s in ks], [res['keypress'][s]['log_ratio_near_vs_other'] for s in ks]); corr['keypress_emg_log_ratio'] = {'rho': float(r), 'p': float(pv)}
    res['quality_auc_spearman'] = corr
    outl = {}
    for k in qkeys + ['p99.9_abs']:
        if k == 'p99.9_abs':
            v = np.array([max(abs(res['quality'][s]['amp_pct_uV'][0.1]), abs(res['quality'][s]['amp_pct_uV'][99.9])) for s in subs])
        else:
            v = np.array([res['quality'][s][k] for s in subs])
        med, mad = np.median(v), np.median(np.abs(v-np.median(v)))+1e-12
        z = (v-med)/(1.4826*mad)
        outl[k] = {s: round(float(zz), 2) for s, zz in zip(subs, z) if abs(zz) > 3.5}
    res['quality_outliers_robust_z>3.5'] = outl
    res['low_auc_subjects'] = {s: {'baseline_auc': LOW_AUC[s], 'audit_auc_d0': float(sub_auc0.get(s, np.nan)),
                                   'quality': res['quality'][s], 'keypress': res['keypress'][s],
                                   'quality_rank_pct': {k: float(np.mean(np.array([res['quality'][t][k] for t in subs]) <= res['quality'][s][k])) for k in qkeys}}
                               for s in LOW_AUC if s in res['quality']}
    P = np.array(kp_profiles)
    res['keypress_grand_profile'] = {'lags_s': lags.round(2).tolist(), 'median_log_rel': np.nanmedian(P, 0).round(4).tolist(),
                                     'n_subjects': int(len(P))}
    kr = np.array([res['keypress'][s]['log_ratio_near_vs_other'] for s in subs if res['keypress'][s]['log_ratio_near_vs_other'] is not None])
    res['keypress_summary'] = {'median_log_ratio': float(np.median(kr)), 'frac_positive': float(np.mean(kr > 0)),
                               'n_subjects_with_keypress': int(len(kr)),
                               'peak_lag_counts': Counter(res['keypress'][s]['peak_lag_s'] for s in subs).most_common(8)}
    res['alignment_anomalies'] = {s: a['anomalies'] for s, a in res['alignment'].items() if a['anomalies']}
    (OUT/'data_audit_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print(f'[done] elapsed={time.perf_counter()-tick:.0f}s', flush=True)


if __name__ == '__main__':
    main()
