"""Read-only independent BDF/CSV audit; outputs only beside this script."""
import csv
import json
import re
from pathlib import Path
from datetime import datetime, timezone, timedelta

import mne
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def audit(sid):
    folder = ROOT / 'EEG_dataset' / sid
    bdf = next(folder.glob('*.bdf'))
    csv_path = folder / f'{sid}_events.csv'
    stamp = re.search(r'_(\d{8})_(\d{6}\.\d{3})_UTC\+0800', bdf.name)
    origin = pd.Timestamp(datetime.strptime(''.join(stamp.groups()), '%Y%m%d%H%M%S.%f').replace(tzinfo=timezone(timedelta(hours=8))))
    events = list(csv.DictReader(csv_path.open(encoding='utf-8-sig', newline='')))
    trials, issues, pending = [], [], None
    last_ns = None
    for rowno, row in enumerate(events, 2):
        ts = pd.Timestamp(row['local_time'])
        if last_ns is not None and ts.value < last_ns:
            issues.append({'row': rowno, 'issue': 'nonmonotonic event'})
        last_ns = ts.value
        if row['event'] == 'trial_start':
            if pending is not None:
                issues.append({'row': rowno, 'issue': 'nested start'})
            pending = (rowno, row, ts)
        elif row['event'] == 'trial_end':
            if pending is None:
                issues.append({'row': rowno, 'issue': 'unpaired end'})
                continue
            start_row, start, start_ts = pending
            if start['phase'] != row['phase'] or ts <= start_ts:
                issues.append({'row': rowno, 'issue': 'phase mismatch or nonpositive duration'})
            a, b = (start_ts.value-origin.value)/1e9, (ts.value-origin.value)/1e9
            trials.append(dict(blk=len(trials), phase=start['phase'], y={'relax': 0, 'task': 1}[start['phase']], start=a, end=b, duration=b-a, csv_start_row=start_row, csv_end_row=rowno, local_start=start['local_time'], local_end=row['local_time']))
            pending = None
    if pending is not None:
        issues.append({'issue': 'unpaired final start'})
    raw = mne.io.read_raw_bdf(str(bdf), preload=False, verbose='ERROR')
    fs = float(raw.info['sfreq'])
    rec_end = raw.n_times/fs
    annotations = [dict(start=float(a), end=float(a+d), description=str(s)) for a, d, s in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description)]
    bad = [a for a in annotations if 'BAD_ACQ_SKIP' in a['description']]
    with bdf.open('rb') as f:
        header = f.read(256)
    header_date, header_time = header[168:176].decode(), header[176:184].decode()
    expected, rejected = [], []
    for tr in trials:
        count = max(0, int(np.floor(tr['duration'] - 6 + 1e-6)) + 1)
        for k in range(count):
            end = tr['start'] + 6 + k
            start = end-6
            reason = []
            if start < 0 or end > rec_end:
                reason.append('record_boundary')
            if any(start < a['end'] and end > a['start'] for a in bad):
                reason.append('BAD_ACQ_SKIP')
            item = dict(t=end, y=tr['y'], blk=tr['blk'])
            if reason:
                rejected.append(dict(**item, reasons=reason))
            else:
                expected.append(item)
    cache_path = ROOT / 'test/11_individual_20261102/sliding1' / f'sub-{sid}.npz'
    with np.load(cache_path) as cache:
        original_t = cache['t'].copy()
        order = np.argsort(original_t)
        t, y, blk = (cache[key][order] for key in ('t', 'y', 'blk'))
    rows = []
    mismatch = dict(label=0, block=0, unmatched_time=0, nominal_phase_boundary=0, record_boundary=0, bad_overlap=0, sample_phase_boundary=0, sample_bad_overlap=0, nonfinite_signal_window=0)
    matched = set()
    max_err = 0.0
    et = np.array([r['t'] for r in expected])
    for i, (end, label, block) in enumerate(zip(t.astype(float), y, blk)):
        j = int(np.argmin(abs(et-end))) if len(et) else -1
        error = abs(et[j]-end) if j >= 0 else float('inf')
        # Cache stores float32 seconds; allow one ULP plus epoch-float error.
        tol = max(float(np.spacing(np.float32(end))), 1e-6)
        ref = expected[j] if error <= tol else None
        flags = []
        if ref is None:
            mismatch['unmatched_time'] += 1
            flags.append('unmatched_time')
            nominal_end = end
            tr = None
        else:
            matched.add(j)
            max_err = max(max_err, error)
            nominal_end = ref['t']
            tr = trials[ref['blk']]
            for key, observed in [('label', int(label)), ('block', int(block))]:
                if observed != ref['y' if key == 'label' else 'blk']:
                    mismatch[key] += 1
                    flags.append(key)
        a = nominal_end - 6
        s, e = round(a*fs), round(nominal_end*fs)
        checks = dict(nominal_phase_boundary=tr is None or a < tr['start']-1e-8 or nominal_end > tr['end']+1e-8,
                      record_boundary=a < 0 or nominal_end > rec_end,
                      bad_overlap=any(a < b['end'] and nominal_end > b['start'] for b in bad),
                      sample_phase_boundary=tr is None or s/fs < tr['start']-1e-9 or (e-1)/fs >= tr['end'],
                      sample_bad_overlap=any(s/fs < b['end'] and (e-1)/fs >= b['start'] for b in bad))
        x = raw.get_data(start=max(0,s), stop=min(raw.n_times,e))[0]
        checks['nonfinite_signal_window'] = not np.isfinite(x).all() or len(x) != 1500
        for key, value in checks.items():
            if value:
                mismatch[key] += 1
                flags.append(key)
        rows.append(dict(sorted_index=i, original_index=int(order[i]), cache_t=end, cache_y=int(label), cache_blk=int(block), independent_t=nominal_end, sample_start=s, sample_stop_exclusive=e, flags=flags))
    for tr in trials:
        selected = [r for r in rows if r['cache_blk'] == tr['blk']]
        tr['cached_windows'] = len(selected)
        tr['expected_windows'] = sum(r['blk'] == tr['blk'] for r in expected)
        tr['rejected_windows'] = sum(r['blk'] == tr['blk'] for r in rejected)
        tr['first_last_cache_t'] = [selected[0]['cache_t'], selected[-1]['cache_t']] if selected else []
    result = dict(subject=sid, bdf=str(bdf), csv=str(csv_path), cache=str(cache_path), first_sample_time=origin.isoformat(), bdf_header_date=header_date, bdf_header_time=header_time, sfreq=fs, n_samples=raw.n_times, record_duration_s=rec_end, annotations=annotations, event_issues=issues, trials=trials, cached_windows=len(t), expected_windows=len(expected), missing_expected_windows=[r for j,r in enumerate(expected) if j not in matched], rejected_candidates=rejected, mismatches=mismatch, max_cache_time_rounding_error_s=max_err, duplicate_cache_times=int(len(t)-len(np.unique(t))), reorder_count=int(np.count_nonzero(order != np.arange(len(t)))), windows=rows)
    raw.close()
    return result


def main():
    results = [audit(sid) for sid in ('004','009','050','018')]
    report = dict(method='Independent nanosecond CSV parsing and phase pairing, filename millisecond origin; no production parser imported. Rebuild phase-anchored 6s/1s windows; read BDF annotations and actual samples. Apply exactly loso.load argsort to cache t/y/blk. float32 time ULP tolerance; sample boundaries checked separately.', auc=None, f1=None, metric_note='No model scores or training in this audit; AUC/F1 not computed.', limitation='Confirms correspondence to supplied raw events, not truth of experimental event logging or score polarity. Sample rounding may include up to half a sample before nominal phase onset.', subjects=results)
    output = OUT / 'audit_results.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding='utf-8')
    for r in results:
        print(json.dumps({k:r[k] for k in ('subject','cached_windows','expected_windows','mismatches','event_issues','max_cache_time_rounding_error_s','duplicate_cache_times','reorder_count')}, ensure_ascii=False))
        print('trials', [(t['blk'],t['phase'],round(t['start'],7),round(t['duration'],7),t['cached_windows'],t['rejected_windows']) for t in r['trials']])
        print('annotations', r['annotations'], 'missing', len(r['missing_expected_windows']))
    print(output)


if __name__ == '__main__':
    main()
