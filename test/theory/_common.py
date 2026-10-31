"""Shared helpers for test/theory/* diagnostics.

load(): audit loading (889 = 858 + 31 features, strict SGKF folds asserted == npz, high/mid/low groups) + real recording
time / block (trial id 0..5, alternating relax/task 5-min blocks) from audit/literature_protocol/real_time.npz.
"""
import json, re, sys, time, warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEST = HERE.parent
sys.path.insert(0, str(TEST / 'audit'))
from common import GROUPS, STREAM_SEEDS, boot, impute_scale, load_all, metrics, Tee  # noqa: E402,F401

T0 = time.perf_counter()


def el():
    return f'elapsed={time.perf_counter() - T0:.1f}s'


def open_log(stem):
    warnings.filterwarnings('ignore')
    tee = Tee(sys.stdout, HERE / (stem + '.log')); sys.stdout = tee
    return tee


def save(stem, obj, tee):
    obj['seconds'] = time.perf_counter() - T0
    (HERE / (stem + '_results.json')).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float),
                                                 encoding='utf-8')
    print(f'[done] {el()} -> {stem}_results.json', flush=True)
    sys.stdout = tee.stream; tee.f.close()


def load():
    D = load_all()
    RT = np.load(TEST / 'audit' / 'literature_protocol' / 'real_time.npz')
    assert np.array_equal(RT['y'], D['y']) and np.array_equal(RT['groups'], D['g'])
    D['t'] = RT['t_end']; D['block'] = RT['trial'].astype(int)
    for s in D['subs']:  # every block is single-label and labels alternate
        q = D['g'] == s
        for b in np.unique(D['block'][q]):
            assert np.unique(D['y'][q & (D['block'] == b)]).size == 1
    D['F'] = np.hstack([D['Fc'], D['Fn']])
    man = json.loads((TEST.parent / 'database' / 'manifest.json').read_text(encoding='utf-8'))
    ap = json.loads((TEST / 'depositioned_aperiodic_emg.json').read_text(encoding='utf-8'))
    D['names'] = list(man['feature_names']) + list(ap['meta']['new_feature_names'])
    assert len(D['names']) == 889
    print(f'[load] rows={len(D["y"])} subjects={len(D["subs"])} real time/block loaded {el()}', flush=True)
    return D


# ---------------------------------------------------------------- feature families
# BANDS index (cogstate.BANDS): 0 delta 1-4, 1 theta 4-8, 2 alpha 8-13, 3 beta 13-30, 4 gamma 30-45 Hz.
# 858 features come from the causally 1-45 Hz band-passed (+50 Hz notch) signal except quality metrics and raw_* nonlinear.
# DWT (fs 250, level 5): A5 0-3.9, D5 3.9-7.8, D4 7.8-15.6, D3 15.6-31, D2 31-62 (filtered -> 31-45), D1 62-125
# (filtered -> stop-band residue). wavedec order index 0..5 = A5, D5, D4, D3, D2, D1.
BAND_FAM = {0: 'delta_ocular', 1: 'theta', 2: 'alpha', 3: 'beta', 4: 'emg_hf'}
DWT_FAM = {'A5': 'delta_ocular', 'D5': 'theta', 'D4': 'alpha', 'D3': 'beta', 'D2': 'emg_hf', 'D1': 'emg_hf'}
DWT_IDX = ['A5', 'D5', 'D4', 'D3', 'D2', 'D1']
FAMILIES = ('emg_hf', 'delta_ocular', 'theta', 'alpha', 'beta', 'aperiodic', 'spectral_shape', 'time_stats', 'other')
ARTIFACT = ('emg_hf', 'delta_ocular')


def family(n):
    m = re.match(r'stft_[\d.]+_(log|relative)_(\d)_', n) or re.match(r'hilbert_(envelope|frequency)_(\d)_', n)
    if m:
        return BAND_FAM[int(m.group(2))]
    m = re.match(r'stft_(log_power|relative)_(\d+)_(\d+)_', n)
    if m:
        return {1: 'delta_ocular', 4: 'theta', 8: 'alpha', 13: 'beta', 30: 'emg_hf'}[int(m.group(2))]
    m = re.match(r'dwt_(db4|sym5|coif2)_(\d)_', n)
    if m:
        return DWT_FAM[DWT_IDX[int(m.group(2))]]
    m = re.match(r'dwt_(log_energy|relative|entropy)_(A5|D\d)$', n)
    if m:
        return DWT_FAM[m.group(2)]
    if n.startswith('dwt_log_ratio_'):
        return 'spectral_shape'
    if n in ('amplitude_max', 'peak_to_peak', 'low_freq_drift', 'spike_fraction'):
        return 'delta_ocular'  # blink / slow eye-movement sensitive (raw-signal amplitude & 1-4 Hz share)
    if n in ('high_freq_ratio', 'diff_energy', 'raw_mobility', 'raw_complexity', 'raw_zero_crossing', 'raw_ac1'):
        return 'emg_hf'  # raw (unfiltered) high-frequency content: 30-45 share, derivative energy, Hjorth on raw
    if n.startswith('hf_'):
        return 'emg_hf'
    if n.startswith(('ap_', 'kn_')):
        return 'aperiodic'
    m = re.match(r'res_(theta|alpha|beta1|beta2|gamlow)_', n)
    if m:
        return {'theta': 'theta', 'alpha': 'alpha', 'beta1': 'beta', 'beta2': 'beta', 'gamlow': 'emg_hf'}[m.group(1)]
    if n in ('filtered_mobility', 'filtered_complexity'):
        return 'spectral_shape'
    if re.match(r'(filtered|raw)_', n) or n in ('rms', 'flat_fraction', 'nan_fraction'):
        return 'time_stats'
    return 'other'


def families(names):
    fam = np.array([family(n) for n in names])
    return fam, {f: int((fam == f).sum()) for f in FAMILIES}


def subj_impute_scale(F, tr):
    """median-impute + StandardScaler fitted on rows tr (of F); zero-variance columns -> 0."""
    A = np.where(np.isfinite(F), F, np.nan)
    med = np.nanmedian(A[tr], 0); med = np.where(np.isfinite(med), med, 0.)
    A = np.where(np.isfinite(A), A, med)
    mu, sd = A[tr].mean(0), A[tr].std(0)
    return (A - mu) / np.where(sd > 1e-12, sd, 1.)


def bacc(y, p, thr=.5):
    yh = p >= thr
    return .5 * ((yh[y == 1]).mean() + (~yh[y == 0]).mean())


def gsummary(per, grp_of, keys):
    """per: {subject: {key: value}} -> {group: {key: {mean, median, n}}} for all + GROUPS."""
    out = {}
    for G in ('all',) + tuple(GROUPS):
        ss = [s for s in per if G == 'all' or grp_of[s] == G]
        if not ss: continue
        out[G] = {}
        for k in keys:
            v = np.array([per[s][k] for s in ss if per[s].get(k) is not None and np.isfinite(per[s][k])], float)
            out[G][k] = {'mean': float(v.mean()) if len(v) else None, 'median': float(np.median(v)) if len(v) else None,
                         'n': int(len(v))}
    return out
