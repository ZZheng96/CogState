"""Idea 3: per-subject causal adaptive threshold on stored fused scores (no training, no test labels).

Base score: fuseH_wf = sig(mean logit(all_prank, resnet_whiten_logit_delta, resnet_fbank_logit_delta)), recomputed from
depositioned_aperiodic_emg_oof.npz + depositioned_resnet_timefreq_oof.npz (same stream order / folds, asserted).
For each subject, in stream order, row t uses the scores of the same subject's rows <= t only (all already causal):
  thr_med{k}  : l_t - median(l_{t-k..t-1})                  (l = logit score; median of previous <=k rows)
  thr_mean{k} : l_t - mean(l_{t-k..t-1})
  thr_mix{k}  : l_t - .5*median(l_{t-k..t-1})               (half-shift, shrink towards global threshold)
  k in {64, 256, all}; if fewer than MIN_PAST=16 previous rows -> no shift (l_t). Output = sig(shifted logit), thr .5.
Rationale: label-free; class ratio within a subject's stream is roughly balanced, so the subject's running median score
is an unsupervised estimate of its decision boundary (removes per-subject offset / calibration bias).
All hyper-parameters fixed before running. Evaluation: all / mature (t>=120); AUC, subject-macro bACC, micro bACC;
#subjects bACC>=.9; paired subject bootstrap 1000 vs fuseH_wf; group (high/mid/low) macro bACC change vs fuseH_wf.
Checks: prefix invariance (future scores changed -> prefix outputs identical) and naive stepwise reference (assert).
"""
import json, sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from depositioned_history_delta import boot, load_db  # noqa: E402
from depositioned_resnet_history_delta import stream_order, logit, sig  # noqa: E402
from causal_kernel_state_mapping import metrics  # noqa: E402
from switch_dual_memory import Tee  # noqa: E402

MATURE, MIN_PAST, STREAM_SEEDS = 120, 16, (0, 1, 2)
KS = (64, 256, None)
MODES = ('med', 'mean', 'mix')


def shift_subject(l, k, mode):
    out = l.copy()
    for t in range(MIN_PAST, len(l)):
        P = l[max(0, t - k):t] if k else l[:t]
        if mode == 'med': out[t] = l[t] - np.median(P)
        elif mode == 'mean': out[t] = l[t] - P.mean()
        else: out[t] = l[t] - .5 * np.median(P)
    return out


def apply(l, order, k, mode):
    o = np.full(len(l), np.nan)
    for loc in order:
        loc = np.asarray(loc); v = np.isfinite(l[loc])
        if v.any(): o[loc[v]] = shift_subject(l[loc[v]], k, mode)
    return o


def main():
    stem = 'depositioned_adaptive_threshold'
    tee = Tee(sys.stdout, OUT / (stem + '.log')); sys.stdout = tee
    man, subs, data = load_db()
    A = np.load(OUT / 'depositioned_aperiodic_emg_oof.npz'); B = np.load(OUT / 'depositioned_resnet_timefreq_oof.npz')
    y, g = A['y'], A['groups']
    for q in ('y', 'groups', 'fold'): assert np.array_equal(A[q], B[q]), q
    diag = json.loads((OUT / 'depositioned_subject_diagnosis.json').read_text(encoding='utf-8'))['diagnosis']['groups']
    grp_of = {s: G for G in ('high', 'mid', 'low') for s in diag[G]['subjects']}
    names = [f'thr_{m}{k or "all"}' for m in MODES for k in KS]
    result = {'meta': {'min_past': MIN_PAST, 'ks': [k or 'all' for k in KS], 'modes': MODES, 'threshold': .5}}
    for sd in STREAM_SEEDS:
        order, t = stream_order(g, subs, sd)
        L = np.mean([logit(A[f'stream{sd}__all_prank']), logit(B[f'stream{sd}__resnet_whiten_logit_delta']),
                     logit(B[f'stream{sd}__resnet_fbank_logit_delta'])], 0)
        ev = np.isfinite(L)
        S = {'fuseH_wf': sig(L)}
        for m in MODES:
            for k in KS:
                n = f'thr_{m}{k or "all"}'; S[n] = sig(apply(L, order, k, m))
        # checks: naive + prefix invariance on 2 subjects
        for loc in order[:2]:
            loc = np.asarray(loc); loc = loc[np.isfinite(L[loc])]; l0 = L[loc]
            for m in MODES:
                for k in KS:
                    o = shift_subject(l0, k, m)
                    for tt in (MIN_PAST - 1, MIN_PAST, len(l0) // 2, len(l0) - 1):
                        P = l0[max(0, tt - k):tt] if k else l0[:tt]
                        ref = l0[tt] if tt < MIN_PAST else l0[tt] - {'med': np.median(P), 'mean': P.mean(), 'mix': .5 * np.median(P)}[m]
                        assert abs(o[tt] - ref) < 1e-12
                    l1 = l0.copy(); cut = len(l0) // 2; l1[cut:] += np.random.default_rng(0).normal(0, 3, len(l0) - cut)
                    o1 = shift_subject(l1, k, m)
                    assert np.array_equal(o[:cut], o1[:cut]) and np.abs(o[cut:] - o1[cut:]).max() > 0
        print(f'[check stream {sd}] naive + prefix invariance OK', flush=True)
        M = {'all': ev, 'mature': ev & (t >= MATURE)}
        R = result[f'stream{sd}'] = {'methods': {}, 'bootstrap': {}, 'groups': {}, 'subjects': {}}
        ev_subs = list(np.unique(g[ev])); PS = {}
        for n in ['fuseH_wf'] + names:
            p = S[n]; r = R['methods'][n] = {s: metrics(y, g, p, M[s]) for s in M}
            PS[n] = {s: metrics(y, g, p, ev & (g == s))['subject_macro_bACC'] for s in ev_subs}
            ba = np.array([q for q in PS[n].values() if q is not None])
            R['subjects'][n] = {'n_ge_0.9': int((ba >= .9).sum()), 'max': float(ba.max())}
            print(f'[result stream {sd}] {n:12s} ' + ' | '.join(
                f'{s}: AUC={r[s]["AUC"]:.4f} mac={r[s]["subject_macro_bACC"]:.4f} mic={r[s]["micro_bACC"]:.4f}' for s in M)
                + f' | subj>=.9={R["subjects"][n]["n_ge_0.9"]} max={R["subjects"][n]["max"]:.3f}', flush=True)
        for n in names:
            R['groups'][n] = {}
            for G in ('high', 'mid', 'low'):
                ss = [s for s in ev_subs if grp_of.get(s) == G and PS[n][s] is not None]
                a = np.array([PS[n][s] for s in ss]); b = np.array([PS['fuseH_wf'][s] for s in ss])
                R['groups'][n][G] = {'n': len(ss), 'mean': float(a.mean()), 'd_mean': float(a.mean() - b.mean()),
                                     'd_median': float(np.median(a) - np.median(b))}
            print(f'[group stream {sd}] {n:12s} ' + ' | '.join(
                f'{G}(n={q["n"]}): mean={q["mean"]:.4f} dmean={q["d_mean"]:+.4f} dmed={q["d_median"]:+.4f}'
                for G, q in R['groups'][n].items()), flush=True)
            for seg in M:
                bb = R['bootstrap'][f'{n}__vs__fuseH_wf__{seg}'] = boot(S[n], S['fuseH_wf'], y, g, M[seg], 1000)
                print(f'[boot stream {sd}] {n:12s} vs fuseH_wf {seg:6s} '
                      f'dAUC={bb["dAUC"]["mean"]:+.4f}[{bb["dAUC"]["p2.5"]:+.4f},{bb["dAUC"]["p97.5"]:+.4f}] '
                      f'dmac={bb["dmacro_bACC"]["mean"]:+.4f}[{bb["dmacro_bACC"]["p2.5"]:+.4f},{bb["dmacro_bACC"]["p97.5"]:+.4f}]',
                      flush=True)
    (OUT / (stem + '.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print('[done]', flush=True)
    sys.stdout = tee.stream; tee.f.close()


if __name__ == '__main__':
    main()
