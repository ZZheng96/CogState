"""Three-way fusion of direction-H features with the filter-bank ResNet branch (no training, stored OOF scores only).

Sources (same protocol: stream order = stream_order(g, subs, seed), outer folds SGKF(5, rs0) asserted identical):
  all_prank, fuse_allprank_resnet, fusionM_prank      <- depositioned_aperiodic_emg_oof.npz
  resnet_whiten_logit_delta, resnet_fbank_logit_delta,
  fusion3_resnet_fbank                                <- depositioned_resnet_timefreq_oof.npz
Variants (fixed before running; sigmoid of mean logit, equal weights, threshold .5):
  fuseH_wf  = (all_prank, resnet_whiten_logit_delta, resnet_fbank_logit_delta)
  fuseH_f   = (all_prank, resnet_fbank_logit_delta)
References: fuse_allprank_resnet (asserted == recomputed from all_prank + whiten), fusion3_resnet_fbank, fusionM_prank.
Evaluation: all / mature (t>=120); AUC, subject-macro bACC, micro bACC; paired subject bootstrap 1000 vs
fuse_allprank_resnet and vs fusionM_prank; per group (depositioned_subject_diagnosis.json) macro bACC change vs
fuse_allprank_resnet; #subjects with bACC>=.9. No test labels used for any score.
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

MATURE = 120
STREAM_SEEDS = (0, 1, 2)
VARIANTS = {'fuseH_wf': ('all_prank', 'resnet_whiten_logit_delta', 'resnet_fbank_logit_delta'),
            'fuseH_f': ('all_prank', 'resnet_fbank_logit_delta')}
REFS = ('fuse_allprank_resnet', 'fusion3_resnet_fbank', 'fusionM_prank')


def main():
    stem = 'depositioned_fusion_h_fbank'
    tee = Tee(sys.stdout, OUT / (stem + '.log')); sys.stdout = tee
    man, subs, data = load_db()
    A = np.load(OUT / 'depositioned_aperiodic_emg_oof.npz'); B = np.load(OUT / 'depositioned_resnet_timefreq_oof.npz')
    y, g = A['y'], A['groups']
    for k in ('y', 'groups', 'fold'): assert np.array_equal(A[k], B[k]), k
    diag = json.loads((OUT / 'depositioned_subject_diagnosis.json').read_text(encoding='utf-8'))['diagnosis']['groups']
    grp_of = {s: G for G in ('high', 'mid', 'low') for s in diag[G]['subjects']}
    result = {'meta': {'variants': VARIANTS, 'refs': REFS, 'threshold': .5, 'mature_index': MATURE}}
    for sd in STREAM_SEEDS:
        _, t = stream_order(g, subs, sd)
        S = {n: A[f'stream{sd}__{n}'] for n in ('all_prank', 'fuse_allprank_resnet', 'fusionM_prank')}
        for n in ('resnet_whiten_logit_delta', 'resnet_fbank_logit_delta', 'fusion3_resnet_fbank'):
            S[n] = B[f'stream{sd}__{n}']
        ev = np.isfinite(S['all_prank']) & np.isfinite(S['resnet_fbank_logit_delta'])
        assert np.abs(sig((logit(S['all_prank']) + logit(S['resnet_whiten_logit_delta'])) / 2)[ev]
                      - S['fuse_allprank_resnet'][ev]).max() < 1e-9
        assert np.abs(A[f'stream{sd}__fusionM_prank'][ev] - B[f'stream{sd}__fusionM_prank'][ev]).max() < 1e-9
        for v, parts in VARIANTS.items():
            S[v] = sig(np.mean([logit(S[p]) for p in parts], 0))
        M = {'all': ev, 'mature': ev & (t >= MATURE)}
        R = result[f'stream{sd}'] = {'methods': {}, 'bootstrap': {}, 'groups': {}, 'subjects': {}}
        ev_subs = list(np.unique(g[ev])); PS = {}
        for n in list(VARIANTS) + list(REFS):
            p = S[n]; r = R['methods'][n] = {s: metrics(y, g, p, M[s]) for s in M}
            PS[n] = {s: metrics(y, g, p, ev & (g == s))['subject_macro_bACC'] for s in ev_subs}
            ba = np.array([q for q in PS[n].values() if q is not None])
            R['subjects'][n] = {'n_ge_0.9': int((ba >= .9).sum()), 'max': float(ba.max())}
            print(f'[result stream {sd}] {n:22s} ' + ' | '.join(
                f'{s}: AUC={r[s]["AUC"]:.4f} mac={r[s]["subject_macro_bACC"]:.4f} mic={r[s]["micro_bACC"]:.4f}' for s in M)
                + f' | subj>=.9={R["subjects"][n]["n_ge_0.9"]} max={R["subjects"][n]["max"]:.3f}', flush=True)
        for n in VARIANTS:
            R['groups'][n] = {}
            for G in ('high', 'mid', 'low'):
                ss = [s for s in ev_subs if grp_of.get(s) == G and PS[n][s] is not None]
                a = np.array([PS[n][s] for s in ss]); b = np.array([PS['fuse_allprank_resnet'][s] for s in ss])
                R['groups'][n][G] = {'n': len(ss), 'mean': float(a.mean()), 'd_mean_vs_fuse_allprank_resnet': float(a.mean() - b.mean()),
                                     'd_median_vs_fuse_allprank_resnet': float(np.median(a) - np.median(b))}
            print(f'[group stream {sd}] {n:10s} ' + ' | '.join(
                f'{G}(n={q["n"]}): mean={q["mean"]:.4f} dmean={q["d_mean_vs_fuse_allprank_resnet"]:+.4f} '
                f'dmed={q["d_median_vs_fuse_allprank_resnet"]:+.4f}' for G, q in R['groups'][n].items()), flush=True)
        for a_ in VARIANTS:
            for b_ in ('fuse_allprank_resnet', 'fusionM_prank'):
                for seg in M:
                    bb = R['bootstrap'][f'{a_}__vs__{b_}__{seg}'] = boot(S[a_], S[b_], y, g, M[seg], 1000)
                    print(f'[boot stream {sd}] {a_:9s} vs {b_:20s} {seg:6s} '
                          f'dAUC={bb["dAUC"]["mean"]:+.4f}[{bb["dAUC"]["p2.5"]:+.4f},{bb["dAUC"]["p97.5"]:+.4f}] '
                          f'dmac={bb["dmacro_bACC"]["mean"]:+.4f}[{bb["dmacro_bACC"]["p2.5"]:+.4f},{bb["dmacro_bACC"]["p97.5"]:+.4f}]',
                          flush=True)
    (OUT / (stem + '.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print('[done]', flush=True)
    sys.stdout = tee.stream; tee.f.close()


if __name__ == '__main__':
    main()
