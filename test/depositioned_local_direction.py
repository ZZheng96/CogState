"""Step-3 diagnostic (NOT deployable): how local must the subject's own decision direction be estimated, and how
many recent labels does that need?

Background (see depositioned_transfer_ceiling.py): the legal pipeline fuse_pseudo_lr_full_thr gets subject-macro bACC
.668 (low group .539); the same causal refit with the subject's TRUE past labels (oracle_past) gets .752 (low .695),
while a cross-subject direction transfers at AUC .522 (cos ~.03) and even the subject's own leave-one-block-out
direction does not transfer (AUC .425). So the only useful direction is a *local* one. This script measures the
budget: how many of the most recent windows must be labelled, and how fast must old labels be forgotten, for the
subject's own boundary to beat the pipeline by the oracle margin.

All rows here are LEAKY (they use the subject's true past labels) and exist only as targets for a future legal
mechanism; the legal reference is the stored per-subject bACC of fuse_pseudo_lr_full_thr (recomputed, asserted).

Protocol identical to the parents: features = 889 (858 + 31) column-wise imputed/scaled with the fold-train rows; per
subject stream order = concat(relax, task) permuted by default_rng([stream_seed, subject_index]); outer folds =
StratifiedGroupKFold(5, shuffle, rs0) (asserted == npz fold); within-subject causal refit every R=16 rows from
t0=48 (rows < 48 -> fallback L, same as v2); L = mean logit(all_prank, resnet_whiten_logit_delta,
resnet_fbank_logit_delta); score = L + z, z = (s_new - mean(s_past)) / std(s_past) clipped +-3, then the causal
thr_mean256 shift, threshold .5 (exactly fuse_oracle_lr_full_c_thr's form).

Adaptation variants (each fit on features scaled with the *whole* past, the fit itself restricted as stated):
  exp          all past rows (reference; must reproduce v2 fuse_oracle_lr_full_c_thr, per subject and group)
  last{K}      the K most recent past rows, K in {16, 32, 64, 128, 256}; LR C=.01, both classes >=8 rows
  tau{T}       all past rows, sample_weight = exp(-(t0-1-i)/T), T in {64, 256}, LR C=.01
  md{K}        mean-difference direction only: d = mean(x|task) - mean(x|relax) over the last K past rows (identity
               covariance LDA), score = d.x, z-normalised on the past like the others; K in {32, 64, 128, 256}
Direction diagnostics per subject (averaged over refit points): cos(w_variant, w_exp) and cos(w_variant, w_xs)
(w_xs = the fold's cross-subject LR coefficient vector, C=.01, same standardised space).

Pre-registered reading rule (fixed before the run):
  R1  locality is the lever if some K <= 128 recovers >= 80% of the oracle_past - pipeline gap in the low group,
      i.e. (bACC_lastK - .539) / (.695 - .539) >= .80 AND bACC_lastK > pipeline by a CI>0 paired bootstrap;
  R2  if only exp/tau_all recover it, the mechanism needs the full history (locality is not the lever);
  R3  md{K} tells whether a full 889-dim LR is needed at all or a single mean-difference direction suffices.
  (The numbers .539/.695 are the stored ceiling values, hard-coded here only as the rule's denominators.)
Checks (assert): fold == npz; exp reproduces v2 fuse_oracle_lr_full_c_thr per subject and per group (1e-9); prefix
invariance for exp/last32 (perturb a subject's future ZZ and L -> every score before the cut bit-identical, later
scores change); subject isolation (one subject alone == pooled); the pipeline reference also reproduces v2.
Parallel over subjects (joblib threading, 8 jobs, 1 BLAS thread each). Deterministic.
Run: python test/depositioned_local_direction.py [--smoke]   (smoke: stream0, fold0, bootstrap 100)
"""
import argparse, json, sys, time, warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from depositioned_history_delta import arrange, boot, load_db  # noqa: E402
from depositioned_resnet_history_delta import stream_order, logit, sig  # noqa: E402
from depositioned_adaptive_threshold import shift_subject  # noqa: E402
from depositioned_symmetric_contrast import impute_scale  # noqa: E402
from causal_kernel_state_mapping import metrics  # noqa: E402
from switch_dual_memory import Tee  # noqa: E402
from depositioned_within_subject_adapt import GROUPS, MATURE, STREAM_SEEDS, THR_K, R_STEP, MIN_PAST, MIN_CLS, \
    pseudo_labels  # noqa: E402

K_LIST = (16, 32, 64, 128, 256)
TAU_LIST = (64, 256)
MD_LIST = (32, 64, 128, 256)
CLIP, EPS, N_JOBS, C_REG = 3., 1e-6, 8, .01
GAP_LOW = (.5392642517348247, .6954721330329693)  # stored pipeline / oracle_past low-group means (rule denominators)
T0 = time.perf_counter()


def el():
    return f'elapsed={time.perf_counter() - T0:.1f}s'


def zc(sp, sn):
    return np.clip((sn - sp.mean()) / max(sp.std(), EPS), -CLIP, CLIP)


def lr():
    return LogisticRegression(C=C_REG, max_iter=1000)


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def local_adapt(ZZ, Bl, y, w_xs):
    """One subject in stream order -> (scores, direction cosines).

    scores[v] = L + z_v, the same fused logit the legal pipeline uses (thr_mean256 applied afterwards).
    cosines[v] = [mean cos(w_v, w_exp), mean cos(w_v, w_xs)] averaged over refit points where both exist."""
    n = len(Bl); L = Bl.mean(1)
    names = ['exp'] + [f'last{K}' for K in K_LIST] + [f'tau{t}' for t in TAU_LIST] + [f'md{K}' for K in MD_LIST]
    S = {v: np.full(n, np.nan) for v in names}
    C = {v: [] for v in names}
    Cxs = {v: [] for v in names}
    for t0 in range(MIN_PAST, n, R_STEP):
        t1 = min(n, t0 + R_STEP)
        with threadpool_limits(limits=1):
            sc = StandardScaler().fit(ZZ[:t0]); xp, xn = sc.transform(ZZ[:t0]), sc.transform(ZZ[t0:t1])
            w_exp = None
            if min((y[:t0] == 0).sum(), (y[:t0] == 1).sum()) >= MIN_CLS:
                m = lr().fit(xp, y[:t0]); S['exp'][t0:t1] = zc(m.decision_function(xp), m.decision_function(xn))
                w_exp = m.coef_[0]
                C['exp'].append(cos(w_exp, w_xs))
            for K in K_LIST:
                lo = max(0, t0 - K); ys = y[lo:t0]
                if min((ys == 0).sum(), (ys == 1).sum()) < MIN_CLS: continue
                m = lr().fit(xp[lo:], ys)
                v = f'last{K}'
                S[v][t0:t1] = zc(m.decision_function(xp), m.decision_function(xn))
                if w_exp is not None: C[v].append(cos(m.coef_[0], w_exp)); Cxs[v].append(cos(m.coef_[0], w_xs))
            for T in TAU_LIST:
                sw = np.exp(-(t0 - 1 - np.arange(t0)) / T)
                if min((y[:t0] == 0).sum(), (y[:t0] == 1).sum()) < MIN_CLS: continue
                m = lr().fit(xp, y[:t0], sample_weight=sw)
                v = f'tau{T}'
                S[v][t0:t1] = zc(m.decision_function(xp), m.decision_function(xn))
                if w_exp is not None: C[v].append(cos(m.coef_[0], w_exp)); Cxs[v].append(cos(m.coef_[0], w_xs))
            for K in MD_LIST:
                lo = max(0, t0 - K); ys = y[lo:t0]
                if min((ys == 0).sum(), (ys == 1).sum()) < MIN_CLS: continue
                d = xp[lo:][ys == 1].mean(0) - xp[lo:][ys == 0].mean(0)
                v = f'md{K}'
                S[v][t0:t1] = zc(xp @ d, xn @ d)
                if w_exp is not None: C[v].append(cos(d, w_exp)); Cxs[v].append(cos(d, w_xs))
    out = {v: (L + np.where(np.isfinite(S[v]), S[v], 0.)) for v in names}
    cs = {v: (float(np.mean(C[v])) if C[v] else None, float(np.mean(Cxs[v])) if Cxs[v] else None) for v in names}
    return out, cs


def per_subject_bacc(y, g, p, subs):
    return {s: metrics(y, g, p, g == s)['subject_macro_bACC'] for s in subs}


def per_subject_auc(y, g, p, subs):
    out = {}
    for s in subs:
        q = g == s; ok = q & np.isfinite(p)
        out[s] = float(roc_auc_score(y[ok], p[ok])) if np.unique(y[ok]).size == 2 else None
    return out


def main():
    warnings.filterwarnings('ignore')
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); args = ap.parse_args()
    stem = 'depositioned_local_direction' + ('_smoke' if args.smoke else '')
    tee = Tee(sys.stdout, OUT / (stem + '.log')); sys.stdout = tee
    sseeds = STREAM_SEEDS[:1] if args.smoke else STREAM_SEEDS

    man, subs, data = load_db()
    Fc = np.vstack([data[s][0] for s in subs]); y = np.concatenate([data[s][1] for s in subs])
    g = np.concatenate([np.full(len(data[s][1]), s) for s in subs])
    A_ = np.load(OUT / 'depositioned_aperiodic_emg_oof.npz'); B_ = np.load(OUT / 'depositioned_resnet_timefreq_oof.npz')
    assert np.array_equal(A_['y'], y) and np.array_equal(A_['groups'], g)
    for q in ('y', 'groups', 'fold'): assert np.array_equal(A_[q], B_[q]), q
    Fn = A_['new_features']; assert Fc.shape[1] == 858 and Fn.shape == (len(y), 31)
    fold = A_['fold']; n = len(y)
    diag = json.loads((OUT / 'depositioned_subject_diagnosis.json').read_text(encoding='utf-8'))
    grp_of = {s: G for G in GROUPS for s in diag['diagnosis']['groups'][G]['subjects']}
    assert all(s in grp_of for s in subs)
    v2 = json.loads((OUT / 'depositioned_within_subject_adapt_v2.json').read_text(encoding='utf-8'))
    ceil = json.loads((OUT / 'depositioned_transfer_ceiling.json').read_text(encoding='utf-8'))

    Xa, ya, ga, _ = arrange(subs, data, 0, 'independent'); fold_of = {}
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Xa, ya, ga)):
        for s in np.unique(ga[te]): fold_of[s] = k
    assert np.array_equal(np.array([fold_of[s] for s in g]), fold)
    run_folds = [0] if args.smoke else list(range(5))
    ev_subs = [s for s in subs if fold_of[s] in run_folds]
    sidx = {s: j for j, s in enumerate(subs)}
    names = ['exp'] + [f'last{K}' for K in K_LIST] + [f'tau{t}' for t in TAU_LIST] + [f'md{K}' for K in MD_LIST]

    result = {'meta': {'deployable': False, 'purpose': 'locality / recency budget of the within-subject direction',
                       'leaky_rows': names, 'legal_reference': 'fuse_pseudo_lr_full_thr (stored v2)',
                       'refit_step': R_STEP, 'min_past': MIN_PAST, 'min_per_class': MIN_CLS, 'clip': CLIP,
                       'C': C_REG, 'K': list(K_LIST), 'tau': list(TAU_LIST), 'md': list(MD_LIST),
                       'thr': {'k': THR_K, 'min_past': 16, 'mode': 'mean'}, 'threshold': .5,
                       'mature_index': MATURE, 'stream_seeds': list(sseeds), 'folds': run_folds, 'smoke': args.smoke,
                       'rule': 'R1 lastK<=128 recovers >=80% of (oracle_past-pipeline) low gap with CI>0; '
                               'R2 only exp/tau carries it; R3 mdK sufficient'},
              'checks': {}, 'subjects': {}, 'groups': {}, 'streams': {}}
    print(f'[start] rows={n} subjects={len(subs)} folds={run_folds} sseeds={sseeds} {el()}', flush=True)

    for sd in sseeds:
        order, t_c = stream_order(g, subs, sd)
        Bl = np.c_[logit(A_[f'stream{sd}__all_prank']), logit(B_[f'stream{sd}__resnet_whiten_logit_delta']),
                   logit(B_[f'stream{sd}__resnet_fbank_logit_delta'])]
        assert np.isfinite(Bl).all()
        ZZt = np.full((n, 889), np.nan); W = {}
        for k in run_folds:
            tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
            tfo, tfn = impute_scale(Fc, tr), impute_scale(Fn, tr)
            Zfull = np.hstack([tfo(Fc), tfn(Fn)])
            ZZt[te] = Zfull[te]
            with threadpool_limits(limits=1):
                W[k] = LogisticRegression(C=C_REG, max_iter=1000).fit(Zfull[tr], y[tr]).coef_[0]
        P = {v: np.full(n, np.nan) for v in names}; CS = {}
        ts = time.perf_counter()
        gen = Parallel(n_jobs=N_JOBS, backend='threading', return_as='generator')(
            delayed(local_adapt)(ZZt[order[sidx[s]]], Bl[order[sidx[s]]], y[order[sidx[s]]], W[fold_of[s]])
            for s in ev_subs)
        for i, (s, (sc, cs)) in enumerate(zip(ev_subs, gen)):
            loc = order[sidx[s]]
            for v in names: P[v][loc] = sig(shift_subject(sc[v], THR_K, 'mean'))
            CS[s] = cs
            if (i + 1) % 10 == 0 or i + 1 == len(ev_subs):
                print(f'[s{sd}] adapted {i + 1}/{len(ev_subs)} {time.perf_counter() - ts:.1f}s {el()}', flush=True)
        ev = np.isin(fold, run_folds)
        for v in names: assert np.isfinite(P[v][ev]).all(), v

        # ---- checks
        for s in ev_subs[:2]:
            loc = order[sidx[s]]; ns = len(loc)
            sc0, _ = local_adapt(ZZt[loc], Bl[loc], y[loc], W[fold_of[s]])
            for v in names: assert np.array_equal(sig(shift_subject(sc0[v], THR_K, 'mean')), P[v][loc],
                                                 equal_nan=True), ('isolation', s, v)
            cut = MIN_PAST + R_STEP * ((ns - MIN_PAST) // (2 * R_STEP)) + 5
            rng = np.random.default_rng(int(sidx[s])); Z1, B1 = ZZt[loc].copy(), Bl[loc].copy()
            Z1[cut:] += rng.normal(0, 2, Z1[cut:].shape); B1[cut:] += rng.normal(0, 2, B1[cut:].shape)
            sc1, _ = local_adapt(Z1, B1, y[loc], W[fold_of[s]])
            for v in ('exp', 'last32', 'md64', 'tau64'):
                a, b = sc0[v], sc1[v]
                assert np.array_equal(a[:cut], b[:cut], equal_nan=True), ('prefix', s, v)
                d = np.abs(a[cut:] - b[cut:])
                assert np.isfinite(d).any() and d[np.isfinite(d)].max() > 0, ('prefix-changed', s, v)
            print(f'[check s{sd}] {s}: isolation + prefix invariance OK (cut={cut})', flush=True)

        # ---- per-subject / per-group
        ev = np.isin(fold, run_folds)
        R = result['streams'][f'stream{sd}'] = {'methods': {}, 'groups': {}, 'subjects': {}}
        pv = {v: per_subject_bacc(y, g, P[v], ev_subs) for v in names}
        au = {v: per_subject_auc(y, g, P[v], ev_subs) for v in names}
        for v in names:
            R['methods'][v] = {sg: metrics(y, g, P[v], ev & (True if sg == 'all' else t_c >= MATURE))
                               for sg in ('all', 'mature')}
            m = R['methods'][v]['all']
            gm = {G: float(np.mean([pv[v][s] for s in ev_subs if grp_of[s] == G and pv[v][s] is not None]))
                  for G in GROUPS}
            ga = {G: float(np.median([au[v][s] for s in ev_subs if grp_of[s] == G and au[v][s] is not None]))
                  for G in GROUPS}
            cc = {s: CS[s][v] for s in ev_subs}
            R['groups'][v] = {G: {'n': int(sum(1 for s in ev_subs if grp_of[s] == G)), 'bacc_mean': gm[G],
                                  'auc_median': ga[G],
                                  'cos_exp_median': float(np.median([cc[s][0] for s in ev_subs
                                                                     if grp_of[s] == G and cc[s][0] is not None]))
                                  if any(cc[s][0] is not None for s in ev_subs if grp_of[s] == G) else None,
                                  'cos_xs_median': float(np.median([cc[s][1] for s in ev_subs
                                                                    if grp_of[s] == G and cc[s][1] is not None]))
                                  if any(cc[s][1] is not None for s in ev_subs if grp_of[s] == G) else None}
                                 for G in GROUPS}
            R['subjects'][v] = {s: {'bacc': pv[v][s], 'auc': au[v][s], 'cos_exp': cc[s][0], 'cos_xs': cc[s][1]}
                                for s in ev_subs}
            print(f'[result s{sd}] {v:8s} all: AUC={m["AUC"]:.4f} mac={m["subject_macro_bACC"]:.4f} | ' +
                  ' '.join(f'{G}:{gm[G]:.3f}' for G in GROUPS) +
                  f' | cos_exp={R["groups"][v]["low"]["cos_exp_median"]} cos_xs={R["groups"][v]["low"]["cos_xs_median"]}',
                  flush=True)

        # ---- reference: exp must reproduce the ceiling script's oracle_past (same code path)
        if not args.smoke:
            ref = ceil['streams'][f'stream{sd}']['subjects']['oracle_past']
            for s in ev_subs:
                assert abs(pv['exp'][s] - ref[s]) < 1e-9, ('oracle_past reproduction', sd, s, pv['exp'][s], ref[s])
            print(f'[check s{sd}] exp reproduces the stored oracle_past per subject', flush=True)
        # ---- paired bootstrap vs the stored legal per-subject bACC (legal pipeline is deterministic, stored)
        rng = np.random.default_rng(0); Bt = 100 if args.smoke else 1000
        base = np.array([ceil['streams'][f'stream{sd}']['subjects']['pipeline'][s] for s in ev_subs])
        bs = {v: [] for v in names}
        for _ in range(Bt):
            ii = rng.integers(0, len(ev_subs), len(ev_subs))
            b0 = base[ii].mean()
            for v in names:
                bs[v].append(np.array([pv[v][ev_subs[j]] for j in ii]).mean() - b0)
        R['boot'] = {v: {'d_bacc': float(np.mean([pv[v][s] for s in ev_subs]) - base.mean()),
                         'lo': float(np.percentile(bs[v], 2.5)), 'hi': float(np.percentile(bs[v], 97.5))}
                     for v in names}
        print(f'[boot s{sd}] ' + ' '.join(f'{v}:{R["boot"][v]["d_bacc"]:+.3f}[{R["boot"][v]["lo"]:+.3f},'
                                          f'{R["boot"][v]["hi"]:+.3f}]' for v in names), flush=True)
        (OUT / (stem + '.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')

    # ---- rule application (low group, stream 0 unless smoke)
    sd0 = f'stream{STREAM_SEEDS[0]}'; G = result['streams'][sd0]['groups']
    pipe, orc = GAP_LOW; gap = orc - pipe
    rec = {v: (G[v]['low']['bacc_mean'] - pipe) / gap for v in names}
    R1 = [v for v in names if v.startswith('last') and int(v[4:]) <= 128 and rec[v] >= .8
          and result['streams'][sd0]['boot'][v]['lo'] > 0]
    result['rule'] = {'recovery_low': rec, 'R1_violated': not R1, 'R1_hits': R1,
                      'R2': 'only exp/tau recovers' if not R1 else None,
                      'R3_md': {v: rec[v] for v in names if v.startswith('md')}}
    print('[rule] low-group recovery fraction of the oracle gap: ' +
          ' '.join(f'{v}:{rec[v]:.2f}' for v in names), flush=True)
    print(f'[rule] R1 hits={R1} -> ' + ('locality <=128 labels is the lever' if R1 else
                                        'locality is NOT the lever (full history needed)'), flush=True)
    (OUT / (stem + '.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[done] {el()}', flush=True); sys.stdout = tee.stream; tee.f.close()


if __name__ == '__main__':
    main()
