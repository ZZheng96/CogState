"""Step-2: subject-adversarial / cross-subject-mixup ResNet branch under the strict causal de-positioned protocol.

Question: can the cross-subject representation itself be made better (so that the low group becomes separable)
instead of adding more within-subject adaptation? The backbone is the repo's ResNet1d on the current 6 s raw window
(raw_waveform_window6.ResNet1d: per-window demean/std/clip +-10, 5-crop TTA, AdamW lr1e-3 wd1e-2, BS 128, max 50
epochs, patience 8 on the held-out-subject val AUC, `augment` = random 4 s crop + amplitude scale U(log.8,log1.25) +
Gaussian noise sd .05), i.e. the same recipe as depositioned_resnet_history_delta.py; early stopping uses the B split
of A/B (first split of StratifiedGroupKFold(4, shuffle, rs=fold+1) inside the fold-train subjects).

Protocol (identical to the parents, all asserted):
  stream order per subject = concat(relax,task) permuted by default_rng([stream_seed, subject_index]); outer folds =
  StratifiedGroupKFold(5, shuffle, rs0) on the seed-0 arrangement (asserted == stored fold). The ResNet sees only the
  current window, so it is trained once per (fold, net seed) and reused for all 3 stream seeds; only the history heads
  and the within-subject adaptation are stream-dependent. Test labels are never used.
Hyper-parameters (all fixed BEFORE running, nothing tuned or selected on the test set):
  NET_SEEDS = (0,1,2)  (same as the parent; score = probability mean over seeds)
  ADV_LAMBDA = 1.0, DANN ramp lambda_eff = ADV_LAMBDA * min(1, epoch/ADV_RAMP), ADV_RAMP = 10 epochs; subject
  classifier = Linear(128, n_subjects_in_A) on the mean-pooled 128-d embedding, gradient reversal between the
  backbone and that classifier, loss = BCE(task) + lambda_eff * CE(subject).
  MIX_ALPHA = 0.4, one mix per training sample (probability 1): partner drawn uniformly among the fold-train windows
  of the SAME LABEL from a DIFFERENT subject (rejection sampling, 10 tries, else no mixing), lambda ~ Beta(.4,.4) per
  sample, x = lambda*x_i + (1-lambda)*x_partner on the normalised 6 s windows BEFORE augmentation; the label is
  unchanged (same-class mix). Mixing only inside the fold-train set (A subjects); no mixing at test time.
Variants (4):
  plain       control, exactly the parent recipe (no subject head, no mixing) -> reproduces the stored
              depositioned_resnet_history_delta `stream1__resnet_current` (seed 0, 1e-4) as a harness check.
  mixup       plain + same-label cross-subject mixup.
  adv         plain + subject-adversarial head (GRL).
  adv_mixup   both.
Downstream (identical for every variant, copied from depositioned_resnet_history_delta / the v2 adaptation):
  l = TTA-mean head logit per window; ls = (l - mean(l[B]))/std(l[B]);
  resnet_<v>_current    = sig(mean over seeds of l)
  resnet_<v>_logit_delta= mean over seeds of LR(C=.05) on B fitted with [ls, ls - mean(past 32 ls)]
  HN_<v>   = sig(mean(logit(all_prank), logit(resnet_whiten_logit_delta), logit(resnet_<v>_logit_delta)))
             (replaces the fbank branch of fuseH_wf)
  F4_<v>   = sig(mean of the four branches: all_prank, whiten_delta, fbank_delta, <v>_logit_delta)
  P_<v>    = within-subject pseudo-label adaptation exactly as fuse_pseudo_lr_full: L_v = mean of the three logits of
             HN_<v>, f1 = L_v + z (pseudo labels from the past L_v top/bottom 30%, LR C=.01 on the past-scaled 889 zz,
             refit every R=16 windows from t0=48, z clipped +-3)
  each of HN/F4/P is reported at the fixed 0.5 threshold and with thr_mean256 (depositioned_adaptive_threshold).
Evaluation (same as depositioned_final_merge.py): all / mature (t>=120) AUC, subject-macro bACC, micro bACC, number of
subjects with bACC>=.9 and the max; paired subject bootstrap (1000; 100 in smoke) of every variant score against
fuse_pseudo_lr_full_thr (best legal pipeline), against fuseH_wf_thr, and against the matched `plain` score (isolates
the effect of the adversarial/mixup term); high/mid/low groups from depositioned_subject_diagnosis.json (report only).
Extra diagnostic: subject-identification linear probe on the frozen TTA-mean embedding of the fold-train subjects
(StandardScaler + LR C=1, 5-fold window CV, fold 0, seed 0) - how much subject identity is still linearly decodable
(plain vs adv), i.e. whether the GRL actually removed subject information.
Checks (assert): fold == npz, A/B/te subject-disjoint; plain reproduces the stored resnet_current; the recomputed
fuseH_wf and fuse_pseudo_lr_full_thr reproduce the stored/v2 numbers (1e-9); embedding locality + subject isolation
(one subject alone == pooled, future raw perturbation leaves earlier logits identical); score-level prefix invariance
for resnet_<v>_logit_delta and P_<v> (perturb a subject's future branch logits and zz, every score before the cut
bit-identical, later scores change); shuffling the test subjects' labels leaves every legal score bit-identical.
Parallel: single GPU, one network at a time; BLAS threads limited to 4. Progress printed per epoch / per run.
Run: python test/depositioned_subject_adversarial.py [--smoke]
     smoke = stream0, fold0, net seed 0 only, 2 epochs, bootstrap 100
"""
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_v] = '4'
import argparse, json, sys, time, warnings
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from raw_waveform_window6 import ResNet1d, augment, normalize, WIN, CROP, N_TTA, BS, MAX_EP, PATIENCE, LR, WD  # noqa: E402
from depositioned_history_delta import arrange, make_clf, hist_subject, boot, load_db  # noqa: E402
from depositioned_resnet_history_delta import load_raw, stream_order, hist_delta, embed, logit, sig  # noqa: E402
from depositioned_final_merge import p1_subject  # noqa: E402
from depositioned_adaptive_threshold import apply as thr_apply  # noqa: E402
from depositioned_symmetric_contrast import impute_scale  # noqa: E402
from causal_kernel_state_mapping import metrics  # noqa: E402
from switch_dual_memory import Tee  # noqa: E402
from depositioned_within_subject_adapt import GROUPS, MATURE, STREAM_SEEDS, THR_K, R_STEP, MIN_PAST, MIN_CLS  # noqa: E402

VARIANTS = ('plain', 'mixup', 'adv', 'adv_mixup')
CFG = {'plain': {'mixup': False, 'adv': False}, 'mixup': {'mixup': True, 'adv': False},
       'adv': {'mixup': False, 'adv': True}, 'adv_mixup': {'mixup': True, 'adv': True}}
NET_SEEDS = (0, 1, 2)
ADV_LAMBDA, ADV_RAMP, MIX_ALPHA, MIX_TRIES = 1., 10, .4, 10
T0 = time.perf_counter()


def el():
    return f'elapsed={time.perf_counter() - T0:.1f}s'


class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def mixup_pair(ids, y, g, pool0, pool1, rng):
    """Same-label, different-subject partner + Beta(.4,.4) coefficient per sample."""
    n = len(ids); part = np.full(n, -1)
    for _ in range(MIX_TRIES):
        need = np.flatnonzero(part < 0)
        if need.size == 0: break
        cls = y[ids[need]]
        cand = np.where(cls == 0, pool0[rng.integers(0, len(pool0), need.size)],
                        pool1[rng.integers(0, len(pool1), need.size)])
        ok = g[cand] != g[ids[need]]
        part[need[ok]] = cand[ok]
    part[part < 0] = ids[part < 0]
    return part, rng.beta(MIX_ALPHA, MIX_ALPHA, n)


def train_net(Xg, yg, subj_t, y, g, tr, va, seed, tag, max_ep, cfg, n_cls):
    tick = time.perf_counter(); dev = Xg.device
    torch.manual_seed(seed); rng = np.random.default_rng(seed); prng = np.random.default_rng(seed + 7777)
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    net = ResNet1d().to(dev)
    sch = nn.Linear(128, n_cls).to(dev) if cfg['adv'] else None
    params = list(net.parameters()) + (list(sch.parameters()) if sch is not None else [])
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    crit, crit2 = nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss()
    pool0, pool1 = tr[y[tr] == 0], tr[y[tr] == 1]
    best, state, stale, best_ep = -1., None, 0, 0
    for ep in range(1, max_ep + 1):
        net.train(); perm = rng.permutation(tr); tot = tot2 = 0.
        lam = ADV_LAMBDA * min(1., ep / ADV_RAMP) if cfg['adv'] else 0.
        for b in range(0, len(perm), BS):
            ids = perm[b:b + BS]
            if len(ids) < 2: continue
            xb = Xg[torch.as_tensor(ids, device=dev)]
            if cfg['mixup']:
                part, co = mixup_pair(ids, y, g, pool0, pool1, prng)
                xb = torch.as_tensor(co, dtype=torch.float32, device=dev)[:, None] * xb + \
                     torch.as_tensor(1. - co, dtype=torch.float32, device=dev)[:, None] * \
                     Xg[torch.as_tensor(part, device=dev)]
            xb = augment(xb, gen)
            e = net.blocks(net.stem(xb)); lg = net.head(e).ravel()
            idt = torch.as_tensor(ids, device=dev)
            loss = crit(lg, yg[idt])
            if cfg['adv']:
                loss2 = crit2(sch(_GRL.apply(e.mean(-1), lam)), subj_t[idt])
                loss = loss + loss2; tot2 += loss2.item() * len(ids)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, 1.); opt.step()
            tot += loss.item() * len(ids)
        _, lv = embed(net, Xg, va); auc = roc_auc_score(y[va], lv)
        if auc > best + 1e-4:
            best, state, stale, best_ep = auc, {k: v.detach().clone() for k, v in net.state_dict().items()}, 0, ep
        else:
            stale += 1
        extra = f' subjCE={tot2 / len(tr):.4f} lam={lam:.2f}' if cfg['adv'] else ''
        print(f'    [{tag}] ep{ep:02d}/{max_ep} loss={tot / len(tr):.4f}{extra} valAUC={auc:.4f} '
              f'best={best:.4f}@{best_ep} t={time.perf_counter() - tick:.0f}s {el()}', flush=True)
        if stale >= PATIENCE: break
    net.load_state_dict(state); net.eval()
    return net, {'best_epoch': int(best_ep), 'epochs_run': int(ep), 'best_val_auc': float(best),
                 'seconds': float(time.perf_counter() - tick)}


def subj_probe(E, g, tr, seed=0):
    """How much subject identity is linearly decodable from the frozen embedding (5-fold window CV, train subjects)."""
    subs = np.unique(g[tr]); m = {s: i for i, s in enumerate(subs)}
    ids = tr[np.isin(g[tr], subs)]
    X = E[ids]; yy = np.array([m[s] for s in g[ids]])
    acc = []
    with threadpool_limits(limits=1):
        for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, yy):
            sc = StandardScaler().fit(X[a])
            clf = LogisticRegression(C=1., max_iter=2000).fit(sc.transform(X[a]), yy[a])
            acc.append(float((clf.predict(sc.transform(X[b])) == yy[b]).mean()))
    return float(np.mean(acc)), int(len(subs))


def main():
    warnings.filterwarnings('ignore', category=FutureWarning)
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); args = ap.parse_args()
    stem = 'depositioned_subject_adversarial' + ('_smoke' if args.smoke else '')
    tee = Tee(sys.stdout, OUT / (stem + '.log')); sys.stdout = tee
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    rseeds = NET_SEEDS[:1] if args.smoke else NET_SEEDS
    sseeds = STREAM_SEEDS[:1] if args.smoke else STREAM_SEEDS
    max_ep = 2 if args.smoke else MAX_EP
    nb = 100 if args.smoke else 1000

    man, subs, data = load_db()
    Xraw, Fc, y, g = load_raw(subs)
    A_ = np.load(OUT / 'depositioned_aperiodic_emg_oof.npz'); B_ = np.load(OUT / 'depositioned_resnet_timefreq_oof.npz')
    assert np.array_equal(A_['y'], y) and np.array_equal(A_['groups'], g)
    for q in ('y', 'groups', 'fold'): assert np.array_equal(A_[q], B_[q]), q
    fold = A_['fold']; n = len(y)
    diag = json.loads((OUT / 'depositioned_subject_diagnosis.json').read_text(encoding='utf-8'))
    grp_of = {s: G for G in GROUPS for s in diag['diagnosis']['groups'][G]['subjects']}
    assert all(s in grp_of for s in subs)
    print(f'[start] rows={n} subjects={len(subs)} device={dev} smoke={args.smoke} variants={VARIANTS} '
          f'net_seeds={rseeds} stream_seeds={sseeds} adv_lambda={ADV_LAMBDA} ramp={ADV_RAMP} mix_alpha={MIX_ALPHA}',
          flush=True)

    Xa, ya, ga, _ = arrange(subs, data, 0, 'independent'); fold_of = {}
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Xa, ya, ga)):
        for s in np.unique(ga[te]): fold_of[s] = k
    assert np.array_equal(np.array([fold_of[s] for s in g]), fold)
    folds = []
    for k in range(5):
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        a, b = next(StratifiedGroupKFold(4, shuffle=True, random_state=k + 1).split(tr[:, None], y[tr], g[tr]))
        A, B = tr[a], tr[b]
        assert not (set(g[A]) & set(g[B])) and not (set(g[tr]) & set(g[te]))
        folds.append((tr, te, A, B))
    run_folds = [0] if args.smoke else list(range(5))
    print(f'[check] fold == depositioned_aperiodic_emg_oof.npz fold; folds={run_folds}', flush=True)

    Xn, n_flat = normalize(Xraw)
    Xg = torch.from_numpy(Xn).to(dev); yg = torch.tensor(y, dtype=torch.float32, device=dev)
    print(f'[data] normalised raw {tuple(Xn.shape)} flat_windows={n_flat}', flush=True)

    result = {'meta': {'rows': int(n), 'subjects': len(subs), 'variants': VARIANTS, 'adv_lambda': ADV_LAMBDA,
                       'adv_ramp_epochs': ADV_RAMP, 'mix_alpha': MIX_ALPHA, 'mix_prob': 1., 'net_seeds': list(rseeds),
                       'stream_seeds': list(sseeds), 'max_epochs': max_ep, 'patience': PATIENCE, 'bs': BS, 'lr': LR,
                       'wd': WD, 'mature_index': MATURE, 'thr': {'k': THR_K, 'min_past': 16, 'mode': 'mean'},
                       'threshold': .5, 'refit_step': R_STEP, 'min_past': MIN_PAST, 'bootstrap_n': nb,
                       'device': dev, 'smoke': args.smoke,
                       'gpu': torch.cuda.get_device_name(0) if dev == 'cuda' else None},
              'runs': [], 'probe': {}, 'checks': {}, 'streams': {}}
    checks = result['checks']

    # ---------------- networks: per fold, all variants (order-free) ----------------
    CUR = {v: np.full(n, np.nan) for v in VARIANTS}          # sig(mean logit over seeds)
    LOG = {}                                                 # (fold, variant, seed) -> per-window logit
    CACHE, CACHEJ = OUT / '_adv_logits_cache.npz', OUT / '_adv_cache_meta.json'
    refcur = None if args.smoke else np.load(
        OUT / 'depositioned_resnet_history_delta_oof.npz')['stream1__resnet_current']
    if CACHE.exists() and CACHEJ.exists() and not args.smoke:   # resume after an interrupted run
        z = np.load(CACHE)
        for key in z.files:
            k_, v_, rs_ = key.split('|'); LOG[(int(k_), v_, int(rs_))] = z[key]
        meta = json.loads(CACHEJ.read_text(encoding='utf-8'))
        result['runs'], result['probe'] = meta['runs'], meta['probe']; checks.update(meta['checks'])
        print(f'[resume] {len(LOG)} cached trainings loaded', flush=True)

    def save_cache():
        np.savez_compressed(CACHE, **{f'{a}|{b}|{c}': q for (a, b, c), q in LOG.items()})
        CACHEJ.write_text(json.dumps({'runs': result['runs'], 'probe': result['probe'], 'checks': checks},
                                     indent=2, ensure_ascii=False), encoding='utf-8')

    for k in run_folds:
        tr, te, A, B = folds[k]
        ssub = {s: i for i, s in enumerate(np.unique(g[A]))}
        subj_t = torch.tensor([ssub.get(s, -1) for s in g], dtype=torch.long, device=dev)
        for v in VARIANTS:
            cf = CFG[v]; lgs = []
            for rs in rseeds:
                if (k, v, rs) in LOG:
                    lgs.append(LOG[(k, v, rs)][te])
                    print(f'[resume] fold{k} {v} rs{rs} from cache', flush=True)
                    continue
                seed = 1000 * rs + k + 1
                tag = f'{v} rs{rs} fold{k + 1}/5'
                net, info = train_net(Xg, yg, subj_t, y, g, A, B, seed, tag, max_ep, cf, len(ssub))
                E, L = embed(net, Xg, np.arange(n))
                LOG[(k, v, rs)] = L.astype(np.float64); lgs.append(L[te])
                result['runs'].append({'variant': v, 'fold': k, 'net_seed': rs, **info,
                                       'test_auc': float(roc_auc_score(y[te], L[te])),
                                       'B_auc': float(roc_auc_score(y[B], L[B]))})
                if rs == 0 and k == run_folds[0]:
                    # embedding locality + subject isolation on one test subject
                    s0 = np.unique(g[te])[0]; loc = np.flatnonzero(g == s0)
                    E0, L0 = embed(net, Xg, loc)
                    iso = float(max(np.abs(E0 - E[loc]).max(), np.abs(L0 - L[loc]).max()))
                    Xs = Xg[torch.as_tensor(loc, device=dev)].clone(); cut = len(loc) // 2
                    Xs[cut:] = Xs[cut:] * -3. + 1.
                    E1, L1 = embed(net, Xg, loc, X_override=Xs)
                    dpre = float(max(np.abs(E1[:cut] - E0[:cut]).max(), np.abs(L1[:cut] - L0[:cut]).max()))
                    dfut = float(np.abs(L1[cut:] - L0[cut:]).max())
                    assert iso < 1e-4 and dpre < 1e-4 and dfut > 1e-3, (v, iso, dpre, dfut)
                    checks[f'{v}_locality_fold{k}'] = {'subject': str(s0), 'isolation': iso, 'prefix': dpre,
                                                       'future_changed': dfut}
                    print(f'[check] {v} fold{k}: embedding isolation={iso:.1e} prefix={dpre:.1e} '
                          f'future_changed={dfut:.2e}', flush=True)
                    pa, nc = subj_probe(E, g, tr, seed=0)
                    result['probe'][v] = {'accuracy': pa, 'n_classes': nc, 'chance': 1. / nc, 'fold': k,
                                          'net_seed': rs}
                    print(f'[probe] {v}: subject-id accuracy={pa:.4f} (chance={1. / nc:.4f}, {nc} classes)', flush=True)
                del net, E, L
                if dev == 'cuda': torch.cuda.empty_cache()
                save_cache()
            CUR[v][te] = sig(np.mean(lgs, 0))
            if v == 'plain' and not args.smoke:
                occ = np.flatnonzero(fold == k)
                d = float(np.abs(CUR['plain'][occ] - refcur[occ]).max())
                checks[f'plain_reproduced_fold{k}'] = d
                assert d < 1e-4, ('plain does not reproduce stored resnet_current', k, d)
                print(f'[check] plain fold{k} reproduces stored stream1__resnet_current (max|d|={d:.2e})', flush=True)
            print(f'[progress] fold {k + 1}/5 {v} done ({len(rseeds)} seeds) {el()}', flush=True)
        (OUT / (stem + '.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')

    # ---------------- per-stream scores ----------------
    v2 = json.loads((OUT / 'depositioned_within_subject_adapt_v2.json').read_text(encoding='utf-8'))
    sidx = {s: j for j, s in enumerate(subs)}
    run_subs = [s for s in subs if fold_of[s] in run_folds]

    def adapt_stream(ZZ, Bl, order):
        """p1_subject (pseudo-label within-subject adaptation of the v2 pipeline) for every evaluated subject."""
        out = np.full(len(y), np.nan)
        gen = Parallel(n_jobs=8, backend='threading', return_as='generator')(
            delayed(p1_subject)(ZZ[order[sidx[s]]], Bl[order[sidx[s]]], y[order[sidx[s]]]) for s in run_subs)
        for s, r in zip(run_subs, gen):
            out[order[sidx[s]]] = r[1]
        return out

    for sd in sseeds:
        order, t_c = stream_order(g, subs, sd)
        ev = np.isin(fold, run_folds)
        print(f'[stream {sd}] assembling scores {el()}', flush=True)
        # test-side 889 zz (identical construction to v2 / final_merge)
        ZZt = np.full((n, 889), np.nan)
        for k in run_folds:
            trk, tek = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
            tfo, tfn = impute_scale(Fc, trk), impute_scale(A_['new_features'], trk)
            ZZt[tek] = np.hstack([tfo(Fc[tek]), tfn(A_['new_features'][tek])])
        ap_ = logit(A_[f'stream{sd}__all_prank']); whd = logit(B_[f'stream{sd}__resnet_whiten_logit_delta'])
        fbd = logit(B_[f'stream{sd}__resnet_fbank_logit_delta'])
        L3 = (ap_ + whd + fbd) / 3
        S = {'fuseH_wf': sig(np.where(ev, L3, np.nan)),
             'fuseH_wf_thr': sig(thr_apply(np.where(ev, L3, np.nan), order, THR_K, 'mean'))}
        if not args.smoke:
            ref = v2[f'stream{sd}']['methods']['fuseH_wf']['all']['subject_macro_bACC']
            got = metrics(y, g, S['fuseH_wf'], ev)['subject_macro_bACC']; assert abs(got - ref) < 1e-9, (sd, got, ref)
            checks[f'fuseH_wf_reproduced_s{sd}'] = ref
        f1p = adapt_stream(ZZt, np.c_[ap_, whd, fbd], order)
        S['fuse_pseudo_lr_full'] = sig(np.where(ev, f1p, np.nan))
        S['fuse_pseudo_lr_full_thr'] = sig(thr_apply(np.where(ev, f1p, np.nan), order, THR_K, 'mean'))
        if not args.smoke:
            for nm in ('fuse_pseudo_lr_full', 'fuse_pseudo_lr_full_thr'):
                refm = v2[f'stream{sd}']['methods'][nm]['all']['subject_macro_bACC']
                gotm = metrics(y, g, S[nm], ev)['subject_macro_bACC']; assert abs(gotm - refm) < 1e-9, (nm, gotm, refm)
                checks[f'{nm}_reproduced_s{sd}'] = refm
        print(f'[check s{sd}] fuseH_wf + fuse_pseudo_lr_full(_thr) reproduce the parents {el()}', flush=True)

        # delta heads (stream-order history of the per-window logit, fitted on B) + the four score families
        DLT = {v: np.full(n, np.nan) for v in VARIANTS}
        F1 = {}
        for k in run_folds:
            tr, te, A, B = folds[k]
            for v in VARIANTS:
                pds = []
                for rs in rseeds:
                    L = LOG[(k, v, rs)]
                    ls = (L - L[B].mean()) / max(L[B].std(), 1e-8)
                    F2 = np.c_[ls, hist_delta(ls[:, None], order)]
                    with threadpool_limits(limits=4):
                        head = make_clf('lr').fit(F2[B], y[B])
                    pds.append(head.predict_proba(F2[te])[:, 1])
                DLT[v][te] = np.mean(pds, 0)
        for v in VARIANTS:
            ld = logit(DLT[v]); Lv = (ap_ + whd + ld) / 3; L4 = (ap_ + whd + fbd + ld) / 4
            S[f'resnet_{v}_current'] = np.where(ev, CUR[v], np.nan)
            S[f'resnet_{v}_logit_delta'] = np.where(ev, DLT[v], np.nan)
            S[f'HN_{v}'] = sig(np.where(ev, Lv, np.nan))
            S[f'HN_{v}_thr'] = sig(thr_apply(np.where(ev, Lv, np.nan), order, THR_K, 'mean'))
            S[f'F4_{v}'] = sig(np.where(ev, L4, np.nan))
            S[f'F4_{v}_thr'] = sig(thr_apply(np.where(ev, L4, np.nan), order, THR_K, 'mean'))
            f1v = adapt_stream(ZZt, np.c_[ap_, whd, ld], order); F1[v] = f1v
            S[f'P_{v}'] = sig(np.where(ev, f1v, np.nan))
            S[f'P_{v}_thr'] = sig(thr_apply(np.where(ev, f1v, np.nan), order, THR_K, 'mean'))
            print(f'[scores s{sd}] {v} branch/fusions/adapted assembled {el()}', flush=True)
        assert all(np.isfinite(S[q][ev]).all() for q in S), 'NaN inside the evaluated rows'

        # ---- checks: delta-head prefix invariance, pipeline prefix invariance, isolation, label shuffle
        for s in list(np.unique(g[ev]))[:2]:
            loc = order[sidx[s]]; ns = len(loc)
            cut = MIN_PAST + R_STEP * ((ns - MIN_PAST) // (2 * R_STEP)) + 5
            rng = np.random.default_rng(int(sidx[s]))
            for v in VARIANTS:
                l0 = DLT[v][loc].astype(np.float64)
                F2_0 = np.c_[l0, l0 - hist_subject(l0[:, None])[1][:, 0]]
                l1 = l0.copy(); l1[cut:] += rng.normal(0, .5, ns - cut)
                F2_1 = np.c_[l1, l1 - hist_subject(l1[:, None])[1][:, 0]]
                with threadpool_limits(limits=4):
                    hd = make_clf('lr').fit(F2_0, y[loc])
                p0, p1 = hd.predict_proba(F2_0)[:, 1], hd.predict_proba(F2_1)[:, 1]
                assert np.abs(p0[:cut] - p1[:cut]).max() == 0., ('delta head prefix', v, s)
                assert np.abs(p1[cut:] - p0[cut:]).max() > 0, ('delta head unchanged', v, s)
            Z0 = ZZt[loc]; B0 = np.c_[ap_[loc], whd[loc], logit(DLT['plain'][loc])]
            _, f0, _ = p1_subject(Z0, B0, y[loc])
            assert np.array_equal(f0, F1['plain'][loc], equal_nan=True), ('isolation', s)
            Z1, B1 = Z0.copy(), B0.copy(); Z1[cut:] += rng.normal(0, 2, Z1[cut:].shape)
            B1[cut:] += rng.normal(0, 2, B1[cut:].shape)
            _, f1c, _ = p1_subject(Z1, B1, y[loc])
            assert np.array_equal(f0[:cut], f1c[:cut], equal_nan=True), ('pipeline prefix', s)
            d = np.abs(f0[cut:] - f1c[cut:]); fin = np.isfinite(d)
            assert fin.any() and d[fin].max() > 0, ('pipeline unchanged', s)
            checks[f's{sd}_prefix_{s}'] = {'cut': int(cut), 'pipeline_prefix': 'ok', 'delta_head_prefix': 'ok',
                                           'isolation': 'ok'}
            print(f'[check s{sd}] {s}: delta-head / pipeline prefix + isolation OK (cut={cut})', flush=True)
        s0 = list(np.unique(g[ev]))[0]; loc = order[sidx[s0]]
        ysh = np.random.default_rng(int(sidx[s0])).permutation(y[loc])
        _, fsh, _ = p1_subject(ZZt[loc], np.c_[ap_[loc], whd[loc], logit(DLT['plain'][loc])], ysh)
        assert np.array_equal(fsh, F1['plain'][loc], equal_nan=True), 'test-label shuffle moved the legal pipeline'
        checks[f's{sd}_label_shuffle'] = 'legal pipeline identical under shuffled test labels'
        print(f'[check s{sd}] test-label shuffle leaves the legal pipeline identical', flush=True)

        # ---- evaluation
        M = {'all': ev, 'mature': ev & (t_c >= MATURE)}
        R = result['streams'][f'stream{sd}'] = {'methods': {}, 'bootstrap': {}, 'subjects': {}, 'groups': {}}
        ev_subs = list(np.unique(g[ev])); PS = {}
        for nm, p in S.items():
            R['methods'][nm] = {sg: metrics(y, g, p, M[sg]) for sg in M}
            PS[nm] = {s: metrics(y, g, p, ev & (g == s))['subject_macro_bACC'] for s in ev_subs}
            ba = np.array([q for q in PS[nm].values() if q is not None])
            R['subjects'][nm] = {'n_ge_0.9': int((ba >= .9).sum()), 'max': float(ba.max()), 'n': int(len(ba))}
            print(f'[result s{sd}] {nm:26s} ' + ' | '.join(
                f'{sg}: AUC={R["methods"][nm][sg]["AUC"]:.4f} mac={R["methods"][nm][sg]["subject_macro_bACC"]:.4f} '
                f'mic={R["methods"][nm][sg]["micro_bACC"]:.4f}' for sg in M) +
                f' | subj>=.9={R["subjects"][nm]["n_ge_0.9"]} max={R["subjects"][nm]["max"]:.3f}', flush=True)
        for nm in S:
            R['groups'][nm] = {}
            for G in GROUPS:
                ss = [s for s in ev_subs if grp_of.get(s) == G and PS[nm][s] is not None]
                if not ss: continue
                a = np.array([PS[nm][s] for s in ss])
                b = np.array([PS['fuse_pseudo_lr_full_thr'][s] for s in ss])
                c = np.array([PS['P_plain_thr'][s] for s in ss])
                R['groups'][nm][G] = {'n': len(ss), 'mean': float(a.mean()), 'median': float(np.median(a)),
                                      'dmean_vs_best': float(a.mean() - b.mean()),
                                      'dmean_vs_plain': float(a.mean() - c.mean())}
            print(f'[group s{sd}] {nm:26s} ' + ' | '.join(
                f'{G}(n={q["n"]}): mean={q["mean"]:.4f} d_best={q["dmean_vs_best"]:+.4f} '
                f'd_plain={q["dmean_vs_plain"]:+.4f}' for G, q in R['groups'][nm].items()), flush=True)
        pairs = []
        for v in VARIANTS:
            pairs += [(f'P_{v}_thr', 'fuse_pseudo_lr_full_thr'), (f'P_{v}_thr', 'P_plain_thr'),
                      (f'HN_{v}_thr', 'fuseH_wf_thr'), (f'HN_{v}_thr', 'HN_plain_thr'),
                      (f'F4_{v}_thr', 'fuseH_wf_thr'), (f'F4_{v}_thr', 'F4_plain_thr'),
                      (f'resnet_{v}_logit_delta', 'resnet_plain_logit_delta')]
        for a_, b_ in pairs:
            if a_ == b_: continue
            for sg in M:
                bb = R['bootstrap'][f'{a_}__vs__{b_}__{sg}'] = boot(S[a_], S[b_], y, g, M[sg], nb)
                print(f'[boot s{sd}] {a_:24s} vs {b_:24s} {sg:6s} '
                      f'dAUC={bb["dAUC"]["mean"]:+.4f}[{bb["dAUC"]["p2.5"]:+.4f},{bb["dAUC"]["p97.5"]:+.4f}] '
                      f'dmac={bb["dmacro_bACC"]["mean"]:+.4f}[{bb["dmacro_bACC"]["p2.5"]:+.4f},'
                      f'{bb["dmacro_bACC"]["p97.5"]:+.4f}] '
                      f'dmic={bb["dmicro_bACC"]["mean"]:+.4f}[{bb["dmicro_bACC"]["p2.5"]:+.4f},'
                      f'{bb["dmicro_bACC"]["p97.5"]:+.4f}]', flush=True)
        print(f'[stream {sd}] done {el()}', flush=True)
        (OUT / (stem + '.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')

    result['seconds'] = time.perf_counter() - T0
    (OUT / (stem + '.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[done] all checks passed {el()} -> {stem}.json', flush=True)
    sys.stdout = tee.stream; tee.f.close()


if __name__ == '__main__':
    main()
