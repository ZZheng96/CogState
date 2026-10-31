"""Shared harness for literature deep-model replications (single channel, 6 s @ 250 Hz, 57 subjects).

Protocols
  STRICT            outer SGKF(5, shuffle, rs=0) by subject (== depositioned_aperiodic_emg_oof.npz['fold']); per-window
                    demean/std/clip only; inner early-stop split = first split of SGKF(4, shuffle, rs=fold+1) on the
                    training subjects (validation subjects never trained on, never test).
  LIT_WINDOW        window-level KFold(5, shuffle, rs=0) ignoring subjects (leaky); per-window normalisation;
                    inner split = random 25 % of training rows (StratifiedKFold(4, shuffle, rs=fold+1) first split).
  LIT_SESSION_NORM  same subject-wise folds as STRICT, but each subject's raw windows are z-scored with that subject's
                    full-session mean/std (non-causal, uses the test subject's whole recording; no labels).
Training (all methods, same recipe as raw_waveform_window6): AdamW(LR, WD), BS, MAX_EP, early stop PATIENCE on val AUC,
random 4 s crop + amplitude scale + noise augmentation; inference = mean logit over N_TTA equispaced 4 s crops.
Metrics: AUC, subject-macro bACC (thr .5), #subjects bACC>=.9; STRICT: paired subject bootstrap vs stored ResNet.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, KFold, StratifiedKFold

HERE = Path(__file__).resolve().parent
TEST = HERE.parents[1]
sys.path.insert(0, str(TEST))
from raw_waveform_window6 import augment, normalize, FS, WIN, CROP, N_TTA, BS, MAX_EP, PATIENCE, LR, WD  # noqa: E402
from depositioned_resnet_history_delta import load_raw  # noqa: E402
from depositioned_history_delta import arrange, boot, load_db  # noqa: E402
from causal_kernel_state_mapping import metrics  # noqa: E402
from switch_dual_memory import Tee  # noqa: E402

PROTOCOLS = ('STRICT', 'LIT_WINDOW', 'LIT_SESSION_NORM')
REF_KEYS = ('stream0__resnet_fbank', 'stream0__resnet_whiten_logit_delta')
T0 = time.perf_counter()


def el():
    return f'elapsed={time.perf_counter() - T0:.1f}s'


def load_all():
    man, subs, data = load_db()
    assert float(man['fs']) == FS and float(man['window_seconds']) * FS == WIN
    Xraw, Fc, y, g = load_raw(subs)
    assert Xraw.shape == (len(y), WIN), Xraw.shape
    A = np.load(TEST / 'depositioned_aperiodic_emg_oof.npz')
    assert np.array_equal(A['y'], y) and np.array_equal(A['groups'], g)
    Xa, ya, ga, _ = arrange(subs, data, 0, 'independent'); fold_of = {}
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Xa, ya, ga)):
        for s in np.unique(ga[te]): fold_of[s] = k
    fold_s = np.array([fold_of[s] for s in g])
    assert np.array_equal(fold_s, A['fold']), 'derived subject folds != stored fold'
    fold_w = np.full(len(y), -1)
    for k, (_, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(y)): fold_w[te] = k
    return subs, Xraw, y, g, fold_s, fold_w


def session_norm(Xraw, g):
    """Subject-level amplitude scaling with the whole session: per-window demean (DC offsets differ hugely between
    windows), then divide by the subject's session std of the demeaned signal (median over windows of window std, robust
    to artefact windows), clip +-10. Relative amplitude between a subject's windows is kept (unlike STRICT).
    Tried first and rejected (input collapsed, AUC ~.53): plain session mean/std of raw x -> DC offsets and artefacts
    inflate the std, median window std ~.01, signal drowned by the .05 augmentation noise."""
    X = np.empty(Xraw.shape, np.float32)
    for s in np.unique(g):
        q = g == s; x = Xraw[q].astype(np.float64); x = x - x.mean(1, keepdims=True)
        X[q] = np.clip(x / max(np.median(x.std(1)), 1e-12), -10, 10)
    return X


def splits(protocol, y, g, fold):
    out = []
    for k in range(5):
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        if protocol == 'LIT_WINDOW':
            a, b = next(StratifiedKFold(4, shuffle=True, random_state=k + 1).split(tr, y[tr]))
        else:
            a, b = next(StratifiedGroupKFold(4, shuffle=True, random_state=k + 1).split(tr[:, None], y[tr], g[tr]))
            A_, B_ = tr[a], tr[b]
            assert not (set(g[A_]) & set(g[B_])) and not (set(g[tr]) & set(g[te])), 'subject leak'
        A_, B_ = tr[a], tr[b]
        assert not (set(A_) & set(B_)) and not (set(tr) & set(te))
        out.append((A_, B_, te))
    return out


@torch.no_grad()
def predict(net, Xg, ids):
    net.eval(); out = []
    offs = np.linspace(0, WIN - CROP, N_TTA).round().astype(int)
    for b in range(0, len(ids), 512):
        xb = Xg[torch.as_tensor(ids[b:b + 512], device=Xg.device)]
        out.append(torch.stack([net(xb[:, None, o:o + CROP]).ravel() for o in offs]).mean(0).cpu().numpy())
    return 1 / (1 + np.exp(-np.concatenate(out).astype(np.float64)))


def seed_all(s):
    torch.manual_seed(s); np.random.seed(s % (2 ** 32))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def train_eval(make, Xg, yg, y, A_, B_, te, seed, tag, max_ep):
    tick = time.perf_counter(); dev = Xg.device
    seed_all(seed); rng = np.random.default_rng(seed)
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    net = make().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD); crit = nn.BCEWithLogitsLoss()
    best, state, stale, best_ep = -1., None, 0, 0
    for ep in range(1, max_ep + 1):
        net.train(); perm = rng.permutation(A_); tot = 0.
        for b in range(0, len(perm), BS):
            ids = perm[b:b + BS]
            if len(ids) < 2: continue
            idt = torch.as_tensor(ids, device=dev)
            loss = crit(net(augment(Xg[idt], gen)).ravel(), yg[idt])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.); opt.step()
            if hasattr(net, 'constrain'): net.constrain()
            tot += loss.item() * len(ids)
        pv = predict(net, Xg, B_)
        vl = float(-np.mean(y[B_] * np.log(np.clip(pv, 1e-7, 1)) + (1 - y[B_]) * np.log(np.clip(1 - pv, 1e-7, 1))))
        auc = roc_auc_score(y[B_], pv)
        if auc > best + 1e-4:
            best, state, stale, best_ep = auc, {k: v.detach().clone() for k, v in net.state_dict().items()}, 0, ep
        else:
            stale += 1
        print(f'    [{tag}] ep{ep:02d}/{max_ep} train_loss={tot / len(A_):.4f} val_loss={vl:.4f} valAUC={auc:.4f} '
              f'best={best:.4f}@{best_ep} t={time.perf_counter() - tick:.0f}s {el()}', flush=True)
        if stale >= PATIENCE: break
    net.load_state_dict(state)
    info = {'best_epoch': int(best_ep), 'epochs_run': int(ep), 'best_val_auc': float(best),
            'params': int(sum(p.numel() for p in net.parameters())), 'seconds': float(time.perf_counter() - tick)}
    return predict(net, Xg, te), info


def summarize(y, g, p, mask):
    m = metrics(y, g, p, mask)
    per = {str(s): float(balanced_accuracy_score(y[q], p[q] >= .5)) for s in np.unique(g[mask]) if (q := mask & (g == s)).any()}
    return {'AUC': m['AUC'], 'subject_macro_bACC': m['subject_macro_bACC'], 'micro_bACC': m['micro_bACC'],
            'n_subjects_bACC_ge_.9': int(sum(v >= .9 for v in per.values())), 'n_subjects': len(per), 'per_subject_bACC': per}


def run(method, make):
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--seeds', default='0')
    ap.add_argument('--protocols', default=','.join(PROTOCOLS))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    protos = args.protocols.split(',')
    out = HERE / method; out.mkdir(exist_ok=True)
    stem = method + ('_smoke' if args.smoke else '')
    partial = set(protos) != set(PROTOCOLS)
    log_stem = stem + ('__' + '_'.join(protos) if partial else '')
    sys.stdout = Tee(sys.stdout, out / (log_stem + '.log'))
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_ep = 2 if args.smoke else MAX_EP
    run_folds = [0] if args.smoke else list(range(5))

    subs, Xraw, y, g, fold_s, fold_w = load_all()
    print(f'[start] method={method} rows={len(y)} subjects={len(subs)} X={Xraw.shape} fs={FS} device={dev} '
          f'smoke={args.smoke} seeds={seeds} protocols={protos} {el()}', flush=True)
    print('[check] derived SGKF folds == depositioned_aperiodic_emg_oof.npz[fold]', flush=True)
    Xw, n_flat = normalize(Xraw); Xs = session_norm(Xraw, g)
    yg = torch.tensor(y, dtype=torch.float32, device=dev)
    R = np.load(TEST / 'depositioned_resnet_timefreq_oof.npz')
    for q in ('y', 'groups'): assert np.array_equal(R[q], y if q == 'y' else g)
    assert np.array_equal(R['fold'], fold_s)

    res_path, npz_path = out / (stem + '.json'), out / (stem + '_oof.npz')
    result = {'meta': {'method': method, 'rows': int(len(y)), 'subjects': len(subs), 'fs': FS, 'win': WIN, 'crop': CROP,
                       'n_tta': N_TTA, 'bs': BS, 'lr': LR, 'wd': WD, 'max_ep': max_ep, 'patience': PATIENCE,
                       'flat_windows': n_flat, 'device': dev,
                       'gpu': torch.cuda.get_device_name(0) if dev == 'cuda' else None, 'smoke': args.smoke,
                       'folds': run_folds, 'threshold': .5, 'reference_keys': list(REF_KEYS)},
              'checks': {'fold_equality': True}, 'runs': [], 'results': {}}
    oof = {'y': y, 'groups': g, 'fold_subject': fold_s, 'fold_window': fold_w}
    if partial and res_path.exists() and npz_path.exists():  # merge: keep other protocols' results, replace these
        old = json.loads(res_path.read_text(encoding='utf-8'))
        with np.load(npz_path) as O: Od = {k: O[k] for k in O.files}
        keep = lambda k: k.split('__')[0] not in protos
        result['runs'] = [r for r in old['runs'] if r['protocol'] not in protos]
        result['results'] = {k: v for k, v in old['results'].items() if keep(k)}
        result['superseded'] = {k: {kk: vv for kk, vv in v.items() if kk != 'per_subject_bACC'}
                                for k, v in old['results'].items() if not keep(k)}
        result['checks'].update({k: v for k, v in old['checks'].items() if k != 'determinism_max_abs_diff'})
        result['meta']['merged_from_previous_run'] = sorted({r['protocol'] for r in result['runs']})
        oof.update({k: v for k, v in Od.items() if k not in oof and keep(k)})
        print(f'[merge] kept {sorted(result["results"])} from previous {res_path.name}', flush=True)

    # determinism check: same seed twice on a small subset -> identical predictions
    sp0 = splits('STRICT', y, g, fold_s)[0]
    Xg = torch.from_numpy(Xw).to(dev)
    sub = lambda a, n: a[np.random.default_rng(0).permutation(len(a))[:n]]
    pa, _ = train_eval(make, Xg, yg, y, sub(sp0[0], 1024), sub(sp0[1], 512), sp0[2][:256], 123, 'determinism a', 1)
    pb, _ = train_eval(make, Xg, yg, y, sub(sp0[0], 1024), sub(sp0[1], 512), sp0[2][:256], 123, 'determinism b', 1)
    dmax = float(np.abs(pa - pb).max()); result['checks']['determinism_max_abs_diff'] = dmax
    print(f'[check] determinism max|dp|={dmax:.2e}', flush=True)
    assert dmax < 1e-4, dmax

    for proto in protos:
        fold = fold_w if proto == 'LIT_WINDOW' else fold_s
        Xg = torch.from_numpy(Xs if proto == 'LIT_SESSION_NORM' else Xw).to(dev)
        sp = splits(proto, y, g, fold)
        mask = np.isin(fold, run_folds)
        for sd in seeds:
            P = np.full(len(y), np.nan)
            for k in run_folds:
                A_, B_, te = sp[k]
                tag = f'{method} {proto} s{sd} fold{k + 1}/5'
                P[te], info = train_eval(make, Xg, yg, y, A_, B_, te, 1000 * sd + k + 1, tag, max_ep)
                fa = float(roc_auc_score(y[te], P[te]))
                result['runs'].append({'protocol': proto, 'seed': sd, 'fold': k, **info, 'test_auc': fa})
                print(f'[fold] {tag} best_ep={info["best_epoch"]}/{info["epochs_run"]} valAUC={info["best_val_auc"]:.4f} '
                      f'testAUC={fa:.4f} params={info["params"]} {info["seconds"]:.0f}s {el()}', flush=True)
            oof[f'{proto}__s{sd}'] = P
            r = summarize(y, g, P, mask)
            if proto == 'STRICT':
                nb = 100 if args.smoke else 1000
                r['bootstrap_vs_ref'] = {rk: boot(P, R[rk], y, g, mask, n=nb) for rk in REF_KEYS}
                r['ref_scores'] = {rk: {kk: v for kk, v in summarize(y, g, R[rk], mask).items() if kk != 'per_subject_bACC'}
                                   for rk in REF_KEYS}
            result['results'][f'{proto}__s{sd}'] = r
            print(f'[result] {method} {proto} s{sd} AUC={r["AUC"]:.4f} macro_bACC={r["subject_macro_bACC"]:.4f} '
                  f'n>=.9={r["n_subjects_bACC_ge_.9"]}/{r["n_subjects"]} {el()}', flush=True)
            if 'bootstrap_vs_ref' in r:
                for rk, b in r['bootstrap_vs_ref'].items():
                    print(f'    vs {rk}: dAUC={b["dAUC"]["mean"]:+.4f} [{b["dAUC"]["p2.5"]:+.4f},{b["dAUC"]["p97.5"]:+.4f}] '
                          f'dmacro={b["dmacro_bACC"]["mean"]:+.4f} [{b["dmacro_bACC"]["p2.5"]:+.4f},{b["dmacro_bACC"]["p97.5"]:+.4f}]',
                          flush=True)
            res_path.write_text(json.dumps(result, indent=1), encoding='utf-8')
            np.savez_compressed(npz_path, **oof)
        if proto == 'STRICT' and len(seeds) > 1:
            Pm = np.mean([oof[f'STRICT__s{sd}'] for sd in seeds], 0)
            oof['STRICT__seedmean'] = Pm
            r = summarize(y, g, Pm, mask)
            r['bootstrap_vs_ref'] = {rk: boot(Pm, R[rk], y, g, mask, n=100 if args.smoke else 1000) for rk in REF_KEYS}
            result['results']['STRICT__seedmean'] = r
            print(f'[result] {method} STRICT seedmean AUC={r["AUC"]:.4f} macro_bACC={r["subject_macro_bACC"]:.4f} {el()}', flush=True)
        del Xg; torch.cuda.empty_cache()
    res_path.write_text(json.dumps(result, indent=1), encoding='utf-8')
    np.savez_compressed(npz_path, **oof)
    print(f'[done] {res_path} {npz_path} {el()}', flush=True)
