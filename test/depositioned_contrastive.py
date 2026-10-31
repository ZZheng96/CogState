"""Direction F: within-subject contrastive pretraining (InfoNCE) of a 1D CNN encoder, de-positioned streams.

Protocol: stream order = concat(relax,task) permuted by default_rng([stream_seed, subject_index]) (stream_order);
outer folds = SGKF(5, shuffle, rs0) on seed-0 arrangement (asserted == fold in depositioned_feature_multiscale_oof.npz).
Hyper-parameters fixed before running (not tuned on results):
  Encoder: ResNet1d stem+blocks of raw_waveform_window6 -> GAP (128) -> Linear(128,64) = embedding h;
           projection g = Linear(64,64)-ReLU-Linear(64,64), L2-normalised, used only in the loss.
  Input: normalize() of raw 6 s window (per-window demean/std/clip 10); views = augment() (random 4 s crop,
         amplitude scale U(.8,1.25) log-uniform, noise sd .05), two independent views.
  InfoNCE (NT-Xent, temperature .1), strictly WITHIN subject: batch = 16 train subjects x 16 anchors; anchor = random
         canonical row i of the subject, positive = row i+1 (adjacent window in the original recording order
         concat(relax,task); the single relax->task boundary pair per subject is kept, so no labels are used);
         negatives = the other 30 views of the SAME subject only (so subject identity cannot solve the task).
  Optimiser: AdamW lr 1e-3 wd 1e-2, grad clip 1, 40 epochs x 50 steps, no early stopping, no labels.
  Encoder is trained on the outer-train subjects only, once per fold (torch seed 1000+fold), and shared by
  the 3 stream seeds (encoder sees one window at a time -> order independent). Embedding = mean over 5 equispaced
  4 s crops (TTA).
  Downstream (per stream seed, train and test subjects both in their own stream order):
    Es = StandardScaler(outer-train rows) of h.
    Zc = causal within-subject standardisation: (Es_t - mean) / sqrt(var + 1e-3) over the subject's stream rows
         max(0,t-127)..t (current + up to 127 previous windows); 0 if fewer than 8 rows available.
    contrastive_lr  = LR(C .05) on [Zc]               (primary)
    contrastive_hgb = HGB(200, depth3, lr .06, l2 1) on [Es, Zc]
    Heads fit on all outer-train rows (encoder never saw labels).
  Fusion: fusionF = sigmoid(mean(logit fusionM_prank, logit contrastive_lr)) (primary);
          fusionF_hgb = same with contrastive_hgb.
  fusionM_prank / position-only from depositioned_feature_multiscale_oof.npz (position-only recomputed, asserted).
Evaluation: AUC, subject-macro / micro bACC at .5, all rows and mature (t >= 120); paired subject bootstrap 1000 vs
fusionM_prank; per-subject bACC >= .9 count; low/mid/high group (depositioned_subject_diagnosis.json) median change.
Checks: position-only AUC ~.5; subject isolation (alone == pooled); shuffled test labels -> identical scores;
prefix invariance (changing future raw windows leaves earlier scores unchanged, changes later ones); Zc vs naive.
"""
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_v] = '4'
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as Fnn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from raw_waveform_window6 import ResNet1d, augment, normalize, WIN, CROP, N_TTA  # noqa: E402
from depositioned_history_delta import arrange, make_clf, boot, load_db  # noqa: E402
from depositioned_resnet_history_delta import load_raw, stream_order, logit, sig  # noqa: E402
from causal_kernel_state_mapping import metrics  # noqa: E402
from switch_dual_memory import Tee  # noqa: E402

MATURE, EMB, TAU = 120, 64, .1
P_SUB, M_ANC, EPOCHS, STEPS, LR_, WD_ = 16, 16, 40, 50, 1e-3, 1e-2
KZ, MINZ, EPS = 128, 8, 1e-3
STREAM_SEEDS = (0, 1, 2)
T0 = time.perf_counter()


def el():
    return f'elapsed={time.perf_counter() - T0:.1f}s'


class Enc(nn.Module):
    def __init__(self):
        super().__init__()
        r = ResNet1d(); self.stem, self.blocks = r.stem, r.blocks
        self.fc = nn.Linear(128, EMB)
        self.proj = nn.Sequential(nn.Linear(EMB, EMB), nn.ReLU(), nn.Linear(EMB, EMB))

    def emb(self, x):
        return self.fc(self.blocks(self.stem(x)).mean(-1))


def nce_loss(z):
    """z (P, 2M, d) L2-normalised; view k's positive is k+M mod 2M; negatives = other views of same subject."""
    P, V, _ = z.shape; M = V // 2
    sim = z @ z.transpose(1, 2) / TAU
    sim = sim.masked_fill(torch.eye(V, dtype=torch.bool, device=z.device)[None], float('-inf'))
    tgt = ((torch.arange(V, device=z.device) + M) % V)[None].expand(P, V)
    loss = Fnn.cross_entropy(sim.reshape(P * V, V), tgt.reshape(-1))
    acc = (sim.argmax(-1) == tgt).float().mean()
    return loss, acc


def train_encoder(Xg, g, tr, seed, tag, epochs):
    tick = time.perf_counter(); dev = Xg.device
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    trs = np.unique(g[tr])
    anchors = {}
    for s in trs:
        loc = np.flatnonzero(g == s); anchors[s] = loc[:-1]  # canonical consecutive rows -> positive = loc+1
    net = Enc().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR_, weight_decay=WD_)
    hist = []
    for ep in range(1, epochs + 1):
        net.train(); tl, ta = 0., 0.
        for _ in range(STEPS):
            ss = rng.choice(trs, P_SUB, replace=False)
            a = np.stack([rng.choice(anchors[s], M_ANC, replace=False) for s in ss])  # (P, M)
            ids = np.concatenate([a, a + 1], 1).ravel()                                  # (P*2M)
            xb = augment(Xg[torch.as_tensor(ids, device=dev)], gen)
            z = Fnn.normalize(net.proj(net.emb(xb)), dim=-1).view(P_SUB, 2 * M_ANC, EMB)
            loss, acc = nce_loss(z)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.); opt.step()
            tl += loss.item(); ta += acc.item()
        hist.append({'ep': ep, 'loss': tl / STEPS, 'pos_top1': ta / STEPS})
        print(f'    [{tag}] ep{ep:02d}/{epochs} nce={tl / STEPS:.4f} pos_top1={ta / STEPS:.3f} (chance {1 / (2 * M_ANC - 1):.3f}) '
              f't={time.perf_counter() - tick:.0f}s {el()}', flush=True)
    net.eval()
    return net, {'history': hist, 'seconds': float(time.perf_counter() - tick)}


@torch.no_grad()
def embed(net, X):
    """X: tensor (n, WIN) normalised -> (n, EMB) mean over 5 equispaced 4 s crops."""
    net.eval(); offs = np.linspace(0, WIN - CROP, N_TTA).round().astype(int); out = []
    for b in range(0, len(X), 512):
        xb = X[b:b + 512]
        out.append(torch.stack([net.emb(xb[:, None, o:o + CROP]) for o in offs]).mean(0))
    return torch.cat(out).cpu().numpy().astype(np.float64)


def cz_subject(es):
    """Causal within-subject z: rows max(0,t-KZ+1)..t (incl. current); 0 if < MINZ rows."""
    n = len(es); c = es[:1]; d = es - c
    cs = np.concatenate([np.zeros((1, es.shape[1])), np.cumsum(d, 0)])
    cq = np.concatenate([np.zeros((1, es.shape[1])), np.cumsum(d * d, 0)])
    hi = np.arange(n) + 1; lo = np.maximum(0, hi - KZ); cnt = (hi - lo).astype(float)[:, None]
    m1 = (cs[hi] - cs[lo]) / cnt; var = np.maximum((cq[hi] - cq[lo]) / cnt - m1 * m1, 0.)
    return np.where(cnt >= MINZ, (d - m1) / np.sqrt(var + EPS), 0.)


def cz_naive(es, t):
    P = es[max(0, t - KZ + 1):t + 1]
    return (es[t] - P.mean(0)) / np.sqrt(P.var(0) + EPS) if len(P) >= MINZ else np.zeros(es.shape[1])


def cz_all(Es, order):
    Z = np.empty_like(Es)
    for loc in order: Z[loc] = cz_subject(Es[loc])
    return Z


def fit_heads(Es, Zc, y, tr, te):
    m_lr = make_clf('lr').fit(Zc[tr], y[tr]); F = np.hstack([Es, Zc])
    m_hgb = make_clf('hgb').fit(F[tr], y[tr])
    return m_lr, m_hgb, m_lr.predict_proba(Zc[te])[:, 1], m_hgb.predict_proba(F[te])[:, 1]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); args = ap.parse_args()
    stem = 'depositioned_contrastive' + ('_smoke' if args.smoke else '')
    tee = Tee(sys.stdout, OUT / (stem + '.log')); sys.stdout = tee
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    sseeds = STREAM_SEEDS[:1] if args.smoke else STREAM_SEEDS
    epochs = 3 if args.smoke else EPOCHS
    nb = 100 if args.smoke else 1000

    man, subs, data = load_db()
    Xraw, Fc, y, g = load_raw(subs)
    prior = np.load(OUT / 'depositioned_feature_multiscale_oof.npz')
    assert np.array_equal(prior['y'], y) and np.array_equal(prior['groups'], g)
    diag = json.loads((OUT / 'depositioned_subject_diagnosis.json').read_text(encoding='utf-8'))['diagnosis']['groups']
    groups = {k: diag[k]['subjects'] for k in ('low', 'mid', 'high')}
    Xa, ya, ga, _ = arrange(subs, data, 0, 'independent'); fold_of = {}
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Xa, ya, ga)):
        for s in np.unique(ga[te]): fold_of[s] = k
    fold = np.array([fold_of[s] for s in g]); assert np.array_equal(fold, prior['fold'])
    run_folds = [0] if args.smoke else list(range(5))
    print(f'[start] rows={len(y)} subjects={len(subs)} device={dev} smoke={args.smoke} sseeds={sseeds} epochs={epochs} '
          f'groups={ {k: len(v) for k, v in groups.items()} }', flush=True)
    Xn, n_flat = normalize(Xraw); Xg = torch.from_numpy(Xn).to(dev)
    sidx = {s: j for j, s in enumerate(subs)}

    result = {'meta': {'emb': EMB, 'tau': TAU, 'subjects_per_batch': P_SUB, 'anchors_per_subject': M_ANC,
                       'epochs': epochs, 'steps_per_epoch': STEPS, 'lr': LR_, 'wd': WD_, 'kz': KZ, 'min_z': MINZ,
                       'eps': EPS, 'mature_index': MATURE, 'threshold': .5, 'stream_seeds': list(sseeds),
                       'smoke': args.smoke, 'device': dev, 'flat_windows': n_flat,
                       'gpu': torch.cuda.get_device_name(0) if dev == 'cuda' else None,
                       'encoder_shared_across_streams': True}, 'encoders': [], 'checks': {}}
    checks = result['checks']; oof = {'y': y, 'groups': g, 'fold': fold}

    # ---------------- encoder pretraining (label free, train subjects only, once per fold) ----------------
    EM, nets = {}, {}
    for k in run_folds:
        tr = np.flatnonzero(fold != k); te = np.flatnonzero(fold == k)
        assert not set(g[tr]) & set(g[te])
        net, info = train_encoder(Xg, g, tr, 1000 + k, f'encoder fold{k + 1}/5', epochs)
        EM[k] = embed(net, Xg); nets[k] = net
        # embedding subject isolation + per-window locality
        s0 = np.unique(g[te])[0]; loc = np.flatnonzero(g == s0)
        d_iso = float(np.abs(embed(net, Xg[torch.as_tensor(loc, device=dev)]) - EM[k][loc]).max())
        assert d_iso < 1e-3, d_iso; checks[f'emb_isolation_fold{k}'] = d_iso
        result['encoders'].append({'fold': k, **info})
        print(f'[encoder] fold{k + 1}/5 done {info["seconds"]:.0f}s final nce={info["history"][-1]["loss"]:.4f} '
              f'pos_top1={info["history"][-1]["pos_top1"]:.3f} iso={d_iso:.1e} {el()}', flush=True)

    # ---------------- downstream per stream ----------------
    names = ('contrastive_lr', 'contrastive_hgb')
    for sd in sseeds:
        ts = time.perf_counter(); order, t = stream_order(g, subs, sd)
        S = {n: np.full(len(y), np.nan) for n in ('position_only__lr', 'position_only__hgb') + names}
        with threadpool_limits(limits=4):
            for k in run_folds:
                tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k); T = t[:, None].astype(float)
                for c in ('lr', 'hgb'):
                    S[f'position_only__{c}'][te] = make_clf(c).fit(T[tr], y[tr]).predict_proba(T[te])[:, 1]
                sc = StandardScaler().fit(EM[k][tr]); Es = sc.transform(EM[k]); Zc = cz_all(Es, order)
                for s0 in list(np.unique(g[te])[:2]) + [np.unique(g[tr])[0]]:
                    loc = order[sidx[s0]]
                    for r in sorted({0, 6, 7, 8, 50, 126, 127, 128, 200, len(loc) - 1} & set(range(len(loc)))):
                        assert np.allclose(Zc[loc[r]], cz_naive(Es[loc], r), atol=1e-8), (s0, r)
                m_lr, m_hgb, p_lr, p_hgb = fit_heads(Es, Zc, y, tr, te)
                S['contrastive_lr'][te], S['contrastive_hgb'][te] = p_lr, p_hgb
                # shuffled test labels -> identical scores
                yp = y.copy(); yp[te] = np.random.default_rng(sd * 10 + k).permutation(y[te])
                _, _, q_lr, q_hgb = fit_heads(Es, Zc, yp, tr, te)
                d_lab = float(max(np.abs(q_lr - p_lr).max(), np.abs(q_hgb - p_hgb).max()))
                assert d_lab == 0., d_lab; checks[f'test_label_shuffle_s{sd}_f{k}'] = d_lab
                # isolation + prefix invariance from raw for 2 test subjects
                for s0 in np.unique(g[te])[:2]:
                    loc = order[sidx[s0]]; cut = len(loc) // 2; pos = np.searchsorted(te, loc)

                    def pipe(Xs):
                        es = sc.transform(embed(nets[k], Xs)); zc = cz_subject(es)
                        return m_lr.predict_proba(zc)[:, 1], m_hgb.predict_proba(np.hstack([es, zc]))[:, 1]
                    Xs = Xg[torch.as_tensor(loc, device=dev)].clone(); a0, b0 = pipe(Xs)
                    Xs2 = Xs.clone(); Xs2[cut:] = Xs2[cut:].flip(-1) * -1.; a1, b1 = pipe(Xs2)
                    d_iso = float(max(np.abs(a0 - p_lr[pos]).max(), np.abs(b0 - p_hgb[pos]).max()))
                    d_pre = float(max(np.abs(a1[:cut] - a0[:cut]).max(), np.abs(b1[:cut] - b0[:cut]).max()))
                    d_fut = float(min(np.abs(a1[cut:] - a0[cut:]).max(), np.abs(b1[cut:] - b0[cut:]).max()))
                    assert d_iso < 1e-4 and d_pre < 1e-4 and d_fut > 1e-4, (d_iso, d_pre, d_fut)
                    checks[f'causal_s{sd}_f{k}_{s0}'] = {'alone_vs_pooled': d_iso, 'prefix_after_future_change': d_pre,
                                                         'future_changed': d_fut}
                print(f'[stream {sd} fold {k + 1}/5] heads fit; testAUC lr={roc_auc_score(y[te], p_lr):.4f} '
                      f'hgb={roc_auc_score(y[te], p_hgb):.4f}; checks ok (naive/label-shuffle/isolation/prefix) {el()}', flush=True)
        ev = np.isfinite(S['contrastive_lr'])
        for c in ('lr', 'hgb'):
            n = f'position_only__{c}'; dmax = float(np.abs(S[n][ev] - prior[f'stream{sd}__{n}'][ev]).max())
            assert dmax < 1e-9, (n, dmax); checks[f'position_reproduced_s{sd}_{c}'] = dmax
        S['fusionM_prank'] = np.where(ev, prior[f'stream{sd}__fusionM_prank'], np.nan)
        S['fusionF'] = sig((logit(S['fusionM_prank']) + logit(S['contrastive_lr'])) / 2)
        S['fusionF_hgb'] = sig((logit(S['fusionM_prank']) + logit(S['contrastive_hgb'])) / 2)

        M = {'all': ev, 'mature': ev & (t >= MATURE)}
        R = result[f'stream{sd}'] = {'methods': {}, 'bootstrap': {}, 'subjects': {}, 'groups': {}}
        per = {}
        for n, p in S.items():
            oof[f'stream{sd}__{n}'] = p
            r = R['methods'][n] = {s: metrics(y, g, p, M[s]) for s in M}
            per[n] = {s: metrics(y, g, p, ev & (g == s)) for s in np.unique(g[ev])}
            ba = np.array([v['subject_macro_bACC'] for v in per[n].values() if v['subject_macro_bACC'] is not None])
            R['subjects'][n] = {'n_ge_0.9': int((ba >= .9).sum()), 'max': float(ba.max()), 'n': int(len(ba))}
            print(f'[result stream {sd}] {n:18s} ' + ' | '.join(
                f'{s}: AUC={r[s]["AUC"]:.4f} mac={r[s]["subject_macro_bACC"]:.4f} mic={r[s]["micro_bACC"]:.4f}' for s in M) +
                f' | subj>=.9={R["subjects"][n]["n_ge_0.9"]} max={R["subjects"][n]["max"]:.3f}', flush=True)
        oof[f't_stream{sd}'] = t
        for c in ('lr', 'hgb'):
            a = R['methods'][f'position_only__{c}']['all']['AUC']
            R.setdefault('sanity', {})[f'position_only__{c}'] = {'AUC': a, 'ok': abs(a - .5) < .03}
            print(f'[sanity stream {sd}] position_only__{c} AUC={a:.4f} -> {"OK" if abs(a - .5) < .03 else "FLAG"}', flush=True)
            if not args.smoke: assert abs(a - .5) < .03
        for n in ('contrastive_lr', 'contrastive_hgb', 'fusionF', 'fusionF_hgb'):
            for seg in M:
                bb = R['bootstrap'][f'{n}__vs__fusionM_prank__{seg}'] = boot(S[n], S['fusionM_prank'], y, g, M[seg], nb)
                print(f'[boot stream {sd}] {n:16s} vs fusionM_prank {seg:6s} '
                      f'dAUC={bb["dAUC"]["mean"]:+.4f}[{bb["dAUC"]["p2.5"]:+.4f},{bb["dAUC"]["p97.5"]:+.4f}] '
                      f'dmac={bb["dmacro_bACC"]["mean"]:+.4f}[{bb["dmacro_bACC"]["p2.5"]:+.4f},{bb["dmacro_bACC"]["p97.5"]:+.4f}] '
                      f'dmic={bb["dmicro_bACC"]["mean"]:+.4f}[{bb["dmicro_bACC"]["p2.5"]:+.4f},{bb["dmicro_bACC"]["p97.5"]:+.4f}]',
                      flush=True)
            for gname, gs in groups.items():
                gs = [s for s in gs if s in per[n]]
                if not gs: continue
                dA = [per[n][s]['AUC'] - per['fusionM_prank'][s]['AUC'] for s in gs]
                dB = [per[n][s]['subject_macro_bACC'] - per['fusionM_prank'][s]['subject_macro_bACC'] for s in gs]
                R['groups'][f'{n}__{gname}'] = {'n': len(gs), 'median_dAUC': float(np.median(dA)),
                                                'median_dbACC': float(np.median(dB)),
                                                'median_AUC': float(np.median([per[n][s]['AUC'] for s in gs])),
                                                'median_bACC': float(np.median([per[n][s]['subject_macro_bACC'] for s in gs]))}
                q = R['groups'][f'{n}__{gname}']
                print(f'[group stream {sd}] {n:16s} {gname:4s} n={len(gs):2d} medAUC={q["median_AUC"]:.4f} '
                      f'medbACC={q["median_bACC"]:.4f} med dAUC={q["median_dAUC"]:+.4f} med dbACC={q["median_dbACC"]:+.4f}', flush=True)
        print(f'[progress] stream {sd} finished in {time.perf_counter() - ts:.0f}s {el()}', flush=True)
        (OUT / (stem + '.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
        np.savez_compressed(OUT / (stem + '_oof.npz'), **oof)
    result['seconds'] = time.perf_counter() - T0
    (OUT / (stem + '.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    np.savez_compressed(OUT / (stem + '_oof.npz'), **oof)
    print(f'[done] {el()} -> {stem}.json', flush=True)
    sys.stdout = tee.stream; tee.f.close()


if __name__ == '__main__':
    main()
