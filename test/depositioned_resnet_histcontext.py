"""Direction 2: ResNet sees the current window AND a causal summary of this subject's past unlabeled windows.

Protocol identical to depositioned_resnet_histnorm.py (stream order = concat(relax,task) permuted by
default_rng([stream_seed, subject_index]); outer SGKF(5, rs0) on seed-0 arrangement; per fold A/B = first split of
SGKF(4, rs=k+1); training on A, early stop on B val AUC, 5-crop TTA, AdamW/BS/patience from raw_waveform_window6;
ResNet seeds 0,1,2 (-> 1000*rs+k+1) on stream 0, seed 0 only on streams 1/2). History = previous <=32 windows of the
same subject in stream order (train subjects too), rebuilt per stream seed. Time-domain input = the 2-channel whiten
input of histnorm ([normalize(x), normalize(irfft(rFFT(x)/sqrt(P_past32)))], fallback = fold-train mean spectrum).
Variants (fixed before running):
  stage-1 resnet_whiten  re-trained with exactly the histnorm code path (needed for embeddings); run-level test AUC is
                     compared against depositioned_resnet_histnorm.json and logged.
  resnet_ctxspec     ResNet1d(2ch whiten) 128-d pooled embedding  ||  MLP(60->64->32, ReLU, dropout .3) on the spectral
                     context d = L_cur - mean(L of past <=32 own windows), L = log(mean |rFFT(x-mean)|^2 in 1 Hz bins
                     0-60 Hz + eps), eps = 1e-6 * mean train bin power; no history -> past = fold-train mean L.
                     d standardized with fold-train (A+B, unlabeled use) mean/std. -> dropout .3 -> Linear(160,1).
                     Trained end-to-end on A, early stop on B; TTA over time crops (d fixed per window).
  resnet_ctxemb      two-stage. Stage-1 = the whiten ResNet above (trained on A). Features per window, embeddings
                     e = TTA-mean 128-d stage-1 embedding standardized with B statistics, e_past = mean of past <=32
                     own e (0 if none), flag = has history: [e, e - e_past, e_past, flag] (385-d).
                     A-bias handling (fixed): A embeddings are in-sample for stage-1 and are NOT used. The head
                     (MLP 385->64 ReLU dropout .3 ->1, AdamW 1e-3/1e-2, BS 128, max 100 ep, patience 10) is trained
                     only on B with 4-fold cross-fitting over B subjects (SGKF(4, rs=k+1) on B): for inner part j,
                     train on the two remaining parts, early stop on part (j+1)%4, predict part j (B out-of-fold) and
                     test (mean logit over the 4 inner heads). B embeddings are out-of-sample for stage-1 weights
                     (B was used only for its early stopping).
  X_logit_delta      LR on B [l, l - mean(past 32 l)] (l standardized with B stats; for ctxemb l on B = cross-fit OOF),
                     prob mean over ResNet seeds.
  fusionC_X          sigmoid(mean(logit hgb_delta32, logit X_logit_delta)); compared with fusionH_whiten (recomputed
                     here = sigmoid(mean(logit hgb_delta32, logit whiten_logit_delta))), resnet_whiten, fusionR_delta.
Baselines hgb_delta32 / position controls recomputed and asserted equal to depositioned_resnet_history_delta_oof.npz;
fusionR_delta loaded from that file. Asserted checks: position-only AUC ~ .5 (full run), subject isolation (subject
computed alone == pooled) and causal prefix invariance (subject's second half raw *50+1 -> first-half inputs and scores
unchanged, second half changed) for every new pipeline. No test labels anywhere; threshold .5.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from raw_waveform_window6 import normalize, WIN, CROP, N_TTA, BS, MAX_EP, PATIENCE, LR, WD  # noqa: E402
from depositioned_history_delta import arrange, make_clf, hist_subject, boot, load_db  # noqa: E402
from depositioned_resnet_history_delta import load_raw, stream_order, hist_delta, logit, sig  # noqa: E402
from depositioned_resnet_histnorm import (ResNet1dC, augment_mc, past_mean, build_subject, fallback, build_all,  # noqa: E402
                                          train_resnet, predict)
from causal_kernel_state_mapping import metrics  # noqa: E402
from switch_dual_memory import Tee  # noqa: E402

K, MATURE, NBIN, FPB = 32, 120, 60, 6  # 1500 pts @250 Hz -> 1/6 Hz resolution -> 6 rFFT bins per 1 Hz
RESNET_SEEDS = (0, 1, 2)
STREAM_SEEDS = (0, 1, 2)
MLP_EP, MLP_PAT = 100, 10
T0 = time.perf_counter()
WV = 'resnet_whiten'


def el():
    return f'elapsed={time.perf_counter() - T0:.1f}s'


# ---------------- spectral context ----------------
def binpow(xs):
    xd = xs.astype(np.float64); xd = xd - xd.mean(1, keepdims=True)
    P = np.abs(np.fft.rfft(xd, axis=1)[:, :NBIN * FPB]) ** 2
    return P.reshape(len(xs), NBIN, FPB).mean(2)


def spec_stats(Xraw, rows):
    P = np.vstack([binpow(Xraw[rows[b:b + 2048]]) for b in range(0, len(rows), 2048)])
    eps = 1e-6 * P.mean()
    return np.log(P + eps).mean(0), eps


def spec_subject(xs, G, eps):
    L = np.log(binpow(xs) + eps); Pm = past_mean(L)
    Pm = np.where(np.isfinite(Pm).all(1, keepdims=True), Pm, G[None, :])
    return L - Pm


class ResNetCtx(nn.Module):
    def __init__(self, in_ch=2, ds=NBIN):
        super().__init__()
        self.trunk = ResNet1dC(in_ch)
        self.spec = nn.Sequential(nn.Linear(ds, 64), nn.ReLU(), nn.Dropout(.3), nn.Linear(64, 32), nn.ReLU())
        self.out = nn.Sequential(nn.Dropout(.3), nn.Linear(128 + 32, 1))

    def forward(self, x, s):
        return self.out(torch.cat([self.trunk.blocks(self.trunk.stem(x)).mean(-1), self.spec(s)], 1))


@torch.no_grad()
def predict_ctx(net, Xs, Ss):
    net.eval(); offs = np.linspace(0, WIN - CROP, N_TTA).round().astype(int); out = []
    for b in range(0, len(Xs), 512):
        xb, sb = Xs[b:b + 512], Ss[b:b + 512]
        out.append(torch.stack([net(xb[:, :, o:o + CROP], sb).ravel() for o in offs]).mean(0))
    return torch.cat(out).cpu().numpy().astype(np.float64)


def train_ctx(Xg, Sg, yg, y, tr, va, seed, tag, max_ep):
    tick = time.perf_counter(); dev = Xg.device
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    net = ResNetCtx(Xg.shape[1], Sg.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD); crit = nn.BCEWithLogitsLoss()
    best, state, stale, best_ep = -1., None, 0, 0
    vat = torch.as_tensor(va, device=dev)
    for ep in range(1, max_ep + 1):
        net.train(); perm = rng.permutation(tr); tot = 0.
        for b in range(0, len(perm), BS):
            ids = perm[b:b + BS]
            if len(ids) < 2: continue
            idt = torch.as_tensor(ids, device=dev)
            loss = crit(net(augment_mc(Xg[idt], gen), Sg[idt]).ravel(), yg[idt])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.); opt.step()
            tot += loss.item() * len(ids)
        auc = roc_auc_score(y[va], predict_ctx(net, Xg[vat], Sg[vat]))
        if auc > best + 1e-4:
            best, state, stale, best_ep = auc, {k: v.detach().clone() for k, v in net.state_dict().items()}, 0, ep
        else:
            stale += 1
        print(f'    [{tag}] ep{ep:02d}/{max_ep} loss={tot / len(tr):.4f} valAUC={auc:.4f} best={best:.4f}@{best_ep} '
              f't={time.perf_counter() - tick:.0f}s {el()}', flush=True)
        if stale >= PATIENCE: break
    net.load_state_dict(state); net.eval()
    return net, {'best_epoch': int(best_ep), 'epochs_run': int(ep), 'best_val_auc': float(best),
                 'seconds': float(time.perf_counter() - tick)}


# ---------------- stage-1 embedding context ----------------
@torch.no_grad()
def embed_c(net, Xs):
    """TTA-mean pooled 128-d embedding of ResNet1dC; logit = head Linear(emb)."""
    net.eval(); offs = np.linspace(0, WIN - CROP, N_TTA).round().astype(int); E = []
    for b in range(0, len(Xs), 512):
        xb = Xs[b:b + 512]
        E.append(torch.stack([net.blocks(net.stem(xb[:, :, o:o + CROP])).mean(-1) for o in offs]).mean(0))
    E = torch.cat(E)
    return E.cpu().numpy().astype(np.float64), net.head[3](E).ravel().cpu().numpy().astype(np.float64)


def emb_feats(Es):
    """Es: standardized embeddings of ONE subject in stream order -> [e, e - e_past, e_past, flag]."""
    Ep = past_mean(Es); flag = np.isfinite(Ep).all(1, keepdims=True).astype(float)
    Ep = np.where(flag > 0, Ep, 0.)
    return np.hstack([Es, Es - Ep, Ep, flag])


def train_mlp(F, y, tr, va, seed, dev, max_ep):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    Fg = torch.tensor(F, dtype=torch.float32, device=dev); yg = torch.tensor(y, dtype=torch.float32, device=dev)
    net = nn.Sequential(nn.Linear(F.shape[1], 64), nn.ReLU(), nn.Dropout(.3), nn.Linear(64, 1)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD); crit = nn.BCEWithLogitsLoss()
    best, state, stale, best_ep = -1., None, 0, 0
    for ep in range(1, max_ep + 1):
        net.train(); perm = rng.permutation(tr)
        for b in range(0, len(perm), BS):
            idt = torch.as_tensor(perm[b:b + BS], device=dev)
            if len(idt) < 2: continue
            loss = crit(net(Fg[idt]).ravel(), yg[idt]); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): auc = roc_auc_score(y[va], net(Fg[torch.as_tensor(va, device=dev)]).ravel().cpu().numpy())
        if auc > best + 1e-4:
            best, state, stale, best_ep = auc, {k: v.detach().clone() for k, v in net.state_dict().items()}, 0, ep
        else:
            stale += 1
        if stale >= MLP_PAT: break
    net.load_state_dict(state); net.eval()
    return net, best, best_ep


@torch.no_grad()
def mlp_logit(net, F, dev):
    return net(torch.tensor(F, dtype=torch.float32, device=dev)).ravel().cpu().numpy().astype(np.float64)


def lr_delta_head(L, B, te, order, y):
    """Standardize logit with B stats; LR on B [l, l - past32 l]; returns head, mu, sd, prob on te."""
    mu, sdv = L[B].mean(), max(L[B].std(), 1e-8)
    ls = np.where(np.isfinite(L), (L - mu) / sdv, 0.); F2 = np.c_[ls, hist_delta(ls[:, None], order)]
    head = make_clf('lr').fit(F2[B], y[B])
    return head, mu, sdv, head.predict_proba(F2[te])[:, 1]


def delta_prob(head, mu, sdv, l):
    ls = (l - mu) / sdv
    return head.predict_proba(np.c_[ls, ls - hist_subject(ls[:, None])[1][:, 0]])[:, 1]


def assert_causal(name, xa_in, xb_in, pooled_p, sc0, sc1, cut, checks, key):
    l0, p0 = sc0; l1, p1 = sc1
    in_pre = float(max(np.abs(a[:cut] - b[:cut]).max() for a, b in zip(xa_in, xb_in)))
    in_fut = float(max(np.abs(a[cut:] - b[cut:]).max() for a, b in zip(xa_in, xb_in)))
    d_iso = float(np.abs(p0 - pooled_p).max())
    d_pre = float(max(np.abs(l1[:cut] - l0[:cut]).max(), np.abs(p1[:cut] - p0[:cut]).max()))
    d_fut = float(np.abs(p1[cut:] - p0[cut:]).max())
    assert in_pre == 0. and in_fut > 1e-3 and d_iso < 1e-4 and d_pre < 1e-4 and d_fut > 1e-4, \
        (name, in_pre, in_fut, d_iso, d_pre, d_fut)
    checks[key] = {'input_prefix_maxabs': in_pre, 'input_future_maxabs': in_fut, 'score_alone_vs_pooled': d_iso,
                   'score_prefix_after_future_change': d_pre, 'score_future_changed': d_fut}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); args = ap.parse_args()
    stem = 'depositioned_resnet_histcontext' + ('_smoke' if args.smoke else '')
    tee = Tee(sys.stdout, OUT / (stem + '.log')); sys.stdout = tee
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    sseeds = STREAM_SEEDS[:1] if args.smoke else STREAM_SEEDS
    max_ep = 2 if args.smoke else MAX_EP
    mlp_ep = 3 if args.smoke else MLP_EP
    nb = 100 if args.smoke else 1000

    man, subs, data = load_db()
    Xraw, Fc, y, g = load_raw(subs)
    prior = np.load(OUT / 'depositioned_resnet_history_delta_oof.npz')
    assert np.array_equal(prior['y'], y) and np.array_equal(prior['groups'], g)
    hn = json.loads((OUT / 'depositioned_resnet_histnorm.json').read_text(encoding='utf-8'))
    hn_auc = {(r['stream'], r['fold'], r['resnet_seed']): r['test_auc'] for r in hn['runs'] if r['variant'] == WV}
    print(f'[start] rows={len(y)} subjects={len(subs)} device={dev} smoke={args.smoke} sseeds={sseeds}', flush=True)

    Xa, ya, ga, _ = arrange(subs, data, 0, 'independent')
    fold_of = {}
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Xa, ya, ga)):
        for s in np.unique(ga[te]): fold_of[s] = k
    fold = np.array([fold_of[s] for s in g]); assert np.array_equal(fold, prior['fold'])
    folds = []
    for k in range(5):
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        a, b = next(StratifiedGroupKFold(4, shuffle=True, random_state=k + 1).split(tr[:, None], y[tr], g[tr]))
        A, B = tr[a], tr[b]
        assert not (set(g[A]) & set(g[B])) and not (set(g[tr]) & set(g[te]))
        inner = [B[i] for _, i in StratifiedGroupKFold(4, shuffle=True, random_state=k + 1).split(B[:, None], y[B], g[B])]
        assert all(not (set(g[inner[i]]) & set(g[inner[j]])) for i in range(4) for j in range(i + 1, 4))
        folds.append((tr, te, A, B, inner))
    run_folds = [0] if args.smoke else list(range(5))
    yg = torch.tensor(y, dtype=torch.float32, device=dev)

    NEW = ['resnet_ctxspec', 'resnet_ctxspec_logit_delta', 'fusionC_ctxspec',
           'resnet_ctxemb', 'resnet_ctxemb_logit_delta', 'fusionC_ctxemb']
    REF = ['resnet_whiten', 'fusionH_whiten', 'fusionR_delta']
    result = {'meta': {'K': K, 'mature_index': MATURE, 'threshold': .5, 'rows': int(len(y)), 'subjects': len(subs),
                       'resnet_seeds_stream0': list(RESNET_SEEDS), 'resnet_seeds_other_streams': [0],
                       'stream_seeds': list(sseeds), 'device': dev, 'smoke': args.smoke,
                       'gpu': torch.cuda.get_device_name(0) if dev == 'cuda' else None,
                       'spec_bins': f'{NBIN} x 1 Hz (0-60 Hz)', 'ctxemb_head': 'MLP 385-64-1 on B, 4-fold cross-fit',
                       'mlp_max_ep': mlp_ep, 'mlp_patience': MLP_PAT},
              'runs': [], 'ctxemb_heads': [], 'checks': {}}
    checks = result['checks']; oof = {'y': y, 'groups': g, 'fold': fold}
    for sd in sseeds:
        rseeds = (RESNET_SEEDS[:1] if args.smoke else RESNET_SEEDS) if sd == 0 else RESNET_SEEDS[:1]
        order, t = stream_order(g, subs, sd)
        names = ['position_only__lr', 'position_only__hgb', 'hgb_delta32', WV, WV + '_logit_delta',
                 'resnet_ctxspec', 'resnet_ctxspec_logit_delta', 'resnet_ctxemb', 'resnet_ctxemb_logit_delta']
        S = {n: np.full(len(y), np.nan) for n in names}
        with threadpool_limits(limits=8):
            for k in run_folds:
                tr, te = folds[k][:2]; T = t[:, None].astype(float)
                for c in ('lr', 'hgb'):
                    S[f'position_only__{c}'][te] = make_clf(c).fit(T[tr], y[tr]).predict_proba(T[te])[:, 1]
                med = np.nanmedian(np.where(np.isfinite(Fc[tr]), Fc[tr], np.nan), 0); med = np.where(np.isfinite(med), med, 0.)
                Ff = np.where(np.isfinite(Fc), Fc, med[None, :])
                Z = StandardScaler().fit(Ff[tr]).transform(Ff); ZD = np.hstack([Z, hist_delta(Z, order)])
                S['hgb_delta32'][te] = make_clf('hgb').fit(ZD[tr], y[tr]).predict_proba(ZD[te])[:, 1]
        ev = np.isfinite(S['hgb_delta32'])
        for n in ('position_only__lr', 'position_only__hgb', 'hgb_delta32'):
            dmax = float(np.abs(S[n][ev] - prior[f'stream{sd}__{n}'][ev]).max())
            assert dmax < 1e-9, (n, dmax)
            checks[f'baseline_reproduced_stream{sd}_{n}'] = dmax
        S['fusionR_delta'] = np.where(ev, prior[f'stream{sd}__fusionR_delta'], np.nan)
        print(f'[stream {sd}] baselines reproduced (hgb_delta32/position); fusionR_delta loaded {el()}', flush=True)

        for k in run_folds:
            tr, te, A, B, inner = folds[k]
            t1 = time.perf_counter(); fb = fallback(Xraw, tr, WV)
            Xin = build_all(Xraw, order, WV, fb)
            G, eps = spec_stats(Xraw, tr)
            Sraw = np.empty((len(y), NBIN))
            for loc in order: Sraw[loc] = spec_subject(Xraw[loc], G, eps)
            smu, ssd = Sraw[tr].mean(0), Sraw[tr].std(0) + 1e-8
            Sin = ((Sraw - smu) / ssd).astype(np.float32)
            Xg = torch.from_numpy(Xin).to(dev); Sg = torch.from_numpy(Sin).to(dev)
            print(f'[stream {sd} fold {k + 1}/5] inputs built ({time.perf_counter() - t1:.1f}s) spec d std(raw)={Sraw[tr].std():.3f} {el()}', flush=True)
            acc = {n: [] for n in ('wl', 'wd', 'sl', 'sd', 'el', 'ed')}
            for rs in rseeds:
                seed = 1000 * rs + k + 1
                # ---- stage-1 whiten (histnorm code path) ----
                tag = f's{sd} whiten rs{rs} fold{k + 1}/5'
                net, info = train_resnet(Xg, yg, y, A, B, seed, tag, max_ep)
                E, L = embed_c(net, Xg)
                Lp = predict(net, Xg); assert np.abs(L - Lp).max() < 1e-3
                prev = hn_auc.get((sd, k, rs)); tauc = float(roc_auc_score(y[te], L[te]))
                result['runs'].append({'stream': sd, 'variant': WV, 'fold': k, 'resnet_seed': rs, **info, 'test_auc': tauc,
                                       'histnorm_test_auc': prev})
                print(f'[resnet] {tag} best_ep={info["best_epoch"]}/{info["epochs_run"]} testAUC={tauc:.4f} '
                      f'(histnorm {prev if prev is None else round(prev, 4)}) {el()}', flush=True)
                hW, muW, sdW, pW = lr_delta_head(L, B, te, order, y)
                acc['wl'].append(L[te]); acc['wd'].append(pW)
                # ---- ctxemb (B-only, 4-fold cross-fit) ----
                esc = StandardScaler().fit(E[B]); Es = esc.transform(E)
                FE = np.zeros((len(y), 3 * Es.shape[1] + 1))
                for loc in order: FE[loc] = emb_feats(Es[loc])
                Le = np.full(len(y), np.nan); lt = []; heads = []
                for j in range(4):
                    H, V = inner[j], inner[(j + 1) % 4]
                    Tr = np.concatenate([inner[i] for i in range(4) if i not in (j, (j + 1) % 4)])
                    m, bv, be = train_mlp(FE, y, Tr, V, seed * 10 + j, dev, mlp_ep)
                    heads.append(m); Le[H] = mlp_logit(m, FE[H], dev); lt.append(mlp_logit(m, FE[te], dev))
                    result['ctxemb_heads'].append({'stream': sd, 'fold': k, 'resnet_seed': rs, 'inner': j, 'val_auc': float(bv),
                                                   'best_epoch': int(be), 'heldout_auc': float(roc_auc_score(y[H], Le[H]))})
                Le[te] = np.mean(lt, 0)
                hE, muE, sdE, pE = lr_delta_head(Le, B, te, order, y)
                acc['el'].append(Le[te]); acc['ed'].append(pE)
                print(f'[ctxemb] s{sd} rs{rs} fold{k + 1}/5 B-oofAUC={roc_auc_score(y[B], Le[B]):.4f} '
                      f'testAUC={roc_auc_score(y[te], Le[te]):.4f} heads_val={[round(h["val_auc"], 3) for h in result["ctxemb_heads"][-4:]]} {el()}', flush=True)
                # ---- ctxspec ----
                tag = f's{sd} ctxspec rs{rs} fold{k + 1}/5'
                cnet, cinfo = train_ctx(Xg, Sg, yg, y, A, B, seed, tag, max_ep)
                Lc = predict_ctx(cnet, Xg, Sg)
                result['runs'].append({'stream': sd, 'variant': 'resnet_ctxspec', 'fold': k, 'resnet_seed': rs, **cinfo,
                                       'test_auc': float(roc_auc_score(y[te], Lc[te])), 'B_auc': float(roc_auc_score(y[B], Lc[B]))})
                print(f'[resnet] {tag} best_ep={cinfo["best_epoch"]}/{cinfo["epochs_run"]} testAUC={result["runs"][-1]["test_auc"]:.4f} {el()}', flush=True)
                hC, muC, sdC, pC = lr_delta_head(Lc, B, te, order, y)
                acc['sl'].append(Lc[te]); acc['sd'].append(pC)
                # ---- causality / isolation checks ----
                if rs == rseeds[0]:
                    for s0 in np.unique(g[te])[:2]:
                        loc = order[list(subs).index(s0)]; cut = len(loc) // 2; pos = np.searchsorted(te, loc)
                        raw2 = Xraw[loc].astype(np.float64).copy(); raw2[cut:] = raw2[cut:] * 50. + 1.

                        def inputs(xr):
                            xi = build_subject(xr, WV, fb); si = ((spec_subject(xr, G, eps) - smu) / ssd).astype(np.float32)
                            return xi, si
                        xa, sa = inputs(Xraw[loc]); xb2, sb2 = inputs(raw2)
                        assert np.array_equal(xa, Xin[loc]) and np.array_equal(sa, Sin[loc])

                        def sc_spec(xi, si):
                            l = predict_ctx(cnet, torch.from_numpy(xi).to(dev), torch.from_numpy(si).to(dev))
                            return l, delta_prob(hC, muC, sdC, l)

                        def sc_emb(xi):
                            e, _ = embed_c(net, torch.from_numpy(xi).to(dev)); fe = emb_feats(esc.transform(e))
                            l = np.mean([mlp_logit(m, fe, dev) for m in heads], 0)
                            return l, delta_prob(hE, muE, sdE, l)
                        assert_causal('ctxspec', (xa, sa), (xb2, sb2), pC[pos], sc_spec(xa, sa), sc_spec(xb2, sb2), cut,
                                      checks, f'causal_stream{sd}_ctxspec_fold{k}_{s0}')
                        assert_causal('ctxemb', (xa,), (xb2,), pE[pos], sc_emb(xa), sc_emb(xb2), cut,
                                      checks, f'causal_stream{sd}_ctxemb_fold{k}_{s0}')
                    print(f'[check] s{sd} fold{k + 1}: ctxspec/ctxemb isolation + prefix invariance ok {el()}', flush=True)
                del net, cnet
            S[WV][te] = sig(np.mean(acc['wl'], 0)); S[WV + '_logit_delta'][te] = np.mean(acc['wd'], 0)
            S['resnet_ctxemb'][te] = sig(np.mean(acc['el'], 0)); S['resnet_ctxemb_logit_delta'][te] = np.mean(acc['ed'], 0)
            S['resnet_ctxspec'][te] = sig(np.mean(acc['sl'], 0)); S['resnet_ctxspec_logit_delta'][te] = np.mean(acc['sd'], 0)
            del Xg, Sg, Xin
            if dev == 'cuda': torch.cuda.empty_cache()
        S['fusionH_whiten'] = sig((logit(S['hgb_delta32']) + logit(S[WV + '_logit_delta'])) / 2)
        for v in ('ctxspec', 'ctxemb'):
            S[f'fusionC_{v}'] = sig((logit(S['hgb_delta32']) + logit(S[f'resnet_{v}_logit_delta'])) / 2)

        M = {'all': ev, 'mature': ev & (t >= MATURE)}
        R = result[f'stream{sd}'] = {'resnet_seeds': list(rseeds), 'methods': {}, 'bootstrap': {}, 'subjects': {}}
        for n, p in S.items():
            oof[f'stream{sd}__{n}'] = p
            r = R['methods'][n] = {s: metrics(y, g, p, M[s]) for s in M}
            ba = np.array([q for s in np.unique(g[ev]) if (q := metrics(y, g, p, ev & (g == s))['subject_macro_bACC']) is not None])
            R['subjects'][n] = {'n_ge_0.9': int((ba >= .9).sum()), 'max': float(ba.max()), 'n': int(len(ba))}
            print(f'[result stream {sd}] {n:28s} ' + ' | '.join(
                f'{s}: AUC={r[s]["AUC"]:.4f} mac={r[s]["subject_macro_bACC"]:.4f} mic={r[s]["micro_bACC"]:.4f}' for s in M) +
                f' | subj>=.9={R["subjects"][n]["n_ge_0.9"]} max={R["subjects"][n]["max"]:.3f}', flush=True)
        if not args.smoke:
            for n in (WV, 'fusionH_whiten'):
                R.setdefault('vs_histnorm_json', {})[n] = {'AUC_here': R['methods'][n]['all']['AUC'],
                                                           'AUC_histnorm': hn[f'stream{sd}']['methods'][n]['all']['AUC']}
            print(f'[repro stream {sd}] {R["vs_histnorm_json"]}', flush=True)
        for c in ('lr', 'hgb'):
            a = R['methods'][f'position_only__{c}']['all']['AUC']
            R.setdefault('sanity', {})[f'position_only__{c}'] = {'AUC': a, 'ok': abs(a - .5) < .03}
            print(f'[sanity stream {sd}] position_only__{c} AUC={a:.4f} -> {"OK" if abs(a - .5) < .03 else "FLAG"}', flush=True)
            if not args.smoke: assert abs(a - .5) < .03
        for a_ in NEW:
            for b_ in REF:
                for seg in M:
                    bb = R['bootstrap'][f'{a_}__vs__{b_}__{seg}'] = boot(S[a_], S[b_], y, g, M[seg], nb)
                    print(f'[boot stream {sd}] {a_:27s} vs {b_:14s} {seg:6s} '
                          f'dAUC={bb["dAUC"]["mean"]:+.4f}[{bb["dAUC"]["p2.5"]:+.4f},{bb["dAUC"]["p97.5"]:+.4f}] '
                          f'dmac={bb["dmacro_bACC"]["mean"]:+.4f}[{bb["dmacro_bACC"]["p2.5"]:+.4f},{bb["dmacro_bACC"]["p97.5"]:+.4f}] '
                          f'dmic={bb["dmicro_bACC"]["mean"]:+.4f}[{bb["dmicro_bACC"]["p2.5"]:+.4f},{bb["dmicro_bACC"]["p97.5"]:+.4f}]',
                          flush=True)
        (OUT / (stem + '.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    result['seconds'] = time.perf_counter() - T0
    (OUT / (stem + '.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    np.savez_compressed(OUT / (stem + '_oof.npz'), **oof)
    print(f'[done] {el()} -> {stem}.json', flush=True)
    sys.stdout = tee.stream; tee.f.close()


if __name__ == '__main__':
    main()
