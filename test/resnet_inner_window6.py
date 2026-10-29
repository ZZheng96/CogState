"""ResNet1d / Hybrid（ResNet 波形 + 86 维宽频特征 MLP）的内层 OOF，以及与 C4+宽频LR 的 z-logit 等权融合。

严格 6 秒单窗：逐窗去均值/除 std/clip±10；86 维特征只来自当前窗，训练折内标准化；无窗口位置/时间表/测试受试者校准。
外层 StratifiedGroupKFold(5, shuffle, rs=0)（= seed_noise_band fold_id）；内层 StratifiedGroupKFold(4, shuffle, rs=fold)
（与 merge_best_window6 相同的内层划分）。
深度模型：
  内层模型 j：在 inner-train 上训练，早停用 inner-train 内 SGKF(5, rs=100*fold+j) 首个划分的验证受试者
             （与 merge_best 中 hand_mlp 的内层口径一致），预测 inner-val -> 内层 OOF logit。
  外层模型：在完整外层训练折上训练固定 epoch 数 = 该 (模型, 种子, 折) 4 个内层模型 best_epoch 的中位数，预测测试折。
CPU 成员（C4 + bb_lr）的内层 OOF 用 merge_best_window6.task 原样重算（确定性），外层分数复用 merge_best_window6_oof.npz。
"""
import hashlib, json, sys, threading, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'test'
DB = ROOT / 'database'
sys.path[:0] = [str(OUT)]
from raw_waveform_window6 import ResNet1d, augment, normalize, WIN, CROP, N_TTA, BS, MAX_EP, PATIENCE, LR, WD  # noqa: E402
from merge_best_window6 import task, youden_thr, subj_auc, to_logit  # noqa: E402

RES_PATH = OUT / 'resnet_inner_window6_results.json'
OOF_PATH = OUT / 'resnet_inner_window6_oof.npz'
LOG_PATH = OUT / 'resnet_inner_window6.log'
SEEDS = (0, 1, 2)
MIN_SEEDS = 2
TIME_LIMIT_S = 150 * 60  # 第 3 个种子仅在预计总时长不超过此值时运行
N_INNER, N_JOBS, N_BOOT, BOOT_SEED = 4, 10, 2000, 20261029
KINDS = ('resnet', 'hybrid')
T0 = time.perf_counter()
_LOG = open(LOG_PATH, 'w', encoding='utf-8')


def log(msg, console=True):
    line = f'[{time.perf_counter() - T0:7.1f}s] {msg}'
    if console:
        print(line, flush=True)
    _LOG.write(line + '\n'); _LOG.flush()


class Hybrid(nn.Module):
    """ResNet1d 波形主干（128 维池化） + 86 维特征 MLP（32 维），拼接后线性分类。"""
    def __init__(self, n_feat):
        super().__init__()
        r = ResNet1d()
        self.stem, self.blocks = r.stem, r.blocks
        self.pool = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.mlp = nn.Sequential(nn.Dropout(.1), nn.Linear(n_feat, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(.3),
                                 nn.Linear(64, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(.3), nn.Linear(128 + 32, 1))

    def forward(self, x, f):
        return self.head(torch.cat([self.pool(self.blocks(self.stem(x))), self.mlp(f)], 1))


class Wrap(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = ResNet1d()

    def forward(self, x, f):
        return self.net(x)


@torch.no_grad()
def predict_logit(net, Xg, Fg, ids):
    """当前 6 s 窗内 N_TTA 个等距 4 s 裁剪的平均 logit（特征分支对所有裁剪共用同窗特征）。"""
    net.eval(); out = []
    offs = np.linspace(0, WIN - CROP, N_TTA).round().astype(int)
    for b in range(0, len(ids), 512):
        it = torch.as_tensor(ids[b:b + 512], device=Xg.device)
        xb, fb = Xg[it], Fg[it]
        out.append(torch.stack([net(xb[:, None, o:o + CROP], fb).ravel() for o in offs]).mean(0).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def train(kind, Xg, yg, y, BB, tr, va, te, seed, tag, fixed_ep=None):
    """va 不为 None：按 va AUC 早停（patience 8，最多 50 ep）；否则在 tr 上训练 fixed_ep 个 epoch。返回 te 的 logit。"""
    tick = time.perf_counter()
    dev = Xg.device
    mu, sd = BB[tr].mean(0), BB[tr].std(0)
    sd = np.where(sd > 1e-12, sd, 1.)
    Fg = torch.from_numpy(np.clip((BB - mu) / sd, -10, 10).astype(np.float32)).to(dev)  # 仅训练集统计量
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    net = (Hybrid(BB.shape[1]) if kind == 'hybrid' else Wrap()).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    crit = nn.BCEWithLogitsLoss()
    n_ep = MAX_EP if fixed_ep is None else fixed_ep
    best, state, stale, best_ep = -1., None, 0, 0
    for ep in range(1, n_ep + 1):
        net.train(); perm = rng.permutation(tr); tot = 0.
        for b in range(0, len(perm), BS):
            ids = perm[b:b + BS]
            if len(ids) < 2:
                continue
            idt = torch.as_tensor(ids, device=dev)
            loss = crit(net(augment(Xg[idt], gen), Fg[idt]).ravel(), yg[idt])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.); opt.step()
            tot += loss.item() * len(ids)
        msg = f'    [{tag}] ep{ep:02d}/{n_ep} loss={tot / len(tr):.4f}'
        if va is not None:
            auc = roc_auc_score(y[va], predict_logit(net, Xg, Fg, va))
            if auc > best + 1e-4:
                best, state, stale, best_ep = auc, {k: v.detach().clone() for k, v in net.state_dict().items()}, 0, ep
            else:
                stale += 1
            msg += f' valAUC={auc:.4f} best={best:.4f}@{best_ep}'
        log(msg + f' t={time.perf_counter() - tick:.0f}s', console=(ep % 5 == 0 or ep == 1))
        if va is not None and stale >= PATIENCE:
            break
    if va is not None:
        net.load_state_dict(state)
    else:
        best_ep = n_ep
    return predict_logit(net, Xg, Fg, te), {'best_epoch': int(best_ep), 'epochs_run': int(ep),
                                           'best_val_auc': None if va is None else float(best),
                                           'seconds': float(time.perf_counter() - tick)}


def main():
    if RES_PATH.exists() or OOF_PATH.exists():
        raise FileExistsError('refusing to overwrite resnet_inner_window6 artifacts')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(f'device={device} torch={torch.__version__}')

    # ---------- 数据（highdim 顺序）与 database 对齐（照搬 merge_best_window6） ----------
    hd = np.load(OUT / 'highdim_deep_features.npz')
    raw_h, y_h, g_h = hd['raw'], hd['y'].astype(int), hd['groups'].astype(str)
    sn = np.load(OUT / 'seed_noise_band_oof.npz'); fold_h = sn['fold_id'].astype(int)
    bbz = np.load(OUT / 'broadband_shape_features.npz')
    BB_h = bbz['X'].astype(np.float64)
    assert (bbz['y'] == y_h).all() and (bbz['groups'].astype(str) == g_h).all() and np.isfinite(BB_h).all()
    mb = np.load(OUT / 'merge_best_window6_oof.npz')
    assert (mb['fold_id'] == fold_h).all() and (mb['y'] == y_h).all()

    key = defaultdict(list)
    for i, r in enumerate(raw_h):
        key[(g_h[i], int(y_h[i]), hashlib.md5(r.tobytes()).hexdigest())].append(i)
    idx, fs, es, ys, gs = [], [], [], [], []
    for p in sorted((DB / 'windows').glob('*/*.npz')):
        d = np.load(p)
        fs.append(d['f']); es.append(d['e']); ys.append(d['label']); gs.append(d['subject'])
        for x, s, l in zip(d['x'], d['subject'], d['label']):
            idx.append(key[(str(s), int(l), hashlib.md5(x.astype(np.float32).tobytes()).hexdigest())].pop(0))
    idx = np.array(idx)
    F = np.vstack(fs).astype(np.float64); E = np.vstack(es).astype(np.float64)
    y = np.concatenate(ys).astype(int); g = np.concatenate(gs).astype(str)
    assert len(np.unique(idx)) == len(y_h) == len(y) and (y_h[idx] == y).all() and (g_h[idx] == g).all()
    inv = np.empty_like(idx); inv[idx] = np.arange(len(idx))
    V = np.load(OUT / 'features_v3.npz')['V'].astype(np.float64)
    W = np.load(OUT / 'b_warm05.npz')['W'].astype(np.float64)[idx]
    keep = F.std(0) > 1e-8
    feats = {'base_hgb': np.hstack([F[:, keep], E]), 'w05_lr': np.hstack([W[:, keep], E]),
             'svc_base': np.hstack([F[:, keep], E, V]), 'bb_lr': BB_h[idx]}
    feats['w05_hgb'] = feats['w05_lr']
    log(f'[align] md5 multiset map ok n={len(y)} subjects={len(np.unique(g))}')

    outer = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(F, y, g))
    fold_db = np.zeros(len(y), int)
    for k, (_, te) in enumerate(outer, 1):
        fold_db[te] = k
    assert (fold_h[idx] == fold_db).all()
    inner_db = {k: [(tr[a], tr[b]) for a, b in StratifiedGroupKFold(N_INNER, shuffle=True, random_state=k)
                    .split(tr[:, None], y[tr], g[tr])] for k, (tr, _) in enumerate(outer, 1)}
    # highdim 顺序的同一划分（受试者集合与 database 顺序一致）
    outer_h = [(np.where(fold_h != k)[0], np.where(fold_h == k)[0]) for k in range(1, 6)]
    inner_h = {}
    for k, (tr, _) in enumerate(outer_h, 1):
        inner_h[k] = [(tr[a], tr[b]) for a, b in StratifiedGroupKFold(N_INNER, shuffle=True, random_state=k)
                      .split(tr[:, None], y_h[tr], g_h[tr])]
        for (a, b), (a2, b2) in zip(inner_h[k], inner_db[k]):
            assert set(g_h[b]) == set(g[b2]) and len(b) == len(b2)
    log('[folds] outer = fold_id; inner = SGKF(4, rs=fold), subject sets identical in both orders')

    # ---------- CPU：C4 + bb_lr 的内层 OOF（与 merge_best 相同任务，后台线程） ----------
    cpu_members = ['base_hgb', 'w05_lr', 'w05_hgb', 'svc_base', 'bb_lr']
    jobs = [(n, k, f'inner{j}', a, b) for n in cpu_members for k in range(1, 6) for j, (a, b) in enumerate(inner_db[k])]
    cpu_out = {}

    def run_cpu():
        try:
            cpu_out['res'] = Parallel(n_jobs=N_JOBS, backend='loky', verbose=0)(
                delayed(task)(n, k, p, tr, te, feats[n], y, g) for n, k, p, tr, te in jobs)
        except Exception as e:  # 主线程 join 后重新抛出
            cpu_out['err'] = e
    log(f'[cpu] {len(jobs)} inner tasks on {N_JOBS} workers (background)')
    tcpu = time.perf_counter()
    th = threading.Thread(target=run_cpu); th.start()

    # ---------- GPU：ResNet / Hybrid 内层 OOF + 外层测试 ----------
    X, n_flat = normalize(raw_h)
    Xg = torch.from_numpy(X).to(device); yg = torch.tensor(y_h, dtype=torch.float32, device=device)
    log(f'[gpu] windows={len(y_h)} flat={n_flat} feat_dim={BB_h.shape[1]}')
    inner_s = {kd: {} for kd in KINDS}   # (seed) -> {fold: array highdim, 训练折位置有效}
    outer_s = {kd: {} for kd in KINDS}   # (seed) -> array highdim
    runs = []
    seeds_done = []
    tg = time.perf_counter()
    for si, seed in enumerate(SEEDS):
        if si >= MIN_SEEDS:
            per_seed = (time.perf_counter() - tg) / si
            proj = time.perf_counter() - T0 + per_seed
            log(f'[plan] per-seed {per_seed:.0f}s, projected total with seed{seed}: {proj:.0f}s (limit {TIME_LIMIT_S}s)')
            if proj > TIME_LIMIT_S:
                log(f'[plan] skip seed{seed}')
                break
        ts = time.perf_counter()
        for kd in KINDS:
            o = np.full(len(y_h), np.nan); ins = {}
            for k, (tr, te) in enumerate(outer_h, 1):
                tf = time.perf_counter()
                sc = np.full(len(y_h), np.nan); beps = []
                for j, (a, b) in enumerate(inner_h[k]):
                    it, iv = next(StratifiedGroupKFold(5, shuffle=True, random_state=100 * k + j)
                                  .split(a[:, None], y_h[a], g_h[a]))
                    tag = f'{kd} s{seed} f{k} inner{j}'
                    sc[b], info = train(kd, Xg, yg, y_h, BB_h, a[it], a[iv], b, 10000 * seed + 100 * k + j + 1, tag)
                    beps.append(info['best_epoch'])
                    runs.append({'kind': kd, 'seed': seed, 'fold': k, 'part': f'inner{j}', **info})
                    log(f'  [{tag}] best_ep={info["best_epoch"]}/{info["epochs_run"]} valAUC={info["best_val_auc"]:.4f} '
                        f'innerOOF_AUC={roc_auc_score(y_h[b], sc[b]):.4f} {info["seconds"]:.0f}s total={time.perf_counter() - T0:.0f}s')
                fixed = max(5, int(round(np.median(beps))))
                tag = f'{kd} s{seed} f{k} outer'
                o[te], info = train(kd, Xg, yg, y_h, BB_h, tr, None, te, 1000 * seed + k, tag, fixed_ep=fixed)
                runs.append({'kind': kd, 'seed': seed, 'fold': k, 'part': 'outer', 'inner_best_epochs': beps, **info})
                ins[k] = sc
                log(f'[fold] {kd} seed{seed} fold{k}/5 innerOOF_AUC={roc_auc_score(y_h[tr], sc[tr]):.4f} '
                    f'outer_ep={fixed} testAUC={roc_auc_score(y_h[te], o[te]):.4f} fold_t={time.perf_counter() - tf:.0f}s '
                    f'total={time.perf_counter() - T0:.0f}s')
            outer_s[kd][seed], inner_s[kd][seed] = o, ins
            log(f'[seed] {kd} seed{seed} outer OOF AUC={roc_auc_score(y_h, o):.4f} '
                f'fold={[round(roc_auc_score(y_h[te], o[te]), 4) for _, te in outer_h]}')
        seeds_done.append(seed)
        log(f'[seed] seed{seed} done in {time.perf_counter() - ts:.0f}s')
    log(f'[gpu] done {time.perf_counter() - tg:.0f}s, waiting CPU')
    th.join()
    if 'err' in cpu_out:
        raise cpu_out['err']
    log(f'[cpu] done ({time.perf_counter() - tcpu:.0f}s since start)')

    # ---------- 汇总分数（highdim 顺序） ----------
    C4 = ['base_hgb', 'w05_lr', 'w05_hgb', 'svc_base']
    members = C4 + ['bb_lr'] + list(KINDS)
    outer_l = {m: mb[f'member__{m}'].astype(np.float64) for m in C4 + ['bb_lr']}
    inner_l = {m: {k: np.full(len(y_h), np.nan) for k in range(1, 6)} for m in C4 + ['bb_lr']}
    for name, k, part, te, s in cpu_out['res']:
        inner_l[name][k][idx[te]] = s
    for kd in KINDS:  # 种子平均 logit
        outer_l[kd] = np.mean([outer_s[kd][s] for s in seeds_done], 0)
        inner_l[kd] = {k: np.mean([inner_s[kd][s][k] for s in seeds_done], 0) for k in range(1, 6)}
    saved_z = json.loads((OUT / 'merge_best_window6_results.json').read_text(encoding='utf-8'))['z_stats']

    Zte = {m: np.zeros(len(y_h)) for m in members}
    Zin = {m: {} for m in members}
    zstats, repro_dev = {}, {}
    for m in members:
        for k, (tr, te) in enumerate(outer_h, 1):
            v = inner_l[m][k][tr]
            assert np.isfinite(v).all() and np.isfinite(outer_l[m][te]).all(), (m, k)
            mu_, sd_ = v.mean(), max(v.std(), 1e-8)
            Zin[m][k] = (v - mu_) / sd_
            Zte[m][te] = (outer_l[m][te] - mu_) / sd_
            zstats.setdefault(m, []).append({'fold': k, 'mu': float(mu_), 'sd': float(sd_),
                                             'inner_oof_auc': float(roc_auc_score(y_h[tr], v))})
        if m in saved_z:
            repro_dev[m] = float(max(max(abs(a['mu'] - b['mu']), abs(a['sd'] - b['sd'])) for a, b in zip(zstats[m], saved_z[m])))
            dz = float(np.abs(Zte[m] - mb[f'member_z__{m}']).max())
            log(f'[repro] {m:9s} max|d mu/sd| vs merge_best = {repro_dev[m]:.2e}, max|dZ_test| = {dz:.2e}')
            repro_dev[m] = {'max_abs_diff_mu_sd': repro_dev[m], 'max_abs_diff_test_z': dz}

    # ---------- 成员指标 ----------
    def thr_metrics(s_te_all, ztr_by_fold):
        pred = np.zeros(len(y_h), bool); rows = []
        for k, (tr, te) in enumerate(outer_h, 1):
            thr = youden_thr(y_h[tr], ztr_by_fold[k])
            pred[te] = s_te_all[te] >= thr
            rows.append({'fold': k, 'auc': float(roc_auc_score(y_h[te], s_te_all[te])),
                         'bacc_at0': float(balanced_accuracy_score(y_h[te], s_te_all[te] >= 0)),
                         'bacc_train_thr': float(balanced_accuracy_score(y_h[te], pred[te])), 'train_thr': thr})
        return pred, rows

    member_info = {}
    for m in members:
        pred, rows = thr_metrics(Zte[m], Zin[m])
        member_info[m] = {'oof_auc_raw_logit': float(roc_auc_score(y_h, outer_l[m])),
                          'oof_auc_z': float(roc_auc_score(y_h, Zte[m])),
                          'bacc_at0_raw_logit': float(balanced_accuracy_score(y_h, outer_l[m] >= 0)),
                          'bacc_at0_z': float(balanced_accuracy_score(y_h, Zte[m] >= 0)),
                          'bacc_train_thr_z': float(balanced_accuracy_score(y_h, pred)),
                          'fold_auc': [r['auc'] for r in rows],
                          'mean_inner_oof_auc': float(np.mean([z['inner_oof_auc'] for z in zstats[m]])),
                          'per_fold': rows}
        if m in KINDS:
            member_info[m]['per_seed_outer_auc'] = {str(s): float(roc_auc_score(y_h, outer_s[m][s])) for s in seeds_done}
            member_info[m]['per_seed_mean_inner_oof_auc'] = {
                str(s): float(np.mean([roc_auc_score(y_h[tr], inner_s[m][s][k][tr]) for k, (tr, _) in enumerate(outer_h, 1)]))
                for s in seeds_done}
            member_info[m]['per_seed_outer_auc_mean'] = float(np.mean(list(member_info[m]['per_seed_outer_auc'].values())))
        log(f'[member] {m:9s} AUC={member_info[m]["oof_auc_raw_logit"]:.4f} (z {member_info[m]["oof_auc_z"]:.4f}) '
            f'bACC@0(z)={member_info[m]["bacc_at0_z"]:.4f} bACC@thr={member_info[m]["bacc_train_thr_z"]:.4f} '
            f'innerOOF={member_info[m]["mean_inner_oof_auc"]:.4f}')

    # ---------- 组合（z-logit 等权，与 merge_best 相同） ----------
    combos = {'C4_bb_resnetz_eq': C4 + ['bb_lr', 'resnet'],
              'C4_bb_hybrid_eq': C4 + ['bb_lr', 'hybrid'],
              'C4_bb_resnetz_hybrid_eq': C4 + ['bb_lr', 'resnet', 'hybrid'],
              'C4_bb_eq': C4 + ['bb_lr']}
    scores, info = {}, {}
    for cname, ms in combos.items():
        s = np.mean([Zte[m] for m in ms], 0)
        zi = {k: np.mean([Zin[m][k] for m in ms], 0) for k in range(1, 6)}
        pred, rows = thr_metrics(s, zi)
        sa = subj_auc(y_h, s, g_h)
        scores[cname] = s
        info[cname] = {'members': ms, 'weighting': 'eq', 'oof_auc': float(roc_auc_score(y_h, s)),
                       'bacc_at0': float(balanced_accuracy_score(y_h, s >= 0)),
                       'bacc_train_thr': float(balanced_accuracy_score(y_h, pred)),
                       'fold_auc': [r['auc'] for r in rows], 'subject_auc_median': float(np.median(sa)),
                       'subject_auc_lt_0.5_frac': float(np.mean(sa < .5)), 'per_fold': rows}
        log(f'[combo] {cname:24s} AUC={info[cname]["oof_auc"]:.4f} bACC@0={info[cname]["bacc_at0"]:.4f} '
            f'bACC@thr={info[cname]["bacc_train_thr"]:.4f} fold={[round(a, 4) for a in info[cname]["fold_auc"]]}')
    ref = mb['combo__C4_bb_resnet_eq'].astype(np.float64)
    scores['REF_C4_bb_resnet_eq_merge_best'] = ref
    info['REF_C4_bb_resnet_eq_merge_best'] = {'oof_auc': float(roc_auc_score(y_h, ref)),
                                              'bacc_at0': float(balanced_accuracy_score(y_h, ref >= 0)),
                                              'fold_auc': [float(roc_auc_score(y_h[te], ref[te])) for _, te in outer_h],
                                              'note': 'merge_best_window6_oof.npz combo__C4_bb_resnet_eq (ResNet raw logit, 无 z)'}
    log(f'[ref]   REF_C4_bb_resnet_eq AUC={info["REF_C4_bb_resnet_eq_merge_best"]["oof_auc"]:.4f}')

    # ---------- 受试者级 bootstrap ΔAUC（共享抽样） ----------
    tb = time.perf_counter()
    rng = np.random.default_rng(BOOT_SEED)
    subs = np.unique(g_h); sidx = {u: np.where(g_h == u)[0] for u in subs}
    bauc = {c: np.empty(N_BOOT) for c in scores}
    for i in range(N_BOOT):
        b = np.concatenate([sidx[u] for u in rng.choice(subs, len(subs))])
        for c, s in scores.items():
            bauc[c][i] = roc_auc_score(y_h[b], s[b])
        if (i + 1) % 250 == 0:
            log(f'[boot] {i + 1}/{N_BOOT} {time.perf_counter() - tb:.0f}s')
    rn = 'REF_C4_bb_resnet_eq_merge_best'
    for c in scores:
        info[c]['auc_ci95'] = [float(np.percentile(bauc[c], 2.5)), float(np.percentile(bauc[c], 97.5))]
        if c != rn:
            dl = bauc[c] - bauc[rn]
            info[c]['delta_vs_REF'] = {'point': float(roc_auc_score(y_h, scores[c]) - roc_auc_score(y_h, ref)),
                                       'ci95': [float(np.percentile(dl, 2.5)), float(np.percentile(dl, 97.5))],
                                       'p_delta_gt0': float(np.mean(dl > 0))}
            dd = info[c]['delta_vs_REF']
            log(f'[delta] {c:24s} vs REF {dd["point"]:+.4f} CI=[{dd["ci95"][0]:+.4f},{dd["ci95"][1]:+.4f}] P(>0)={dd["p_delta_gt0"]:.3f}')

    # ---------- 保存 ----------
    out = {'y': y_h, 'groups': g_h, 'fold_id': fold_h}
    for kd in KINDS:
        out[f'member__{kd}'] = outer_l[kd]
        out[f'member_z__{kd}'] = Zte[kd]
        for s in seeds_done:
            out[f'member_seed{s}__{kd}'] = outer_s[kd][s]
        for k in range(1, 6):
            out[f'inner_oof_fold{k}__{kd}'] = inner_l[kd][k]  # 外层第 k 折训练窗位置有效，其余 NaN
    out |= {f'combo__{c}': s for c, s in scores.items()}
    np.savez_compressed(OOF_PATH, **out)
    elapsed = time.perf_counter() - T0
    result = {
        'protocol': {
            'test_subject_isolation': True, 'test_history': False, 'window_position_schedule_r_t': False,
            'test_subject_calibration': False, 'per_window_only': True,
            'outer_cv': 'StratifiedGroupKFold(5, shuffle=True, random_state=0) by subject = seed_noise_band fold_id',
            'inner_cv': 'StratifiedGroupKFold(4, shuffle=True, random_state=outer_fold) on outer-train subjects (same as merge_best_window6)',
            'deep_inner_models': 'train on inner-train; early stopping (val AUC, patience 8, max 50 ep) on first split of '
                                 'SGKF(5, shuffle, rs=100*fold+inner) of inner-train subjects; predict inner-val -> inner OOF logit',
            'deep_outer_model': 'train on FULL outer-train fold for fixed epochs = median best_epoch of the 4 inner models '
                                '(min 5), no early stopping; predict test fold (deviation from raw_waveform_window6, which '
                                'trained on 3/4 of outer-train with early stopping on the remaining 1/4)',
            'preprocessing': 'per window demean, /std, clip +-10; augmentation/TTA identical to raw_waveform_window6',
            'hybrid': 'ResNet1d trunk (128-d pooled) + MLP on 86-d broadband_shape_features (standardized with the training '
                      'set of each model, clip +-10; Dropout.1-64-BN-ReLU-Dropout.3-32-ReLU), concat 160 -> Dropout.3 -> Linear',
            'seed_aggregation': 'mean logit over seeds for both inner OOF and outer test',
            'cpu_members': 'C4 + bb_lr inner OOF recomputed with merge_best_window6.task (deterministic); outer scores reused '
                           'from merge_best_window6_oof.npz member__*',
            'fusion': 'per member per outer fold z = (logit - mu)/sd with mu/sd from inner OOF on outer-train; equal-weight mean',
            'threshold': 'bacc_at0: fixed 0; bacc_train_thr: Youden threshold on fused inner-OOF z of outer-train',
            'bootstrap': f'subject-level, {N_BOOT} draws, seed {BOOT_SEED}, shared draws, delta = combo - REF',
            'nothing_selected_on_test_folds': True},
        'seeds_run': seeds_done,
        'cpu_reproduction_vs_merge_best': repro_dev,
        'z_stats': zstats, 'members': member_info, 'combos': info,
        'baseline_raw_waveform_window6_resnet1d': {'per_seed_auc_mean': 0.6046, 'seed_mean_auc': 0.6191},
        'runs': runs,
        'meta': {'elapsed_s': elapsed, 'device': device,
                 'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None, 'torch': str(torch.__version__),
                 'n_windows': int(len(y_h)), 'n_subjects': int(len(subs))}}
    RES_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    log(f'[done] {elapsed:.0f}s -> {RES_PATH.name}, {OOF_PATH.name}, {LOG_PATH.name}')


if __name__ == '__main__':
    main()
