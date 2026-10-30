"""实验 A（诊断，不可部署）：同被试按状态块留一的个体上限。

每个被试只用本人 canonical 非重叠 6 秒窗；状态块 = 本人时间顺序上连续同标签 trial 段
（由 dense_window_protocol_cache 的 canonical 行恢复 trial id，与 causal_kernel_state_mapping 的标签切换口径一致）。
留一块：测试块的窗只来自该块，训练窗来自其余块；canonical 窗互不重叠，因此训练窗与测试块无时间重叠。
模型固定超参：LR(StandardScaler, C=1, balanced) / HGB(150, depth3, lr0.06, balanced) on 16 维 Welch 谱特征；
GPU 分组卷积并行训练的小型 1D CNN（每个 (被试, 留出块) 一个独立模型）。阈值固定 0.5（类平衡训练）。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed
from scipy.signal import welch
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(OUT)]
from causal_kernel_state_mapping import features as spec16  # noqa: E402  16 维：7 log 带功率+7 相对功率+谱熵+平坦度

T0 = time.perf_counter()
LOGF = None


def log(msg):
    s = f'[{time.strftime("%H:%M:%S")} +{time.perf_counter() - T0:7.1f}s] {msg}'
    print(s, flush=True)
    if LOGF:
        LOGF.write(s + '\n'); LOGF.flush()


def load(smoke):
    d = np.load(OUT / 'highdim_deep_features.npz')
    raw, y, g = d['raw'], d['y'].astype(int), d['groups'].astype(str)
    assert raw.shape[1] <= 1500 and raw.shape[1] == 1500
    c = np.load(OUT / 'dense_window_protocol_cache.npz')
    ci = np.where(c['canon'])[0]
    assert np.array_equal(c['label'][ci], y) and np.array_equal(c['subject'][ci], g)
    trial = c['trial'][ci]
    block = np.full(len(y), -1)
    nb = 0
    for s in np.unique(g):
        ii = np.flatnonzero(g == s)
        assert (np.diff(ii) == 1).all()
        # trial id 在被试内应非递减（时间顺序）
        assert (np.diff(trial[ii]) >= 0).all(), s
        b = np.r_[0, np.cumsum(y[ii][1:] != y[ii][:-1])]
        block[ii] = nb + b
        nb += b[-1] + 1
    assert nb == 339 or smoke
    subs = np.unique(g)
    if smoke:
        keep = np.isin(g, subs[:4])
        raw, y, g, block, trial = raw[keep], y[keep], g[keep], block[keep], trial[keep]
    return raw, y, g, block, trial


def make_tasks(y, g, block):
    """每个 (被试, 留出块)：训练=本人其余块（需双类），测试=该块。"""
    tasks = []
    for s in np.unique(g):
        ii = np.flatnonzero(g == s)
        for b in np.unique(block[ii]):
            te = ii[block[ii] == b]
            tr = ii[block[ii] != b]
            assert not set(block[tr]) & set(block[te])
            if len(np.unique(y[tr])) < 2:
                continue
            tasks.append((s, int(b), tr, te))
    return tasks


def fit_tab(kind, Xtr, ytr, Xte):
    if kind == 'lr':
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1, class_weight='balanced', max_iter=3000))
    else:
        m = HistGradientBoostingClassifier(max_iter=150, max_depth=3, learning_rate=.06,
                                           class_weight='balanced', random_state=0)
    return m.fit(Xtr, ytr).predict_proba(Xte)[:, 1]


class GroupCNN(nn.Module):
    """M 个独立小 CNN 以分组卷积并行：输入 [B, M, L]，输出 [B, M] logit。"""
    def __init__(self, M, c1=8, c2=16, c3=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(M, M * c1, 25, stride=2, padding=12, groups=M), nn.BatchNorm1d(M * c1), nn.GELU(), nn.MaxPool1d(4),
            nn.Conv1d(M * c1, M * c2, 9, padding=4, groups=M), nn.BatchNorm1d(M * c2), nn.GELU(), nn.MaxPool1d(4),
            nn.Conv1d(M * c2, M * c3, 9, padding=4, groups=M), nn.BatchNorm1d(M * c3), nn.GELU(),
            nn.AdaptiveAvgPool1d(1))
        self.head = nn.Conv1d(M * c3, M, 1, groups=M)
        self.M = M

    def forward(self, x):
        return self.head(self.net(x)).squeeze(-1)


def run_cnn(raw, y, tasks, device, steps, batch, chunk, seed=0):
    out = {}
    for c0 in range(0, len(tasks), chunk):
        T = tasks[c0:c0 + chunk]; M = len(T)
        torch.manual_seed(seed); rng = np.random.default_rng(seed + c0)
        # 每模型用本人训练块 std 做幅值缩放（不看测试块），窗内去均值
        sd = np.array([raw[tr].std() for _, _, tr, _ in T], np.float32)
        model = GroupCNN(M).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
        pos = [tr[y[tr] == 1] for _, _, tr, _ in T]; neg = [tr[y[tr] == 0] for _, _, tr, _ in T]
        R = torch.from_numpy(raw).to(device)
        sdt = torch.from_numpy(sd).to(device)[None, :, None]
        yb = torch.from_numpy(np.r_[np.ones(batch // 2), np.zeros(batch // 2)].astype(np.float32)).to(device)
        lossf = nn.BCEWithLogitsLoss()
        model.train()
        for st in range(steps):
            idx = np.stack([np.r_[rng.choice(p, batch // 2), rng.choice(n, batch // 2)] for p, n in zip(pos, neg)], 1)
            x = R[torch.from_numpy(idx).to(device)]  # [B, M, L]
            x = (x - x.mean(-1, keepdim=True)) / sdt
            opt.zero_grad()
            loss = lossf(model(x), yb[:, None].expand(-1, M))
            loss.backward(); opt.step()
            if (st + 1) % 100 == 0:
                log(f'  [cnn] chunk {c0 // chunk + 1}/{(len(tasks) + chunk - 1) // chunk} M={M} step {st + 1}/{steps} loss={loss.item():.4f}')
        model.eval()
        L = max(len(te) for *_, te in T)
        pad = np.stack([np.r_[te, np.full(L - len(te), te[0])] for *_, te in T], 1)  # [L, M]
        probs = []
        with torch.no_grad():
            for i in range(0, L, 64):
                x = R[torch.from_numpy(pad[i:i + 64]).to(device)]
                x = (x - x.mean(-1, keepdim=True)) / sdt
                probs.append(torch.sigmoid(model(x)).cpu().numpy())
        P = np.concatenate(probs)
        for j, (s, b, tr, te) in enumerate(T):
            out[(s, b)] = P[:len(te), j]
        del model, opt, R
        torch.cuda.empty_cache()
    return out


def subj_metrics(y, p, m):
    yy, pp = y[m], p[m]
    ok = np.isfinite(pp)
    yy, pp = yy[ok], pp[ok]
    if len(np.unique(yy)) < 2:
        return None, None
    return float(balanced_accuracy_score(yy, pp >= .5)), float(roc_auc_score(yy, pp))


def quality(raw, y, F, g, s):
    m = g == s
    r = raw[m].astype(np.float64)
    f, P = welch(r, fs=250, nperseg=250, axis=1)
    hf = P[:, (f >= 52) & (f < 100)].sum(1) / (P[:, (f >= 1) & (f < 45)].sum(1) + 1e-20)
    dif = np.abs(np.diff(r, axis=1)) < 1e-9
    # 每窗最长零差分连续段（平台/零段）
    longest = np.array([np.max(np.diff(np.flatnonzero(np.r_[True, ~row, True])) - 1) for row in dif])
    sdw = r.std(1)
    med = np.median(sdw); mad = np.median(np.abs(sdw - med)) + 1e-12
    Fm, ym = F[m], y[m]
    d = []
    for k in range(F.shape[1]):
        a, b = Fm[ym == 0, k], Fm[ym == 1, k]
        sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)) + 1e-12
        d.append(abs(a.mean() - b.mean()) / sp)
    return {'amp_std_median': float(med), 'hf_noise_ratio_median': float(np.median(hf)),
            'flat_or_bad_frac': float(np.mean((longest >= 25) | (np.abs(sdw - med) / mad > 6) | (sdw < 1e-6))),
            'flat_frac': float(np.mean((longest >= 25) | (sdw < 1e-6))),
            'amp_outlier_frac': float(np.mean(np.abs(sdw - med) / mad > 6)),
            'max_abs_cohen_d': float(np.max(d)), 'argmax_cohen_d_feature': int(np.argmax(d))}


def main():
    global LOGF
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    stem = 'per_subject_ceiling' + ('_smoke' if args.smoke else '')
    paths = [OUT / f'{stem}.log', OUT / f'{stem}_results.json', OUT / f'{stem}_oof.npz']
    if not args.smoke and any(p.exists() for p in paths):
        raise FileExistsError('refusing to overwrite per_subject_ceiling artifacts')
    LOGF = paths[0].open('w', encoding='utf-8')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    raw, y, g, block, trial = load(args.smoke)
    F = spec16(raw)
    tasks = make_tasks(y, g, block)
    subs = np.unique(g)
    log(f'[data] windows={len(y)} subjects={len(subs)} blocks={len(np.unique(block))} '
        f'tasks(two-class train)={len(tasks)} device={device}')

    P = {k: np.full(len(y), np.nan) for k in ('lr', 'hgb', 'cnn')}
    for kind in ('lr', 'hgb'):
        res = Parallel(n_jobs=12, backend='loky')(delayed(fit_tab)(kind, F[tr], y[tr], F[te]) for *_, tr, te in tasks)
        for (s, b, tr, te), p in zip(tasks, res):
            P[kind][te] = p
        log(f'[{kind}] done {len(tasks)} block-folds')
    steps = 60 if args.smoke else 400
    cnn = run_cnn(raw.astype(np.float32), y, tasks, device, steps=steps, batch=32, chunk=120)
    for s, b, tr, te in tasks:
        P['cnn'][te] = cnn[(s, b)]
    log('[cnn] done')
    P['mean3'] = np.nanmean(np.stack([P['lr'], P['hgb'], P['cnn']]), 0)

    mb = np.load(OUT / 'merge_best_window6_oof.npz')
    hd_g = np.load(OUT / 'highdim_deep_features.npz')['groups'].astype(str)
    mbs, mby = mb['combo__C4_bb_resnet_eq'], mb['y'].astype(int)
    assert np.array_equal(mb['groups'].astype(str), hd_g)

    rows = []
    for s in subs:
        m = g == s
        q = quality(raw, y, F, g, s)
        mm = hd_g == s
        auc_mb = float(roc_auc_score(mby[mm], mbs[mm]))
        r = {'subject': s, 'n_blocks': int(len(np.unique(block[m]))), 'n_windows': int(m.sum()),
             'n_scored_windows': int(np.isfinite(P['lr'][m]).sum()),
             'merge_best_auc': auc_mb, 'direction_flip': int(auc_mb < .5), **q}
        for k in P:
            r[f'{k}_bacc'], r[f'{k}_auc'] = subj_metrics(y, P[k], m)
        rows.append(r)

    summary = {}
    for k in P:
        v = np.array([r[f'{k}_bacc'] for r in rows if r[f'{k}_bacc'] is not None])
        a = np.array([r[f'{k}_auc'] for r in rows if r[f'{k}_auc'] is not None])
        ok = np.isfinite(P[k])
        summary[k] = {'n_subjects': int(len(v)), 'bacc_median': float(np.median(v)), 'bacc_q1': float(np.quantile(v, .25)),
                      'bacc_q3': float(np.quantile(v, .75)), 'bacc_mean': float(v.mean()),
                      'frac_bacc_ge_0.9': float(np.mean(v >= .9)), 'frac_bacc_ge_0.8': float(np.mean(v >= .8)),
                      'frac_bacc_lt_0.6': float(np.mean(v < .6)),
                      'auc_median': float(np.median(a)), 'auc_q1': float(np.quantile(a, .25)), 'auc_q3': float(np.quantile(a, .75)),
                      'pooled_micro_bacc': float(balanced_accuracy_score(y[ok], P[k][ok] >= .5)),
                      'pooled_auc_within_subject_scores': float(roc_auc_score(y[ok], P[k][ok]))}
        log(f'[summary] {k:5s} bACC median={summary[k]["bacc_median"]:.4f} IQR=[{summary[k]["bacc_q1"]:.4f},'
            f'{summary[k]["bacc_q3"]:.4f}] mean={summary[k]["bacc_mean"]:.4f} >=0.9={summary[k]["frac_bacc_ge_0.9"]:.3f} '
            f'AUC median={summary[k]["auc_median"]:.4f} micro={summary[k]["pooled_micro_bacc"]:.4f}')

    factors = ['amp_std_median', 'hf_noise_ratio_median', 'flat_or_bad_frac', 'flat_frac', 'amp_outlier_frac', 'max_abs_cohen_d',
               'direction_flip', 'merge_best_auc', 'n_blocks', 'n_windows']
    corr = {}
    for k in P:
        corr[k] = {}
        for f_ in factors:
            pairs = [(r[f'{k}_bacc'], r[f_]) for r in rows if r[f'{k}_bacc'] is not None]
            a, b = np.array(pairs).T
            rho, pv = spearmanr(a, b) if np.std(b) > 0 else (np.nan, np.nan)
            corr[k][f_] = {'rho': float(rho), 'p': float(pv)}
        log(f'[spearman] {k:5s} ' + ' '.join(f'{f_}={corr[k][f_]["rho"]:+.2f}(p={corr[k][f_]["p"]:.3f})' for f_ in factors))

    order = sorted([r for r in rows if r['mean3_bacc'] is not None], key=lambda r: -r['mean3_bacc'])
    log('[table] subject  bACC(lr/hgb/cnn/mean3)  AUC_mean3  blocks  amp  hf  bad  maxd  mbAUC')
    for r in order:
        log(f'  {r["subject"]} {r["lr_bacc"]:.3f}/{r["hgb_bacc"]:.3f}/{r["cnn_bacc"]:.3f}/{r["mean3_bacc"]:.3f} '
            f'{r["mean3_auc"]:.3f} {r["n_blocks"]} {r["amp_std_median"]:.3g} {r["hf_noise_ratio_median"]:.3f} '
            f'{r["flat_or_bad_frac"]:.3f} {r["max_abs_cohen_d"]:.2f} {r["merge_best_auc"]:.3f}')
    k8 = max(1, len(order) // 4)
    grp = {}
    for name, part in (('top_quartile', order[:k8]), ('bottom_quartile', order[-k8:])):
        grp[name] = {'subjects': [r['subject'] for r in part],
                     **{f_: float(np.median([r[f_] for r in part])) for f_ in factors}}
    log(f'[groups] {json.dumps(grp, ensure_ascii=False)}')

    result = {'protocol': {
        'deployable': False, 'purpose': 'within-subject leave-one-state-block-out ceiling diagnostic',
        'block_definition': 'consecutive same-label trials in subject time order (canonical rows of dense cache give trial id; '
                            'label-run count = 339, same as causal_kernel_state_mapping switch definition)',
        'train_test': 'train = canonical non-overlapping 6 s windows of the same subject from other blocks; test = held-out block; '
                      'no dense windows used, so no train window overlaps the test block in time',
        'skip_rule': 'block-folds whose remaining blocks contain only one class are skipped (those windows unscored)',
        'models': {'lr': 'StandardScaler+LogisticRegression(C=1, balanced) on 16 Welch features',
                   'hgb': 'HistGB(max_iter=150, depth=3, lr=0.06, class_weight=balanced)',
                   'cnn': f'grouped-conv 1D CNN (8/16/16 ch, k25s2/9/9, GAP), AdamW lr2e-3 wd1e-3, {steps} steps, '
                          'balanced batch 32, per-window demean + train-block std scaling, seed 0',
                   'mean3': 'mean of three probabilities (pre-specified, not selected)'},
        'threshold': 0.5, 'hyperparameter_search': False,
        'quality_factors': {'amp_std_median': 'median per-window std', 'hf_noise_ratio_median': 'P(52-100Hz)/P(1-45Hz) median',
                            'flat_or_bad_frac': 'windows with >=25-sample zero-diff run or std >6 MAD from median or std~0',
                            'max_abs_cohen_d': 'max |Cohen d| R vs T over the 16 features (uses labels, diagnostic)',
                            'direction_flip': 'merge_best_window6 combo__C4_bb_resnet_eq subject AUC < 0.5'}},
        'summary': summary, 'spearman_vs_subject_bacc': corr, 'groups': grp, 'per_subject': rows,
        'n_tasks': len(tasks), 'smoke': args.smoke, 'seconds': time.perf_counter() - T0}
    paths[1].write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    np.savez_compressed(paths[2], y=y, groups=g, block=block, trial=trial, **{f'p_{k}': v for k, v in P.items()})
    log(f'[done] -> {paths[1].name}, {paths[2].name}')


if __name__ == '__main__':
    main()
