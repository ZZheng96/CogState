"""评估 D:\\BCIC2026\\ref 中两份资料在本赛题上的可用性。

资料：
  1) braindecode 0.4（2020）：exponential_running_standardize / ShallowFBCSPNet / deep4 / CroppedLoss
  2) Schirrmeister et al. 2017, "Deep learning with CNNs for EEG decoding and visualization"

本脚本只做两件"可能迁移得动"的事，其余（多通道空间滤波、双向 BN、FBCSP）在单通道 6 秒因果约束下不可用：

实验一  braindecode 的 causal EMA 归一化（alpha=1e-3 @250Hz ≈ 4 s 时间常数）
        在窗内对原始波形做归一化后重算 858 维特征，比较与"不做归一化"的差异。
        理由：单通道无法用 ICA/ASR，在线自适应是唯一合规的幅值漂移对策。

实验二  ShallowFBCSPNet 单通道化（conv-time → BN → square → avgpool → log → linear）
        + 论文的 cropped training（训练时随机裁剪子窗做增强）。
        理由：这是论文的核心贡献，参数量小、归纳偏置等于"可学习滤波器组 + 对数功率"。

用法：python test/ref_braindecode_eval.py --part all
"""
from __future__ import annotations
import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'test'
sys.path[:0] = [str(ROOT / 'src'), str(OUT)]

FEAT = OUT / 'highdim_deep_features.npz'
EMG = OUT / 'emg_features_v2.npz'
RESULT = OUT / 'ref_braindecode_results.json'
N_JOBS = 12


# ---------------------------------------------------------------- 实验一：EMA 归一化

def exponential_running_standardize(x, alpha=1e-3, eps=1e-4):
    """braindecode.datautil.signalproc.exponential_running_standardize 的单通道实现。

    m_t = alpha*x_t + (1-alpha)*m_{t-1}；v_t 同理；x'_t = (x_t - m_t)/sqrt(v_t)。
    严格因果：t 时刻只用 0..t 的样本。前 1 秒用普通 z-score 做暖启动。
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = np.empty(n)
    v = np.empty(n)
    acc_m, acc_v = x[0], 0.0
    for i in range(n):
        acc_m = alpha * x[i] + (1 - alpha) * acc_m
        d = x[i] - acc_m
        acc_v = alpha * d * d + (1 - alpha) * acc_v
        m[i], v[i] = acc_m, acc_v
    y = (x - m) / np.maximum(np.sqrt(v), eps)
    warm = int(1.0 * 250)
    if warm > 1:
        y[:warm] = (x[:warm] - x[:warm].mean()) / max(x[:warm].std(), eps)
    return y


def warmup_zscore(x, seconds=1.0, fs=250.0):
    """因果暖启动 z-score：只用窗首 1 秒的均值/方差整窗归一化（对照组）。"""
    x = np.asarray(x, dtype=float)
    k = int(seconds * fs)
    mu, sd = x[:k].mean(), max(x[:k].std(), 1e-4)
    return (x - mu) / sd


def _feat_one(args):
    from highdim_deep_ablation import highdim_features
    x, mode = args
    if mode == 'ema1e-3':
        x = exponential_running_standardize(x, 1e-3)
    elif mode == 'ema3e-4':
        x = exponential_running_standardize(x, 3e-4)
    elif mode == 'warm1s':
        x = warmup_zscore(x, 1.0)
    elif mode == 'ema1e-3_warm1s':
        x = warmup_zscore(exponential_running_standardize(x, 1e-3), 1.0)
    return highdim_features(x).astype(np.float32).tolist()


def extract_variant(raw, mode):
    with ProcessPoolExecutor(max_workers=N_JOBS) as pool:
        chunks = pool.map(_feat_one, ((x, mode) for x in raw), chunksize=64)
        return np.asarray(list(chunks), dtype=np.float32)


# ---------------------------------------------------------------- 评测工具

def cv_oof(X, y, groups, model='logistic', k=300, seed=0):
    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups))
    oof = np.full(len(y), np.nan)
    for tr, te in splits:
        A, T = X[tr].astype(np.float64), X[te].astype(np.float64)
        A = A - A.mean(1, keepdims=True)
        T = T - T.mean(1, keepdims=True)
        sc = StandardScaler().fit(A)
        A, T = sc.transform(A), sc.transform(T)
        if k and k < X.shape[1]:
            sel = SelectKBest(f_classif, k=k).fit(A, y[tr])
            A, T = sel.transform(A), sel.transform(T)
        if model == 'logistic':
            clf = LogisticRegression(C=1, max_iter=2000, class_weight='balanced', random_state=0).fit(A, y[tr])
            oof[te] = clf.predict_proba(T)[:, 1]
        elif model == 'histgb':
            from sklearn.ensemble import HistGradientBoostingClassifier
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_leaf_nodes=15,
                                                 min_samples_leaf=40, l2_regularization=1., random_state=0)
            clf.fit(A, y[tr])
            oof[te] = clf.predict_proba(T)[:, 1]
        else:
            raise ValueError(model)
    return oof


def report(y, s, groups):
    r = {'AUC': float(roc_auc_score(y, s)), 'bACC': float(balanced_accuracy_score(y, s >= .5))}
    per = [(roc_auc_score(y[groups == g], s[groups == g]), balanced_accuracy_score(y[groups == g], s[groups == g] >= .5))
           for g in np.unique(groups) if len(set(y[groups == g])) == 2]
    r['subject_AUC'] = float(np.mean([p[0] for p in per]))
    r['subject_bACC'] = float(np.mean([p[1] for p in per]))
    return r


def log(tag, r, extra=''):
    print(f'[{tag}] AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f} '
          f'子均AUC={r["subject_AUC"]:.4f} 子均bACC={r["subject_bACC"]:.4f} {extra}', flush=True)


# ---------------------------------------------------------------- 实验二：ShallowFBCSPNet 单通道版

def shallow_net(drop=.5, n_filt=40, k=25, pool=75, stride=15):
    import torch
    from torch import nn

    class Square(nn.Module):
        def forward(self, x):
            return x * x

    class Log(nn.Module):
        def forward(self, x):
            return torch.log(torch.clamp(x, min=1e-6))

    return nn.Sequential(
        nn.Conv1d(1, n_filt, k, padding=0), nn.BatchNorm1d(n_filt), Square(),
        nn.AvgPool1d(pool, stride), Log(), nn.Dropout(drop),
        nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(n_filt, 1))


def train_fold(raw, y, tr, te, crop=None, epochs=40, seed=0, val_frac=.2, verbose=False):
    """训练一个折；crop 为训练时随机裁剪长度（None=用整窗）。返回测试折概率。"""
    import torch
    from torch import nn
    torch.manual_seed(seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 按受试者切出验证集做早停
    gtr = np.unique(tr)
    rng = np.random.default_rng(seed)
    val_sub = set(rng.choice(gtr, max(1, int(len(gtr) * val_frac)), replace=False))
    is_val = np.isin(tr, list(val_sub))
    idx_tr, idx_va = tr[~is_val], tr[is_val]
    mu, sd = raw[idx_tr].mean(), raw[idx_tr].std() + 1e-8

    def tensor(ids, crop_len=None, rng=None):
        x = (raw[ids] - mu) / sd
        if crop_len is not None:
            starts = rng.integers(0, x.shape[1] - crop_len + 1, len(ids))
            x = np.stack([a[s:s + crop_len] for a, s in zip(x, starts)])
        return torch.tensor(np.ascontiguousarray(x[:, None, :]), dtype=torch.float32)

    net = shallow_net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.BCEWithLogitsLoss()
    ytr = torch.tensor(y[tr][~is_val], dtype=torch.float32)
    Xva = tensor(idx_va).to(dev)
    yva = y[tr][is_val]
    best, best_state, patience = -1, None, 0
    bs = 128
    for ep in range(epochs):
        net.train()
        Xtr = tensor(idx_tr, crop, rng)  # 每轮重裁，等价于论文的随机 crop 增强
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            xb = Xtr[b].to(dev)
            opt.zero_grad()
            loss = lossf(net(xb).squeeze(1), ytr[b].to(dev))
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(net(Xva).squeeze(1)).cpu().numpy()
        a = roc_auc_score(yva, pv)
        if a > best:
            best, patience = a, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            patience += 1
            if patience >= 8:
                break
        if verbose:
            print(f'    ep{ep+1} valAUC={a:.4f} best={best:.4f}', flush=True)
    net.load_state_dict(best_state)
    net.eval()
    out = []
    with torch.no_grad():
        Xte = tensor(te).to(dev)
        if crop is None:
            out = torch.sigmoid(net(Xte).squeeze(1)).cpu().numpy()
        else:
            # 整窗多裁剪平均（仍只用这 6 秒内的信号）
            acc = np.zeros(len(te))
            starts = np.linspace(0, raw.shape[1] - crop, 5).astype(int)
            for s in starts:
                acc += torch.sigmoid(net(Xte[:, :, s:s + crop]).squeeze(1)).cpu().numpy()
            out = acc / len(starts)
    n_par = sum(p.numel() for p in net.parameters())
    return out, n_par


def run_shallow(X_ref, raw, y, groups, seed=0):
    res = {}
    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X_ref, y, groups))
    for tag, crop in (('整窗 6s', None), ('cropped 4s', 1000), ('cropped 3s', 750)):
        oof = np.full(len(y), np.nan)
        n_par = 0
        t0 = time.perf_counter()
        for f, (tr, te) in enumerate(splits, 1):
            out, n_par = train_fold(raw, y, tr, te, crop=crop, seed=seed + f)
            oof[te] = out
            print(f'  [{tag}] fold{f}/5 AUC={roc_auc_score(y[te], out):.4f} ({time.perf_counter()-t0:.0f}s)', flush=True)
        r = report(y, oof, groups)
        r['params'] = int(n_par)
        res[tag] = r
        log(f'shallow · {tag}（{n_par} 参数）', r)
    return res, oof


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', default='all', choices=['ema', 'shallow', 'all'])
    args = ap.parse_args()
    tick = time.perf_counter()
    F = np.load(FEAT)
    X0, raw, y, groups = F['X'].astype(np.float32), F['raw'], F['y'], F['groups']
    E = np.load(EMG)['E'].astype(np.float32)
    keep = X0.std(0) > 1e-8
    X0, E = X0[:, keep], E
    print(f'[data] 窗 {X0.shape} 原始 {raw.shape} 肌电 {E.shape} 受试者 {len(np.unique(groups))}', flush=True)
    res = {}
    if RESULT.exists():
        res = json.loads(RESULT.read_text(encoding='utf-8'))

    if args.part in ('ema', 'all'):
        base = cv_oof(np.hstack([X0, E]), y, groups, 'logistic', 300)
        res['base_logistic300'] = report(y, base, groups)
        log('基线 · 858+20 · Logistic300', res['base_logistic300'])
        np.savez_compressed(OUT / 'ref_oof_base.npz', y=y, groups=groups, base=base)

        for mode in ('ema1e-3', 'warm1s', 'ema3e-4'):
            t0 = time.perf_counter()
            Xa = extract_variant(raw, mode)
            Xa = Xa[:, keep] if Xa.shape[1] == len(keep) else Xa
            np.savez_compressed(OUT / f'ref_feat_{mode}.npz', X=Xa)
            print(f'[extract] {mode} {Xa.shape} {time.perf_counter()-t0:.0f}s', flush=True)
            o = cv_oof(np.hstack([Xa, E]), y, groups, 'logistic', 300)
            res[f'{mode}_logistic300'] = report(y, o, groups)
            log(f'{mode} · Logistic300', res[f'{mode}_logistic300'])
            if mode == 'ema1e-3':
                o2 = cv_oof(np.hstack([Xa, E]), y, groups, 'histgb', 0)
                res[f'{mode}_histgb'] = report(y, o2, groups)
                log(f'{mode} · HistGB 全维', res[f'{mode}_histgb'])
        o3 = cv_oof(np.hstack([X0, E]), y, groups, 'histgb', 0)
        res['base_histgb'] = report(y, o3, groups)
        log('基线 · HistGB 全维', res['base_histgb'])

    if args.part in ('shallow', 'all'):
        Xr = np.hstack([X0, E])
        rs, oof = run_shallow(Xr, raw, y, groups)
        res['shallow'] = rs
        np.savez_compressed(OUT / 'ref_oof_shallow.npz', y=y, groups=groups, last=oof)

    res['_meta'] = {'seconds': time.perf_counter() - tick, 'n_windows': int(len(y))}
    RESULT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[done] {time.perf_counter()-tick:.0f}s -> {RESULT}', flush=True)


if __name__ == '__main__':
    main()
