"""严格 6 秒单窗：组合 C 四成员 + F3 三路的可部署融合。

对齐：database 窗与 highdim 窗用（受试者, 标签, raw md5）多重集一一映射；
C4 成员按原脚本在 database 顺序重训（StratifiedGroupKFold(5,shuffle,rs=0) 的受试者→折分配两种顺序一致），
最后全部换回 highdim 顺序，与 seed_noise_band 的 fold_id 对齐。
融合：每成员在外层训练折内做 4 折分组内层 CV 得到 OOF -> logit -> 用内层 OOF 的均值/方差 z 标准化测试折分数；
权重 = 等权 或 内层 OOF 上的非负 Logistic 堆叠。ResNet 无内层 OOF，按固定等权份额、raw logit 加入。
测试折上不选择任何东西。
"""
import hashlib, json, sys, threading, time
from collections import defaultdict
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.signal import welch
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'test'
DB = ROOT / 'database'
sys.path[:0] = [str(OUT)]
RES_PATH = OUT / 'merge_best_window6_results.json'
OOF_PATH = OUT / 'merge_best_window6_oof.npz'
N_INNER = 4
N_JOBS = 10
BOOT_SEED = 20261029
N_BOOT = 1000
T0 = time.perf_counter()


def log(msg):
    print(f'[{time.perf_counter() - T0:7.1f}s] {msg}', flush=True)


def to_logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# ---------------- 成员训练（在 worker 中运行） ----------------
def fit_member(name, Xtr, ytr, gtr, Xte):
    """返回测试样本的 logit 尺度分数（概率模型取 logit，SVC 取 decision_function）。"""
    if name in ('base_hgb', 'w05_lr', 'w05_hgb'):
        from highdim_deep_ablation import FeatureTransform
        A, T = Xtr.astype(np.float64), Xte.astype(np.float64)
        A = A - A.mean(1, keepdims=True); T = T - T.mean(1, keepdims=True)
        sc = StandardScaler().fit(A); A, T = sc.transform(A), sc.transform(T)
        if name == 'w05_lr':
            tf = FeatureTransform(300, 'kbest').fit(A, ytr)
            A, T = tf.transform(A).astype(np.float64), tf.transform(T).astype(np.float64)
            m = LogisticRegression(C=1, max_iter=3000, class_weight='balanced', random_state=0)
        else:
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_depth=3,
                                               l2_regularization=1., random_state=0)
        m.fit(A, ytr)
        return to_logit(m.predict_proba(T)[:, 1])
    if name == 'svc_base':
        from highdim_deep_ablation import FeatureTransform
        tf = FeatureTransform(300, 'kbest').fit(Xtr, ytr)
        A, T = tf.transform(Xtr).astype(np.float64), tf.transform(Xte).astype(np.float64)
        m = make_pipeline(StandardScaler(), SVC(C=1, gamma='scale', kernel='rbf', cache_size=2048)).fit(A, ytr)
        return m.decision_function(T)
    if name in ('bb_lr', 'wbb_lr'):
        from broadband_shape_window6 import fit_logreg
        pte, _, _ = fit_logreg(Xtr.astype(np.float64), ytr, gtr, Xte.astype(np.float64))
        return to_logit(pte)
    raise KeyError(name)


def task(name, fold, part, tr, te, X, y, g):
    t = time.perf_counter()
    s = fit_member(name, X[tr], y[tr], g[tr], X[te])
    print(f'  [task] {name:9s} fold={fold} part={part} n_tr={len(tr)} n_te={len(te)} '
          f'auc={roc_auc_score(y[te], s):.4f} {time.perf_counter() - t:.0f}s', flush=True)
    return name, fold, part, te, s


# ---------------- 融合工具 ----------------
def youden_thr(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    return float(thr[np.argmax(tpr - fpr)])


def nonneg_stack(Z, y):
    """非负系数 Logistic（截距自由），按 L-BFGS-B 最小化平均 logloss；返回归一化到和为 1 的权重。"""
    k = Z.shape[1]

    def f(w):
        z = Z @ w[:k] + w[k]
        loss = np.mean(np.logaddexp(0, z) - y * z)
        r = 1 / (1 + np.exp(-z)) - y
        return loss, np.r_[Z.T @ r / len(y), r.mean()]

    w = minimize(f, np.r_[np.full(k, .1), 0.], jac=True, method='L-BFGS-B',
                 bounds=[(0, None)] * k + [(None, None)]).x[:k]
    return (w / w.sum()) if w.sum() > 1e-12 else np.full(k, 1 / k), w


def subj_auc(y, s, g):
    v = [roc_auc_score(y[g == u], s[g == u]) for u in np.unique(g) if len(np.unique(y[g == u])) == 2]
    return np.array(v)


def main():
    if RES_PATH.exists() or OOF_PATH.exists():
        raise FileExistsError('refusing to overwrite merge_best_window6 artifacts')
    import torch
    from seed_noise_band import DEVICE, fit_mlp, predict
    torch.set_num_threads(2)

    # ---------- 数据与对齐 ----------
    hd = np.load(OUT / 'highdim_deep_features.npz')
    raw_h, y_h, g_h, X0_h = hd['raw'], hd['y'].astype(int), hd['groups'].astype(str), hd['X']
    sn = np.load(OUT / 'seed_noise_band_oof.npz')
    fold_h = sn['fold_id'].astype(int)
    assert (sn['y'] == y_h).all() and (sn['groups'].astype(str) == g_h).all()

    key = defaultdict(list)
    for i, r in enumerate(raw_h):
        key[(g_h[i], int(y_h[i]), hashlib.md5(r.tobytes()).hexdigest())].append(i)
    idx, fs, es, ys, gs = [], [], [], [], []
    for p in sorted((DB / 'windows').glob('*/*.npz')):
        d = np.load(p)
        fs.append(d['f']); es.append(d['e']); ys.append(d['label']); gs.append(d['subject'])
        for x, s, l in zip(d['x'], d['subject'], d['label']):
            idx.append(key[(str(s), int(l), hashlib.md5(x.astype(np.float32).tobytes()).hexdigest())].pop(0))
    idx = np.array(idx)  # database 第 i 行 = highdim 第 idx[i] 行
    F = np.vstack(fs).astype(np.float64); E = np.vstack(es).astype(np.float64)
    y = np.concatenate(ys).astype(int); g = np.concatenate(gs).astype(str)
    assert len(np.unique(idx)) == len(y_h) == len(y) and (y_h[idx] == y).all() and (g_h[idx] == g).all()
    V = np.load(OUT / 'features_v3.npz')['V'].astype(np.float64)  # database 顺序
    W = np.load(OUT / 'b_warm05.npz')['W'].astype(np.float64)[idx]  # highdim -> database
    BB = np.load(OUT / 'broadband_shape_features.npz')['X'].astype(np.float64)[idx]
    keep = F.std(0) > 1e-8
    feats = {'base_hgb': np.hstack([F[:, keep], E]),
             'w05_lr': np.hstack([W[:, keep], E]),
             'w05_hgb': None,
             'svc_base': np.hstack([F[:, keep], E, V]),
             'bb_lr': BB}
    feats['w05_hgb'] = feats['w05_lr']
    log(f'[align] multiset md5 map ok n={len(y)}; base={feats["base_hgb"].shape} warm={feats["w05_lr"].shape} '
        f'svc={feats["svc_base"].shape} bb={BB.shape}')

    # 任务 4：warm0.5 后的 86 维宽频特征
    t = time.perf_counter()
    from broadband_shape_window6 import compute_features
    k0 = 125
    mu = raw_h[:, :k0].astype(np.float64).mean(1, keepdims=True)
    sd = np.maximum(raw_h[:, :k0].astype(np.float64).std(1, keepdims=True), 1e-4)
    rw = (raw_h.astype(np.float64) - mu) / sd
    fw, Pw = welch(rw, fs=250., nperseg=500, noverlap=250, axis=1, detrend='constant')
    fam = compute_features(rw, fw, Pw)
    WBB = np.hstack([fam[k][0] for k in fam]).astype(np.float64)
    WBB[~np.isfinite(WBB)] = 0.
    feats['wbb_lr'] = WBB[idx]
    log(f'[extract] warm0.5 broadband {WBB.shape} {time.perf_counter() - t:.0f}s')

    # 折（database 顺序；与 highdim fold_id 的受试者分配一致）
    outer = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(F, y, g))
    fold_db = np.zeros(len(y), int)
    for k, (_, te) in enumerate(outer, 1):
        fold_db[te] = k
    assert (fold_h[idx] == fold_db).all(), 'fold mismatch between orders'
    inner = {}
    for k, (tr, te) in enumerate(outer, 1):
        inner[k] = [(tr[a], tr[b]) for a, b in
                    StratifiedGroupKFold(N_INNER, shuffle=True, random_state=k).split(tr[:, None], y[tr], g[tr])]
    log('[folds] outer=highdim fold_id; inner=StratifiedGroupKFold(4,shuffle,rs=fold) on train subjects')

    # ---------- CPU 成员（joblib）与 GPU MLP 内层并行 ----------
    cpu_members = ['svc_base', 'base_hgb', 'w05_lr', 'w05_hgb', 'wbb_lr', 'bb_lr']
    jobs = []
    for name in cpu_members:
        for k, (tr, te) in enumerate(outer, 1):
            if name != 'bb_lr':
                jobs.append((name, k, 'outer', tr, te))
            for j, (a, b) in enumerate(inner[k]):
                jobs.append((name, k, f'inner{j}', a, b))
    log(f'[cpu] {len(jobs)} tasks on {N_JOBS} workers')
    cpu_out = {}

    def run_cpu():
        res = Parallel(n_jobs=N_JOBS, backend='loky', verbose=0)(
            delayed(task)(n, k, p, tr, te, feats[n], y, g) for n, k, p, tr, te in jobs)
        cpu_out['res'] = res

    th = threading.Thread(target=run_cpu); th.start()

    # MLP（highdim 顺序，与 seed_noise_band 完全同口径）在每个外层训练折内做内层 OOF
    X0 = X0_h.astype('float32')
    mlp_inner = {}
    tm = time.perf_counter()
    for k in range(1, 6):
        tr_h = np.where(fold_h != k)[0]
        kp = X0[tr_h].std(0) > 1e-8
        m_, s_ = X0[tr_h][:, kp].mean(0), np.maximum(X0[tr_h][:, kp].std(0), 1e-6)
        Xk = ((X0[:, kp] - m_) / s_).astype('float32')
        sc = np.zeros(len(y_h))
        for j, (a, b) in enumerate(inner[k]):
            a_h, b_h = idx[a], idx[b]
            it, iv = next(StratifiedGroupKFold(5, shuffle=True, random_state=100 * k + j)
                          .split(a_h[:, None], y_h[a_h], g_h[a_h]))
            p = np.zeros(len(b_h))
            for seed in range(5):
                model, ep = fit_mlp(Xk, y_h, a_h[it], a_h[iv], seed)
                p += predict(model, Xk[b_h]) / 5
            sc[b_h] = to_logit(p)
            print(f'  [mlp] fold={k} inner{j} auc={roc_auc_score(y_h[b_h], p):.4f} '
                  f'{time.perf_counter() - tm:.0f}s', flush=True)
        mlp_inner[k] = sc[idx]  # 存为 database 顺序（仅训练折位置有效）
    log(f'[mlp] inner OOF done {time.perf_counter() - tm:.0f}s, waiting CPU tasks')
    th.join()
    log('[cpu] done')

    # ---------- 汇总分数（database 顺序） ----------
    members = ['base_hgb', 'w05_lr', 'w05_hgb', 'svc_base', 'hand_mlp', 'bb_lr', 'wbb_lr', 'resnet']
    outer_s = {m: np.full(len(y), np.nan) for m in members}
    inner_s = {m: {k: np.full(len(y), np.nan) for k in range(1, 6)} for m in members if m != 'resnet'}
    for name, k, part, te, s in cpu_out['res']:
        (outer_s[name] if part == 'outer' else inner_s[name][k])[te] = s
    outer_s['bb_lr'] = to_logit(np.load(OUT / 'broadband_shape_window6_oof.npz')['logreg__all_new'])[idx]
    outer_s['hand_mlp'] = to_logit(sn['oof_seed_mean'])[idx]
    outer_s['resnet'] = to_logit(np.load(OUT / 'raw_waveform_window6_oof.npz')['resnet1d_seed_mean'])[idx]
    for k in range(1, 6):
        inner_s['hand_mlp'][k] = mlp_inner[k]
    for m in members:
        assert np.isfinite(outer_s[m]).all(), m
        if m != 'resnet':
            for k, (tr, _) in enumerate(outer, 1):
                assert np.isfinite(inner_s[m][k][tr]).all(), (m, k)

    # z 标准化（训练折内层 OOF 的 mu/sd）
    Zte = {m: np.zeros(len(y)) for m in members}
    Zin = {m: {} for m in members if m != 'resnet'}
    zstats = {}
    for m in members:
        for k, (tr, te) in enumerate(outer, 1):
            if m == 'resnet':
                Zte[m][te] = outer_s[m][te]
                continue
            v = inner_s[m][k][tr]
            mu_, sd_ = v.mean(), max(v.std(), 1e-8)
            Zin[m][k] = (v - mu_) / sd_
            Zte[m][te] = (outer_s[m][te] - mu_) / sd_
            zstats.setdefault(m, []).append({'fold': k, 'mu': float(mu_), 'sd': float(sd_),
                                             'inner_oof_auc': float(roc_auc_score(y[tr], v))})

    # ---------- 组合 ----------
    C4 = ['base_hgb', 'w05_lr', 'w05_hgb', 'svc_base']
    F3 = ['hand_mlp', 'resnet', 'bb_lr']
    combos = {
        'C4_eq': (C4, 'eq'), 'C4_stack': (C4, 'stack'),
        'F3_eq': (F3, 'eq'),
        'C4F3_eq': (C4 + F3, 'eq'), 'C4F3_stack': (C4 + F3, 'stack'),
        'C4_bb_resnet_eq': (C4 + ['bb_lr', 'resnet'], 'eq'),
        'C4F3_wbb_eq': (C4 + F3 + ['wbb_lr'], 'eq'), 'C4F3_wbb_stack': (C4 + F3 + ['wbb_lr'], 'stack'),
    }
    scores, info = {}, {}
    for cname, (ms, how) in combos.items():
        s = np.zeros(len(y)); thr_pred = np.zeros(len(y), int); fold_rows = []
        has_res = 'resnet' in ms
        zm = [m for m in ms if m != 'resnet']
        for k, (tr, te) in enumerate(outer, 1):
            Zi = np.column_stack([Zin[m][k] for m in zm])
            if how == 'eq':
                w = np.full(len(zm), 1 / len(zm)); wraw = None
            else:
                w, wraw = nonneg_stack(Zi, y[tr])
            share = (len(zm) / len(ms)) if has_res else 1.
            s[te] = share * (np.column_stack([Zte[m][te] for m in zm]) @ w) + \
                ((1 / len(ms)) * Zte['resnet'][te] if has_res else 0.)
            thr = youden_thr(y[tr], share * (Zi @ w))  # ResNet 无训练折分数：阈值仅由非 ResNet 部分的内层 OOF 选出
            thr_pred[te] = s[te] >= thr
            fold_rows.append({'fold': k, 'auc': float(roc_auc_score(y[te], s[te])),
                              'bacc_at0': float(balanced_accuracy_score(y[te], s[te] >= 0)),
                              'train_thr': thr,
                              'weights': {m: float(x) * share for m, x in zip(zm, w)} |
                                         ({'resnet': 1 / len(ms)} if has_res else {}),
                              'stack_raw_coef': None if wraw is None else [float(x) for x in wraw]})
        sa = subj_auc(y, s, g)
        scores[cname] = s
        info[cname] = {'members': ms, 'weighting': how,
                       'oof_auc': float(roc_auc_score(y, s)),
                       'bacc_at0': float(balanced_accuracy_score(y, s >= 0)),
                       'bacc_train_thr': float(balanced_accuracy_score(y, thr_pred)),
                       'train_thr_note': ('threshold chosen on inner OOF of non-ResNet part only (ResNet has no train-fold scores)'
                                          if has_res else 'Youden threshold on inner OOF of the fused score'),
                       'fold_auc': [r['auc'] for r in fold_rows],
                       'subject_auc_median': float(np.median(sa)),
                       'subject_auc_lt_0.5_frac': float(np.mean(sa < .5)),
                       'per_fold': fold_rows}
        log(f'[combo] {cname:16s} AUC={info[cname]["oof_auc"]:.4f} bACC@0={info[cname]["bacc_at0"]:.4f} '
            f'bACC@thr={info[cname]["bacc_train_thr"]:.4f} subj<0.5={info[cname]["subject_auc_lt_0.5_frac"]:.3f}')

    # 参照（不可部署 / 旧口径）
    ws = np.load(OUT / 'warmup_sweep_oof.npz'); kt = np.load(OUT / 'kernel_trees_oof.npz')
    orig = {'base_hgb': ws['base_histgb'], 'w05_lr': ws['warm0.5_lr'], 'w05_hgb': ws['warm0.5_histgb'],
            'svc_base': kt['svc_rbf_c1']}
    assert (ws['y'] == y).all() and (kt['y'] == y).all()
    ref = {'C4_rank_retrained_nondeployable': np.mean([rankdata(outer_s[m]) / len(y) for m in C4], 0),
           'C4_rank_original_oof_nondeployable': np.mean([rankdata(orig[m]) / len(y) for m in C4], 0),
           'F3_rawlogit_eq_original': np.mean([outer_s[m] for m in F3], 0)}
    for rn, s in ref.items():
        thr0 = .5 if 'rank' in rn else 0.
        sa = subj_auc(y, s, g)
        scores[rn] = s
        info[rn] = {'oof_auc': float(roc_auc_score(y, s)), 'bacc_at_default': float(balanced_accuracy_score(y, s >= thr0)),
                    'default_thr': thr0, 'fold_auc': [float(roc_auc_score(y[te], s[te])) for _, te in outer],
                    'subject_auc_median': float(np.median(sa)), 'subject_auc_lt_0.5_frac': float(np.mean(sa < .5)),
                    'note': 'reference only, rank average uses pooled test-fold ranks' if 'rank' in rn else 'reference (fusion3_window6 F3)'}
        log(f'[ref]   {rn:36s} AUC={info[rn]["oof_auc"]:.4f} bACC={info[rn]["bacc_at_default"]:.4f}')

    # ---------- 复现 ----------
    known = {'base_hgb': .6293, 'w05_lr': .6450, 'w05_hgb': .6528, 'svc_base': .6454}
    repro = {}
    for m in members:
        a = float(roc_auc_score(y, outer_s[m]))
        r = {'oof_auc': a, 'fold_auc': [float(roc_auc_score(y[te], outer_s[m][te])) for _, te in outer],
             'mean_inner_oof_auc': float(np.mean([z['inner_oof_auc'] for z in zstats.get(m, [])])) if m in zstats else None,
             'outer_source': 'retrained' if m in C4 + ['wbb_lr'] else 'saved OOF (deterministic reuse)'}
        if m in known:
            r['known_auc'] = known[m]; r['delta_vs_known'] = a - known[m]
            r['spearman_vs_original_oof'] = float(spearmanr(outer_s[m], orig[m])[0])
            r['original_oof_auc_recomputed'] = float(roc_auc_score(y, orig[m]))
        repro[m] = r
        log(f'[member] {m:9s} AUC={a:.4f}' + (f' known={known[m]:.4f} d={a - known[m]:+.4f} '
                                              f'rho={r["spearman_vs_original_oof"]:.4f}' if m in known else ''))
    repro['C4_rank_avg'] = {'retrained': info['C4_rank_retrained_nondeployable']['oof_auc'],
                            'original_oof': info['C4_rank_original_oof_nondeployable']['oof_auc'], 'known': .6639}

    # ---------- bootstrap ----------
    tb = time.perf_counter()
    rng = np.random.default_rng(BOOT_SEED)
    subs = np.unique(g); sidx = {u: np.where(g == u)[0] for u in subs}
    boots = [np.concatenate([sidx[u] for u in rng.choice(subs, len(subs))]) for _ in range(N_BOOT)]
    bauc = {c: np.empty(N_BOOT) for c in scores}
    for i, b in enumerate(boots):
        for c, s in scores.items():
            bauc[c][i] = roc_auc_score(y[b], s[b])
        if (i + 1) % 200 == 0:
            log(f'[boot] {i + 1}/{N_BOOT}')
    boot = {}
    for c in scores:
        boot[c] = {'auc_ci95': [float(np.percentile(bauc[c], 2.5)), float(np.percentile(bauc[c], 97.5))]}
        for refn in ('F3_eq', 'C4_eq', 'F3_rawlogit_eq_original'):
            if c == refn:
                continue
            dl = bauc[c] - bauc[refn]
            boot[c][f'delta_vs_{refn}'] = {
                'point': float(roc_auc_score(y, scores[c]) - roc_auc_score(y, scores[refn])),
                'ci95': [float(np.percentile(dl, 2.5)), float(np.percentile(dl, 97.5))],
                'p_delta_gt0': float(np.mean(dl > 0))}
        info[c]['bootstrap'] = boot[c]
    log(f'[boot] done {time.perf_counter() - tb:.0f}s')
    for c in combos:
        b = boot[c]
        msg = ' '.join(f'{r}:{b[f"delta_vs_{r}"]["point"]:+.4f}[{b[f"delta_vs_{r}"]["ci95"][0]:+.4f},'
                       f'{b[f"delta_vs_{r}"]["ci95"][1]:+.4f}]P={b[f"delta_vs_{r}"]["p_delta_gt0"]:.3f}'
                       for r in ('F3_eq', 'C4_eq') if f'delta_vs_{r}' in b)
        log(f'[delta] {c:16s} {msg}')

    # ---------- 保存（highdim 顺序） ----------
    inv = np.empty_like(idx); inv[idx] = np.arange(len(idx))  # highdim 第 j 行 = database 第 inv[j] 行
    assert (y[inv] == y_h).all() and (fold_db[inv] == fold_h).all()
    out = {'y': y_h, 'groups': g_h, 'fold_id': fold_h}
    out |= {f'member__{m}': outer_s[m][inv] for m in members}
    out |= {f'member_z__{m}': Zte[m][inv] for m in members}
    out |= {f'combo__{c}': s[inv] for c, s in scores.items()}
    np.savez_compressed(OOF_PATH, **out)

    elapsed = time.perf_counter() - T0
    result = {
        'protocol': {
            'test_subject_isolation': True, 'test_history': False, 'window_position_schedule_r_t': False,
            'test_subject_calibration': False, 'per_window_only': True, 'fusion_deployable': True,
            'outer_cv': 'StratifiedGroupKFold(5, shuffle=True, random_state=0) by subject = highdim fold_id (seed_noise_band_oof.npz)',
            'inner_cv': f'StratifiedGroupKFold({N_INNER}, shuffle=True, random_state=outer_fold) on outer-train subjects, shared by all members',
            'alignment': 'database windows mapped 1:1 to highdim windows by multiset (subject, label, md5(raw float32)); '
                         'C4 members retrained in database order with original preprocessing, then reindexed to highdim order',
            'z_norm': 'per member, per outer fold: test logit standardized with mean/std of that member\'s inner OOF logits on the outer-train fold',
            'weights_source': 'eq = fixed equal weights; stack = non-negative logistic regression (free intercept, L-BFGS-B) fit on inner OOF z-scores of the outer-train fold, normalized to sum 1',
            'resnet_fallback': 'ResNet1d has no inner OOF / train-fold scores: enters with fixed equal share 1/n_members on raw logit of saved outer OOF (seed mean), no z-norm',
            'reused_outer_oof': {'hand_mlp': 'seed_noise_band_oof.npz oof_seed_mean', 'bb_lr': 'broadband_shape_window6_oof.npz logreg__all_new',
                                 'resnet': 'raw_waveform_window6_oof.npz resnet1d_seed_mean'},
            'inner_retrained': ['base_hgb', 'w05_lr', 'w05_hgb', 'svc_base', 'hand_mlp (5 seeds, early-stop split = first split of SGKF(5,rs=100*fold+inner))', 'bb_lr', 'wbb_lr'],
            'threshold_source': 'bacc_at0: fixed 0 on fused z/logit score; bacc_train_thr: Youden (max TPR-FPR) threshold on outer-train inner OOF fused score',
            'bootstrap': f'subject-level, {N_BOOT} draws, seed {BOOT_SEED}, shared draws for all combos',
            'nothing_selected_on_test_folds': True},
        'reproduction': repro,
        'z_stats': zstats,
        'combos': info,
        'meta': {'elapsed_s': elapsed, 'device_mlp': DEVICE, 'cpu_jobs': N_JOBS,
                 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                 'torch': str(torch.__version__), 'n_windows': int(len(y)), 'n_subjects': int(len(subs))},
    }
    RES_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    log(f'[done] {elapsed:.0f}s -> {RES_PATH.name}, {OOF_PATH.name}')


if __name__ == '__main__':
    main()
