"""去漂移特征层上下文：先减受试者内长时因果均值（跨 ≥2 块，抵消块序漂移），再做短时因果平均。

判据：池化 AUC 之外，必须报告方向对称 AUC（R->T 与 T->R 相邻块对的受试者内 AUC），
两个方向都不低于对照 q 才算真实增益（见报告 4.15）。
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from b_calib_variants import subject_quantile
from singlem_eval import smooth, boot
from feat_context import feat_smooth, metrics, K


def direction_auc(v, y, g, mask):
    """受试者内相邻块对 AUC，按首块标签分 R->T / T->R。"""
    blk = np.empty(len(y), int)
    for s in np.unique(g):
        i = np.where(g == s)[0]; blk[i] = np.r_[0, np.cumsum(np.diff(y[i]) != 0)]
    out = {'R->T': [], 'T->R': []}
    for s in np.unique(g):
        for b in range(blk[g == s].max()):
            first = y[(g == s) & (blk == b)][0]
            sel = (g == s) & ((blk == b) | (blk == b+1)) & mask
            if len(set(y[sel])) == 2:
                out['R->T' if first == 0 else 'T->R'].append(roc_auc_score(y[sel], v[sel]))
    return {k: float(np.mean(a)) for k, a in out.items()}


def detrend(Z, g, L):
    """减去过去 L 窗的因果均值（含当前窗）。"""
    return Z - feat_smooth(Z, g, L)


def main():
    tick = time.perf_counter()
    d = np.load(OUT/'highdim_deep_features.npz')
    X0, y, g = d['X'].astype(np.float64), d['y'].astype(int), d['groups']
    E = np.load(OUT/'emg_features_v2.npz')['E'].astype(np.float64)
    base = np.hstack([X0[:, X0.std(0) > 1e-8], E])
    pos = np.empty(len(y), int)
    for s in np.unique(g):
        i = np.where(g == s)[0]; pos[i] = np.arange(len(i))
    mask = pos >= K
    blen = [np.diff(np.flatnonzero(np.r_[1, np.diff(y[g == s]) != 0, 1])) for s in np.unique(g)]
    print(f'[info] 块长中位数={np.median(np.concatenate(blen)):.0f} 窗', flush=True)
    Z = subject_quantile(base, g, K)
    gb = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_depth=3,
                                                l2_regularization=1., random_state=0)
    res, oofs = {}, {}

    def report(v, tag):
        r = metrics(v, y, mask); r.update(direction_auc(v, y, g, mask)); res[tag] = r
        print(f'[result] {tag:16s} AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f} '
              f'R->T={r["R->T"]:.4f} T->R={r["T->R"]:.4f} ({time.perf_counter()-tick:.0f}s)', flush=True)
        (OUT/'detrend_context_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                                        encoding='utf-8')

    def run(F, tag):
        oof = np.full(len(y), np.nan)
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(F, y, g):
            oof[te] = gb().fit(F[tr], y[tr]).predict_proba(F[te])[:, 1]
        oofs[tag] = oof
        v = smooth(oof, g, 5); report(v, tag)
        return v

    if 'check' in sys.argv:
        return check(Z, y, g, pos, gb, tick)
    if 'ens' in sys.argv:
        return ensemble(Z, y, g, mask, gb, tick)

    pool = {}
    for f in ('b_branch_oof.npz', 'b_calib_variants_oof.npz', 'b_calib2_oof.npz'):
        q = np.load(OUT/f); pool.update({k: q[k] for k in q.files if k not in ('y', 'groups')})
    p3 = smooth(np.mean([rankdata(pool[k])/len(y) for k in
                         ('quant120_histgb', 'quant120w_histgb', 'quant120_svc', 'base_svc')], 0), g, 5)
    report(p3, 'P3')
    vq = run(Z, 'q')
    best = None
    for L in (100, 200):
        D = detrend(Z, g, L)
        run(D, f'd{L}')
        v = run(feat_smooth(D, g, 3), f'd{L}s3')
        run(np.hstack([Z, feat_smooth(D, g, 3)]), f'q+d{L}s3')
        r = res[f'd{L}s3']
        if best is None or min(r['R->T'], r['T->R']) > min(res[best]['R->T'], res[best]['T->R']):
            best = f'd{L}s3'
    for tag in [t for t in res if t not in ('P3', 'q')]:
        b = boot(smooth(oofs[tag], g, 5), vq, y, g, mask)
        res[tag]['vs_q'] = b
        print(f'[boot] {tag:10s}-q Δ={b["mean"]:+.4f} [{b["p2.5"]:+.4f},{b["p97.5"]:+.4f}]', flush=True)
    mix = np.mean([rankdata(p3), rankdata(smooth(oofs[best], g, 5))], 0)/len(y)
    report(mix, f'P3+{best}')
    res[f'P3+{best}']['vs_P3'] = b = boot(mix, p3, y, g, mask)
    print(f'[boot] P3+{best}-P3 Δ={b["mean"]:+.4f} [{b["p2.5"]:+.4f},{b["p97.5"]:+.4f}]', flush=True)
    res['seconds'] = time.perf_counter()-tick
    (OUT/'detrend_context_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[done]', flush=True)


def ensemble(Z, y, g, mask, gb, tick):
    """去趋势窗长多样化 + 模型多样化的 rank 平均。成员均固定超参，不按结果挑选。"""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    path = OUT/'detrend_context_results.json'
    res = json.loads(path.read_text(encoding='utf-8'))
    ens = res.setdefault('ensemble', {})
    svc = lambda: make_pipeline(StandardScaler(), SVC(C=1, gamma='scale', cache_size=4096))
    members = {}

    def run(F, tag, mk):
        oof = np.full(len(y), np.nan)
        for f, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(F, y, g)):
            m = mk().fit(F[tr], y[tr])
            oof[te] = m.predict_proba(F[te])[:, 1] if hasattr(m, 'predict_proba') else m.decision_function(F[te])
            print(f'  {tag} fold{f} ({time.perf_counter()-tick:.0f}s)', flush=True)
        members[tag] = v = smooth(oof, g, 5)
        r = metrics(v, y, mask); r.update(direction_auc(v, y, g, mask)); ens[tag] = r
        print(f'[ens] {tag:14s} AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f} R->T={r["R->T"]:.4f} '
              f'T->R={r["T->R"]:.4f}', flush=True)
        path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')

    for L in (150, 200, 300):
        run(detrend(Z, g, L), f'd{L}_histgb', gb)
    run(detrend(Z, g, 10**6), 'dinf_histgb', gb)
    run(detrend(Z, g, 200), 'd200_svc', svc)
    combos = {'histgb_all': [k for k in members if k.endswith('histgb')],
              'all': list(members)}
    vq_oof = np.full(len(y), np.nan)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Z, y, g):
        vq_oof[te] = gb().fit(Z[tr], y[tr]).predict_proba(Z[te])[:, 1]
    vq = smooth(vq_oof, g, 5)
    for name, ks in combos.items():
        v = np.mean([rankdata(members[k]) for k in ks], 0)/len(y)
        r = metrics(v, y, mask); r.update(direction_auc(v, y, g, mask))
        r['members'] = ks; r['vs_q'] = boot(v, vq, y, g, mask); ens[name] = r
        print(f'[ens] {name:14s} AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f} R->T={r["R->T"]:.4f} '
              f'T->R={r["T->R"]:.4f} Δq={r["vs_q"]["mean"]:+.4f}', flush=True)
    path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[done]', flush=True)


def check(Z, y, g, pos, gb, tick):
    """泄漏复核：
    - seed1/2：换划分；
    - pos>=200：只评估去趋势窗口已完全由真实历史填满的窗；
    - dinf：扩展均值（全部过去），无窗长超参；
    - 块序打乱：每人把整块随机重排后再算特征与标签，打破固定 R,T,R,T 时间表。
      若增益来自时间表/漂移对齐，打乱后应消失；若来自"相对自身历史的状态对比"，应保留。
    """
    path = OUT/'detrend_context_results.json'
    res = json.loads(path.read_text(encoding='utf-8'))
    chk = res.setdefault('check', {})

    def run(F, yy, gg, mm, tag, seed=0):
        oof = np.full(len(yy), np.nan)
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(F, yy, gg):
            oof[te] = gb().fit(F[tr], yy[tr]).predict_proba(F[te])[:, 1]
        v = smooth(oof, gg, 5)
        r = metrics(v, yy, mm); r.update(direction_auc(v, yy, gg, mm)); chk[tag] = r
        print(f'[check] {tag:18s} AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f} R->T={r["R->T"]:.4f} '
              f'T->R={r["T->R"]:.4f} ({time.perf_counter()-tick:.0f}s)', flush=True)
        path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
        return v

    mask = pos >= K
    D200 = detrend(Z, g, 200)
    quick = 'balance' in sys.argv or 'samelabel' in sys.argv
    for seed in (1, 2) if not quick else ():
        vq = run(Z, y, g, mask, f'seed{seed}_q', seed)
        vd = run(D200, y, g, mask, f'seed{seed}_d200', seed)
        b = boot(vd, vq, y, g, mask); chk[f'seed{seed}_d200_vs_q'] = b
        print(f'[boot] seed{seed} d200-q Δ={b["mean"]:+.4f} [{b["p2.5"]:+.4f},{b["p97.5"]:+.4f}]', flush=True)
    if not quick:
        m200 = pos >= 200
        run(Z, y, g, m200, 'pos200_q'); run(D200, y, g, m200, 'pos200_d200')
        run(detrend(Z, g, 10**6), y, g, mask, 'dinf')

    # 块序打乱：特征按原始顺序做 subject_quantile 前的 base 不可得，这里对 Z 重排后
    # 重新计算去趋势（Z 的分位校准只用前 120 窗，重排后的校准段变化影响很小，作为保守近似）
    rng = np.random.default_rng(0)
    perm = []
    for s in np.unique(g):
        i = np.where(g == s)[0]
        cuts = np.flatnonzero(np.r_[1, np.diff(y[i]) != 0])
        blocks = np.split(i, cuts[1:])
        perm += [blocks[j] for j in rng.permutation(len(blocks))]
    perm = np.concatenate(perm)
    yp, gp, Zp = y[perm], g[perm], Z[perm]
    pp = np.empty(len(yp), int)
    for s in np.unique(gp):
        i = np.where(gp == s)[0]; pp[i] = np.arange(len(i))
    mp = pp >= K
    if 'samelabel' in sys.argv:
        # 同标签块内重排：保留 R,T 交替与"历史含两类"的性质，但打乱块的真实时间位置
        # （R1/R2/R3 互换、T1/T2/T3 互换）。块间拼接处的漂移不连续与随机块序打乱相同。
        # 若 d200 在此仍显著优于 q，增益依赖的是交替结构（赛制假设），而非单调漂移；
        # 若退回 q 水平，则说明增益依赖块的真实时间位置，即漂移/位置混杂。
        for rep in range(3):
            rng = np.random.default_rng(rep)
            perm = []
            for s in np.unique(g):
                i = np.where(g == s)[0]
                cuts = np.flatnonzero(np.r_[1, np.diff(y[i]) != 0])
                blocks = np.split(i, cuts[1:])
                lab = [y[b[0]] for b in blocks]
                pools = {c: list(rng.permutation([j for j in range(len(blocks)) if lab[j] == c])) for c in (0, 1)}
                perm += [blocks[pools[lab[j]].pop()] for j in range(len(blocks))]
            perm = np.concatenate(perm)
            ys, gs, Zs = y[perm], g[perm], Z[perm]
            vq = run(Zs, ys, gs, mask, f'samelabel{rep}_q')
            vd = run(detrend(Zs, gs, 200), ys, gs, mask, f'samelabel{rep}_d200')
            b = boot(vd, vq, ys, gs, mask); chk[f'samelabel{rep}_d200_vs_q'] = b
            print(f'[boot] samelabel{rep} d200-q Δ={b["mean"]:+.4f} [{b["p2.5"]:+.4f},{b["p97.5"]:+.4f}]',
                  flush=True)
        path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
        print('[done]', flush=True)
        return

    if 'balance' not in sys.argv:
        run(Zp, yp, gp, mp, 'blockshuf_q')
        run(detrend(Zp, gp, 200), yp, gp, mp, 'blockshuf_d200')
        print('[done]', flush=True)
        return

    # 分层诊断（标签只用于划分评估子集，不进特征）：按过去 200 窗中 task 占比分层。
    # 若 d200 的增益来自"相对平衡历史的对比"，则在打乱数据的平衡历史子集上也应胜过 q。
    def tfrac(yy, gg):
        f = np.empty(len(yy))
        for s in np.unique(gg):
            i = np.where(gg == s)[0]
            f[i] = feat_smooth(yy[i, None].astype(float), np.zeros(len(i)), 200)[:, 0]
        return f
    bal = {}
    for name, (F0, yy, gg, mm) in {'orig': (Z, y, g, mask), 'blockshuf': (Zp, yp, gp, mp)}.items():
        tf = tfrac(yy, gg)
        vq = run(F0, yy, gg, mm, f'{name}_q_tmp')
        vd = run(detrend(F0, gg, 200), yy, gg, mm, f'{name}_d200_tmp')
        for lo, hi in ((0, .35), (.35, .65), (.65, 1.01)):
            sub = mm & (tf >= lo) & (tf < hi)
            if len(set(yy[sub])) < 2:
                continue
            r = {'n': int(sub.sum()), 'q': metrics(vq, yy, sub)['AUC'], 'd200': metrics(vd, yy, sub)['AUC']}
            bal[f'{name}_tfrac[{lo},{hi})'] = r
            print(f'[balance] {name:9s} T占比[{lo:.2f},{hi:.2f}) n={r["n"]:5d} '
                  f'q={r["q"]:.4f} d200={r["d200"]:.4f}', flush=True)
        chk.pop(f'{name}_q_tmp'); chk.pop(f'{name}_d200_tmp')
    chk['history_balance'] = bal
    path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[done]', flush=True)


if __name__ == '__main__':
    main()
