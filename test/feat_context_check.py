"""复核 feat_context 的 qs3 增益：换划分种子、细扫窗长、受试者内打乱对照。

- 种子 1/2：qs3 是从 k∈{3,5,8} 中选出来的，换划分检验是否稳定优于逐窗特征 q。
- k=2/4：细扫，确认 3 不是孤立尖峰。
- 打乱对照：受试者内随机打乱窗口顺序后再做同样的因果平均。若增益来自状态连续性，
  打乱后应退回 q 水平；若打乱后仍高，说明是平均本身带来的某种伪影。
"""
import json, sys, time
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from b_calib_variants import subject_quantile
from singlem_eval import smooth, boot
from feat_context import feat_smooth, metrics, K


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
    Z = subject_quantile(base, g, K)
    gb = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_depth=3,
                                                l2_regularization=1., random_state=0)
    res = {}

    def run(F, tag, seed, yy=y, gg=g, mm=mask):
        oof = np.full(len(yy), np.nan)
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(F, yy, gg):
            oof[te] = gb().fit(F[tr], yy[tr]).predict_proba(F[te])[:, 1]
        v = smooth(oof, gg, 5)
        res[tag] = metrics(v, yy, mm)
        print(f'[result] {tag:22s} AUC={res[tag]["AUC"]:.4f} bACC={res[tag]["bACC"]:.4f} '
              f'({time.perf_counter()-tick:.0f}s)', flush=True)
        (OUT/'feat_context_check.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
        return v

    if 'pairs' in sys.argv:
        # 漂移诊断：每人 6 块固定 R,T,R,T,R,T。若增益来自"越早越像 relax"的单调漂移，
        # 则在 R→T 相邻块对上 AUC 升高、在 T→R 相邻块对上降低；真实状态信息应两个方向都升。
        z = np.load(OUT/'feat_context_oof.npz')
        pool = {}
        for f in ('b_branch_oof.npz', 'b_calib_variants_oof.npz', 'b_calib2_oof.npz'):
            q = np.load(OUT/f); pool.update({k: q[k] for k in q.files if k not in ('y', 'groups')})
        from scipy.stats import rankdata
        p3 = np.mean([rankdata(pool[k])/len(y) for k in
                      ('quant120_histgb', 'quant120w_histgb', 'quant120_svc', 'base_svc')], 0)
        blk = np.empty(len(y), int)
        for s in np.unique(g):
            i = np.where(g == s)[0]; blk[i] = np.r_[0, np.cumsum(np.diff(y[i]) != 0)]
        from sklearn.metrics import roc_auc_score
        out = {}
        for name, s0 in (('P3', p3), ('q_histgb', z['q_histgb']), ('qs3_histgb', z['qs3_histgb'])):
            v = smooth(s0, g, 5)
            for direction in ('R->T', 'T->R'):
                aucs = []
                for s in np.unique(g):
                    for b in range(5):
                        first = y[(g == s) & (blk == b)]
                        if len(first) == 0 or (direction == 'R->T') != (first[0] == 0):
                            continue
                        sel = (g == s) & ((blk == b) | (blk == b+1)) & mask
                        if len(set(y[sel])) == 2:
                            aucs.append(roc_auc_score(y[sel], v[sel]))
                out[f'{name}_{direction}'] = {'mean_AUC': float(np.mean(aucs)), 'n': len(aucs)}
                print(f'[pairs] {name:11s} {direction} 受试者内相邻块 AUC={np.mean(aucs):.4f} (n={len(aucs)})',
                      flush=True)
        res.update(json.loads((OUT/'feat_context_check.json').read_text(encoding='utf-8')))
        res['drift_pairs'] = out
        (OUT/'feat_context_check.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
        return

    if 'extra' in sys.argv:
        # qs2 的多种子复核 + A 支（不做受试者校准）上的同一做法
        prev = json.loads((OUT/'feat_context_check.json').read_text(encoding='utf-8'))
        res.update(prev)
        for seed in (1, 2):
            run(feat_smooth(Z, g, 2), f'seed{seed}_qs2', seed)
        allm = np.ones(len(y), bool)
        run(base, 'A_raw', 0, mm=mask)
        run(feat_smooth(base, g, 3), 'A_s3', 0, mm=mask)
        run(base, 'A_raw_allwin', 0, mm=allm)
        run(feat_smooth(base, g, 3), 'A_s3_allwin', 0, mm=allm)
        print('[done]', flush=True)
        return

    for seed in (1, 2):
        vq = run(Z, f'seed{seed}_q', seed)
        vs = run(feat_smooth(Z, g, 3), f'seed{seed}_qs3', seed)
        b = boot(vs, vq, y, g, mask)
        res[f'seed{seed}_qs3_vs_q'] = b
        print(f'[boot] seed{seed} qs3-q Δ={b["mean"]:+.4f} [{b["p2.5"]:+.4f},{b["p97.5"]:+.4f}]', flush=True)
    for k in (2, 4):
        run(feat_smooth(Z, g, k), f'seed0_qs{k}', 0)

    # 打乱对照：受试者内重排（含校准段之外的全部窗口），在打乱后的序列上做相同流程
    rng = np.random.default_rng(0)
    perm = np.concatenate([rng.permutation(np.where(g == s)[0]) for s in np.unique(g)])
    Zp, yp, gp, mp = Z[perm], y[perm], g[perm], mask[perm]
    run(Zp, 'shuffle_q', 0, yp, gp, mp)
    run(feat_smooth(Zp, gp, 3), 'shuffle_qs3', 0, yp, gp, mp)
    res['seconds'] = time.perf_counter()-tick
    (OUT/'feat_context_check.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[done]', flush=True)


if __name__ == '__main__':
    main()
