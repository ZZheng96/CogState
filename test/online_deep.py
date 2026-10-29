"""在线特征标准化（合规）+ 大 MLP / Logistic 重训，并与异构模型做集成。

与 causal_history.py 的区别：那里只对已有 OOF 分数做后处理；这里把因果累计统计直接放进特征，
重训深度模型，并保存 OOF 供集成评估。
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT/'test'
sys.path[:0] = [str(ROOT/'src'), str(OUT)]
from causal_history import rebuild_meta, report, running_stats
from highdim_deep_ablation import FeatureTransform, fit_network, predict, network


def main():
    tick = time.perf_counter()
    F = np.load(OUT/'highdim_deep_features.npz')
    X, y, groups = F['X'], F['y'], F['groups']
    meta = rebuild_meta()
    end = np.array([m[2] for m in meta])
    streams = [np.where(groups == s)[0][np.argsort(end[groups == s])] for s in np.unique(groups)]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    oof = {'online_logistic300': np.full(len(y), np.nan), 'online_mlp300': np.full(len(y), np.nan)}

    for fold, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, groups), 1):
        t0 = time.perf_counter()
        mu0, var0 = X[tr].mean(0), X[tr].var(0)+1e-20
        Xn = np.empty_like(X, dtype=float)
        for ids in streams:
            m, s = running_stats(X[ids].astype(float), 5, mu0, var0); Xn[ids] = (X[ids]-m)/s
        # 内层按受试者隔离，用于早停
        a, b = next(StratifiedGroupKFold(4, shuffle=True, random_state=fold).split(X[tr], y[tr], groups[tr]))
        inner, val = tr[a], tr[b]
        tf = FeatureTransform(300, 'kbest').fit(Xn[inner], y[inner])
        A, V, T = tf.transform(Xn[inner]), tf.transform(Xn[val]), tf.transform(Xn[te])

        lr = LogisticRegression(C=1, max_iter=2000, class_weight='balanced', random_state=0).fit(A, y[inner])
        oof['online_logistic300'][te] = lr.predict_proba(T)[:, 1]

        net, info = fit_network('large', A, y[inner], V, y[val], device, fold)
        oof['online_mlp300'][te] = predict(net, T, device)
        print(f'[fold {fold}/5] logistic={roc_auc_score(y[te], oof["online_logistic300"][te]):.4f} '
              f'mlp={roc_auc_score(y[te], oof["online_mlp300"][te]):.4f} best_epoch={info["best_epoch"]} '
              f'{time.perf_counter()-t0:.0f}s', flush=True)

    results = {}
    for k, v in oof.items():
        results[k] = report(y, v, groups, .5)
        print(f'[{k}] AUC={results[k]["AUC"]:.4f} bACC={results[k]["bACC"]:.4f} '
              f'subjAUC={results[k]["subject_AUC"]:.4f} subjBACC={results[k]["subject_bACC"]:.4f}', flush=True)

    from scipy.stats import rankdata
    import numpy as _np
    r1, r2 = rankdata(oof['online_logistic300'])/len(y), rankdata(oof['online_mlp300'])/len(y)
    results['ensemble_rank'] = report(y, .5*r1+.5*r2, groups, .5)
    print(f'[ensemble_logistic+mlp] AUC={results["ensemble_rank"]["AUC"]:.4f} bACC={results["ensemble_rank"]["bACC"]:.4f} '
          f'subjAUC={results["ensemble_rank"]["subject_AUC"]:.4f}', flush=True)
    results['meta'] = {'device': device, 'seconds': time.perf_counter()-tick,
                       'normalisation': 'causal running mean/std per subject, prior n0=5'}
    np.savez_compressed(OUT/'online_deep_oof.npz', y=y, groups=groups, **oof)
    (OUT/'online_deep_results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[done] {time.perf_counter()-tick:.1f}s on {device}', flush=True)


if __name__ == '__main__':
    main()
