"""块边缘过渡窗消融：每个 relax/task 块开头/结尾若干秒可能混入指令与状态切换瞬态。

复用 highdim_deep_features.npz 的 858 维特征与窗口时间元数据，只改变参与训练/评估的窗口集合。
"""
import json, sys, time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT/'test'
sys.path[:0] = [str(ROOT/'src'), str(OUT)]
from causal_history import rebuild_meta, report, running_stats
from highdim_deep_ablation import FeatureTransform


def edge_table(meta):
    """每个窗口距本块起点/终点的秒数。"""
    import collections
    by = collections.defaultdict(list)
    for i, m in enumerate(meta):
        by[(m[0], m[1])].append(i)
    pos, remain = np.zeros(len(meta)), np.zeros(len(meta))
    for ids in by.values():
        ids = np.array(ids); start = meta[ids[0]][2] - 6; stop = meta[ids[-1]][2]
        pos[ids] = meta[ids[0]][2] - start + (np.arange(len(ids)))*6
        remain[ids] = stop - np.array([meta[i][2] for i in ids])
    return pos, remain


def run(X, y, groups, tag):
    oof = np.full(len(y), np.nan)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, groups):
        t = FeatureTransform(300, 'kbest').fit(X[tr], y[tr])
        clf = LogisticRegression(C=1, max_iter=2000, class_weight='balanced', random_state=0).fit(t.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(t.transform(X[te]))[:, 1]
    r = report(y, oof, groups, .5)
    print(f'[{tag}] n={len(y)} AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f} subjAUC={r["subject_AUC"]:.4f} subjBACC={r["subject_bACC"]:.4f}', flush=True)
    return r, oof


def main():
    tick = time.perf_counter()
    F, P = np.load(OUT/'highdim_deep_features.npz'), np.load(OUT/'highdim_deep_oof.npz')
    X, y, groups = F['X'], F['y'], F['groups']
    meta = rebuild_meta()
    pos, remain = edge_table(meta)
    end = np.array([m[2] for m in meta])
    print(f'[meta] pos p5={np.percentile(pos,5):.0f}s remain p5={np.percentile(remain,5):.0f}s', flush=True)

    results = {}
    for e in (0, 15, 30, 45, 60, 90):
        keep = (pos >= e) & (remain >= e)
        Xk, yk, gk = X[keep], y[keep], groups[keep]
        results[f'edge{e}'], _ = run(Xk, yk, gk, f'edge{e}s (只在保留窗上评估)')
        results[f'edge{e}']['n_windows'] = int(keep.sum())
        # 可部署形式：训练只用块中心窗，推理对全部窗预测（测试时无需知道块边界）
        oof = np.full(len(y), np.nan)
        for tr_all, te_all in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, groups):
            tr = tr_all[keep[tr_all]]
            t = FeatureTransform(300, 'kbest').fit(X[tr], y[tr])
            clf = LogisticRegression(C=1, max_iter=2000, class_weight='balanced', random_state=0).fit(t.transform(X[tr]), y[tr])
            oof[te_all] = clf.predict_proba(t.transform(X[te_all]))[:, 1]
        results[f'edge{e}_evalall'] = report(y, oof, groups, .5)
        r = results[f'edge{e}_evalall']
        print(f'[edge{e}s 训练过滤/全窗评估] AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f} subjAUC={r["subject_AUC"]:.4f}', flush=True)

    # 边缘过滤 + 在线特征标准化（合规自适应）：训练用块中心窗，推理对全部窗
    e = 45
    keep = (pos >= e) & (remain >= e)
    streams = [np.where(groups == s)[0][np.argsort(end[groups == s])] for s in np.unique(groups)]
    oof = np.full(len(y), np.nan)
    for fold, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, groups), 1):
        mu0, var0 = X[tr].mean(0), X[tr].var(0)+1e-20
        Xn = np.empty_like(X, dtype=float)
        for ids in streams:
            m, s = running_stats(X[ids].astype(float), 5, mu0, var0); Xn[ids] = (X[ids]-m)/s
        trk = tr[keep[tr]]
        t = FeatureTransform(300, 'kbest').fit(Xn[trk], y[trk])
        clf = LogisticRegression(C=1, max_iter=2000, class_weight='balanced', random_state=0).fit(t.transform(Xn[trk]), y[trk])
        oof[te] = clf.predict_proba(t.transform(Xn[te]))[:, 1]
        print(f'[edge{e}+online fold {fold}/5] test AUC={roc_auc_score(y[te], oof[te]):.4f} elapsed={time.perf_counter()-tick:.0f}s', flush=True)
    results['edge45_online_evalall'] = report(y, oof, groups, .5)
    print(f'[edge45+online 全窗评估] AUC={results["edge45_online_evalall"]["AUC"]:.4f} '
          f'bACC={results["edge45_online_evalall"]["bACC"]:.4f} subjAUC={results["edge45_online_evalall"]["subject_AUC"]:.4f}', flush=True)

    results['meta'] = {'seconds': time.perf_counter()-tick}
    (OUT/'edge_ablation_results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[done] {time.perf_counter()-tick:.1f}s', flush=True)


if __name__ == '__main__':
    main()
