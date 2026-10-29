"""利用协议不变量（6 块、严格 R-T-R-T-R-T、首块必为 relax）做因果的受试者极性/偏置校正。

注意：本脚本用到协议先验推出的块标签，属于灰色地带（相当于部分测试标签由实验设计确定），
结果单独标注，不覆盖任何既有正式结果。
"""
import json, sys, time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT/'test'
sys.path[:0] = [str(ROOT/'src'), str(OUT)]
from causal_history import rebuild_meta, report

K_FIRST = 30            # 首块最短 217s -> 前 30 窗（180s）一定仍在块 1 内


def main():
    tick = time.perf_counter()
    F, P = np.load(OUT/'highdim_deep_features.npz'), np.load(OUT/'highdim_deep_oof.npz')
    X, y, groups = F['X'], F['y'], F['groups']
    meta = rebuild_meta()
    assert len(meta) == len(y) and all(m[0] == g and m[3] == t for m, g, t in zip(meta, groups, y))
    trial = np.array([f'{m[0]}#{m[1]}' for m in meta]); end = np.array([m[2] for m in meta])
    subjects = np.unique(groups)
    streams = {s: np.where(groups == s)[0][np.argsort(end[groups == s])] for s in subjects}
    blocks = {s: [np.where(trial == f'{s}#{i}')[0][np.argsort(end[trial == f"{s}#{i}"])] for i in range(6)] for s in subjects}
    # 协议先验标签：块 i 的标签 = i%2==0 ? relax(0) : task(1)
    block_label = {i: i % 2 for i in range(6)}
    print(f'[meta] {len(y)} windows; protocol labels added, {time.perf_counter()-tick:.1f}s', flush=True)

    res = {}
    for name in ('large_mlp300', 'logistic300'):
        p = np.clip(P[name], 1e-6, 1-1e-6); z = np.log(p/(1-p))
        R = {'baseline 全流': report(y, z, groups, 0.)}
        # 1) oracle 极性（用真标签，仅上限参考）
        zo = z.copy()
        for s in subjects:
            ids = streams[s]
            if roc_auc_score(y[ids], z[ids]) < .5:
                zo[ids] = -zo[ids]
        R['oracle 极性翻转 (用真标签)'] = report(y, zo, groups, 0.)
        # 2) 协议先验极性：首块必为 relax，用前 K_FIRST 个窗判断符号
        sign = {}
        for s in subjects:
            ids = streams[s][:K_FIRST]
            sign[s] = -1. if z[ids].mean() > 0 else 1.
        zp = np.empty_like(z)
        for s in subjects:
            zp[streams[s]] = sign[s]*z[streams[s]]
        R[f'协议先验极性 (前{K_FIRST}窗)'] = report(y, zp, groups, 0.)
        R[f'协议先验极性 flipped 受试者数'] = float(sum(v < 0 for v in sign.values()))
        # 3) 协议先验 + 因果在线截距校正：用已过去的、协议标签已知的整块估计偏置
        #    逐块滚动：预测块 k 时，只用块 1..k-1 的 (z, 协议标签) 拟合 1 维 logistic
        zc = z.copy()
        for s in subjects:
            for k in range(2, 6):
                past = np.concatenate([blocks[s][i] for i in range(k)])
                tgt = np.concatenate([np.full(len(blocks[s][i]), block_label[i]) for i in range(k)])
                if len(np.unique(tgt)) < 2:
                    continue
                lr = LogisticRegression(C=1e3, max_iter=1000).fit(z[past, None], tgt)
                zc[blocks[s][k]] = lr.decision_function(z[blocks[s][k], None])
        R['协议先验 + 因果截距校正 (块2-6)'] = report(np.concatenate([y[blocks[s][k]] for s in subjects for k in range(2, 6)]),
                                                     np.concatenate([zc[blocks[s][k]] for s in subjects for k in range(2, 6)]),
                                                     np.concatenate([np.full(len(blocks[s][k]), s) for s in subjects for k in range(2, 6)]), 0.)
        R['基线 同窗口集 (块2-6)'] = report(np.concatenate([y[blocks[s][k]] for s in subjects for k in range(2, 6)]),
                                            np.concatenate([z[blocks[s][k]] for s in subjects for k in range(2, 6)]),
                                            np.concatenate([np.full(len(blocks[s][k]), s) for s in subjects for k in range(2, 6)]), 0.)
        # 4) 每受试者用过去块的协议标签重训轻量模型（10 个频带均值 + 跨被试分数）
        R['每受试者过去块重训 (块2-6, 真边界)'] = past_block_model(X, y, groups, z, subjects, blocks, block_label, name)
        res[name] = R
        for k, v in R.items():
            if isinstance(v, dict):
                print(f'[{name}] {k:36s} AUC={v["AUC"]:.4f} bACC={v["bACC"]:.4f} subjAUC={v["subject_AUC"]:.4f}', flush=True)
            else:
                print(f'[{name}] {k}: {v}', flush=True)

    res['meta'] = {'k_first': K_FIRST, 'seconds': time.perf_counter()-tick}
    (OUT/'protocol_prior_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[done] {time.perf_counter()-tick:.1f}s', flush=True)


def past_block_model(X, y, groups, z, subjects, blocks, block_label, tag):
    """块 k 的预测只用块 1..k-1（协议标签）训练；特征=10 个 STFT 频带均值 + 跨被试 logit。"""
    names = [f'stft_log_power_{b}_mean' for b in ('1_4', '4_8', '8_13', '13_30', '30_45')] + \
            [f'stft_relative_{b}_mean' for b in ('1_4', '4_8', '8_13', '13_30', '30_45')]
    import json as _json
    meta = _json.loads((OUT/'highdim_deep_results.json').read_text(encoding='utf-8'))
    idx = [meta['feature_names'].index(n) for n in names]
    F = np.column_stack([X[:, idx], z])
    out_y, out_s, out_g = [], [], []
    for s in subjects:
        for k in range(1, 6):
            tr = np.concatenate([blocks[s][i] for i in range(k)])
            tgt = np.concatenate([np.full(len(blocks[s][i]), block_label[i]) for i in range(k)])
            if len(np.unique(tgt)) < 2:
                continue
            mu, sd = F[tr].mean(0), F[tr].std(0)+1e-12
            lr = LogisticRegression(C=.05, max_iter=2000, class_weight='balanced').fit((F[tr]-mu)/sd, tgt)
            te = blocks[s][k]
            sc = lr.decision_function((F[te]-mu)/sd)
            out_y.append(y[te]); out_s.append(sc); out_g.append(np.full(len(te), s))
    return report(np.concatenate(out_y), np.concatenate(out_s), np.concatenate(out_g), 0.)


if __name__ == '__main__':
    main()
