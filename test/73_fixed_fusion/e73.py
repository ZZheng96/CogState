"""E73 fixed-weight cache audit and explicitly limited comparisons."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import zipfile
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
TESTS = HERE.parent
spec = importlib.util.spec_from_file_location('e67_e73', TESTS / '67_mlp_loso/e67.py')
E67 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E67)
PARTS = ('full', 'early', 'late')
METRICS = ('auc', 'f1', 'bacc')


def save(name, obj):
    (HERE / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def summarize(rows):
    names = list(rows[0]['metrics'])
    aggregate, counts = {}, {}
    for name in names:
        aggregate[name], counts[name] = {}, {}
        for part in PARTS:
            aggregate[name][part], counts[name][part] = {}, {}
            for metric in METRICS:
                vals = [r['metrics'][name][part][metric] for r in rows if r['metrics'][name][part][metric] is not None]
                aggregate[name][part][metric] = {'macro': float(np.mean(vals)) if vals else None, 'n_subjects': len(vals)}
                delta = [r['metrics'][name][part][metric] - r['metrics']['w0.50'][part][metric] for r in rows if r['metrics'][name][part][metric] is not None and r['metrics']['w0.50'][part][metric] is not None]
                counts[name][part][metric] = {'improved': sum(d > 0 for d in delta), 'regressed': sum(d < 0 for d in delta), 'equal': sum(d == 0 for d in delta), 'undefined': len(rows)-len(delta)}
    for row in rows:
        row['delta_vs_baseline'] = {name: {part: {m: row['metrics'][name][part][m]-row['metrics']['w0.50'][part][m] if row['metrics'][name][part][m] is not None and row['metrics']['w0.50'][part][m] is not None else None for m in METRICS} for part in PARTS} for name in names}
    return {'aggregate': aggregate, 'relative_to_baseline_counts': counts, 'per_subject': rows}


def main():
    inventory = []
    scanned = 0
    for p in sorted(TESTS.rglob('*.npz')):
        scanned += 1
        with zipfile.ZipFile(p) as z:
            keys = [k[:-4] for k in z.namelist()]
        component_keys = [k for k in keys if 'lr_scaled' in k or 'mlp_scaled' in k or k in ('std_lr', 'std_mlp')]
        if component_keys:
            inventory.append({'path': str(p), 'component_keys': component_keys})
    e67dir = TESTS / '67_mlp_loso'
    checks = json.loads((e67dir / 'protocol.json').read_text(encoding='utf-8'))['data_checks']
    with np.load(TESTS / '53_e17_post_loso/e53_scores.npz') as e53:
        subs = list(e53['subjects'].astype(str))
        assert len(subs) == 57
        for q in subs:
            source = Path(checks[q]['source'])
            assert hashlib.sha256(source.read_bytes()).hexdigest() == checks[q]['source_sha256']
        diagnostic, full, evidence = [], [], []
        for i, q in enumerate(subs, 1):
            tr = [x for x in subs if x != q]
            with np.load(e67dir / f'fold_{i:02d}_{q}.npz') as z:
                assert str(z['target']) == q and list(z['train_subjects'].astype(str)) == tr
                with np.load(checks[q]['source']) as source:
                    order = np.argsort(source['t'])
                    for key, original in (('t', 't'), ('y', 'y'), ('block', 'blk')):
                        np.testing.assert_array_equal(z[key], source[original][order])
                np.testing.assert_array_equal(z['baseline_raw'], e53[q])
                np.testing.assert_array_equal(z['raw'], 2*z['mlp_scaled'])
                assert np.isfinite(z['mlp_scaled']).all()
                # Recompute unchanged postprocessing from verified raw caches.
                scores = {'w0.50': E67.stage(e53[q])['dt1200_hmm300'], 'w0.00': E67.stage(z['raw'])['dt1200_hmm300']}
                full.append({'target': q, 'metrics': {name: E67.evaluate(z['y'], z['block'], score) for name, score in scores.items()}})
                cp = TESTS / '63_layer_ablation' / f'fold_{q}_components.npz'
                if cp.exists():
                    with np.load(cp) as c:
                        assert str(c['target']) == q and list(c['train_subjects'].astype(str)) == tr
                        assert json.loads(str(c['data_hashes'])) == {s: checks[s]['source_sha256'] for s in subs}
                        for key, original in (('t', 't'), ('y', 'y'), ('block', 'block')):
                            np.testing.assert_array_equal(c[f'{original}_{q}'], z[key])
                        lr, mlp = c[f'lr_scaled_{q}'], c[f'mlp_scaled_{q}']
                        np.testing.assert_array_equal(mlp, z['mlp_scaled'])
                        np.testing.assert_array_equal(c[f'raw_{q}'], e53[q])
                        evidence.append({'target': q, 'e53_vs_e63_raw_max_abs': float(np.max(np.abs(e53[q]-c[f'raw_{q}']))), 'e67_mlp_equal_e63': True})
                        diagnostic.append({'target': q, 'metrics': {f'w{w:.2f}': E67.evaluate(z['y'], z['block'], E67.stage(2*(w*lr+(1-w)*mlp))['dt1200_hmm300']) for w in (0.5, 0.25, 0.)}})
    missing = [q for q in subs if q not in [r['target'] for r in diagnostic]]
    protocol = {'experiment': 'E73_fixed_fusion', 'status': 'blocked_full_57_weight025', 'weights_lr': [0.5, 0.25, 0.0], 'fusion': '2*(w*LR_scaled+(1-w)*MLP_scaled)', 'weight_selection': 'none; fixed before target-label evaluation', 'subjects': subs, 'postprocessing': 'E67.stage: EMA half-life1200 first-score initialization, update then subtract; forward HMM dwell300, temperature1; continuous sequence, no block reset', 'time_unit': 'retained frames, not wall-clock seconds', 'threshold': 0, 'positive_class': 1, 'parts': {'full': 'all', 'early': 'block<=2', 'late': 'block>=4'}, 'macro': 'unweighted subject mean, null AUC/bACC for single-class slices; positive F1 zero_division=0', 'complete57_scope': 'E53 baseline and E67 MLP only; NOT a validated same-fit three-weight experiment', 'diagnostic_scope': 'six E63 held-out targets only; other subject arrays in these files are training predictions, not LOSO', 'missing_verified_lr_subjects': missing, 'recovery_rejected': 'E53 raw minus E67 MLP is unverified in 51 independently retrained folds; six exact matches cannot prove 57-fold component identity', 'execution': 'no training; CPU cache audit and unchanged reference serial postprocessing, no heavy feature/GPU computation', 'alignment_limit': 'cache t/y/block and source hashes checked; no new BDF/event audit'}
    result = {'status': protocol['status'], 'full57_three_weights': {'w0.25': None, 'reason': '51 folds lack verified matched LR/MLP components'}, 'available57_endpoints_not_same_fit': summarize(full), 'six_subject_diagnostic_only': summarize(diagnostic), 'evidence': {'npz_scanned': scanned, 'six_fold_identity': evidence, 'component_cache_inventory': inventory}}
    lines = ['# E73 固定融合：完整比较阻塞，部分结果可核实', '', '**状态：未完成57人三权重比较，不宣称完成或选出最优权重。**', '', 'E67有57折MLP；E63仅004、009、010、018、034、050六个留出折有配套LR/MLP。其他51折缺少经验证的配套LR。E53只保存融合raw。六折E53 raw与E63融合逐元素完全一致，E67 MLP与E63也完全一致，但不能证明其他51折重训MLP与E53原模型一致，故不以E53 raw减E67 MLP构造LR。E63文件内其他56人的输出是该折训练人输出，不能冒充其留出预测。', '', '固定w_LR=0.5/0.25/0；raw=2*(w*LR_scaled+(1-w)*MLP_scaled)。统一EMA1200+前向HMM300，连续序列不按块重置；阈值>0为认知负荷。early为block<=2，late为block>=4。目标标签仅评分，不用于权重选择。', '', '## 可核实的57人端点（不是已验证同次拟合三权重比较）', '0.5使用E53已有融合raw；0使用E67纯MLP。0.25的57人AUC/F1/bACC均未计算，值为null。', '']
    for title, section in [('57人端点', result['available57_endpoints_not_same_fit']), ('六人诊断，禁止外推57人', result['six_subject_diagnostic_only'])]:
        lines += [f'## {title}', '', '|w_LR|范围|宏AUC|宏F1|宏bACC|', '|---|---|---:|---:|---:|']
        for name, parts in section['aggregate'].items():
            for part, vals in parts.items():
                lines.append('|'+name[1:]+'|'+part+'|'+'|'.join(f"{vals[m]['macro']:.6f}" if vals[m]['macro'] is not None else 'NA' for m in METRICS)+'|')
        lines += ['', '相对0.5基线的改善/退步/持平人数（逐指标判断；详细逐人指标与差值见结果JSON）：', '', '|w_LR|范围|AUC 改/退/平|F1 改/退/平|bACC 改/退/平|', '|---|---|---|---|---|']
        for name, parts in section['relative_to_baseline_counts'].items():
            if name == 'w0.50': continue
            for part, vals in parts.items():
                lines.append('|'+name[1:]+'|'+part+'|'+'|'.join('/'.join(str(vals[m][k]) for k in ('improved','regressed','equal')) for m in METRICS)+'|')
        lines.append('')
    lines += ['## 阻塞与边界', '', '完整执行需要补齐其余51折可追溯的同折LR/MLP组件（含目标/训练人、尺度与样本顺序），或独立证明E67重训MLP逐折与E53原MLP一致。本次未重训、未未经验证混用、未改src/原始数据/旧实验；未用目标标签选权重。缓存哈希及时间/标签/块顺序校验不等于重新证明BDF事件对齐。', '', '时间常数沿用旧协议，以保留帧计数；EMA/HMM具有跨帧状态，是否满足赛事“有效感受野不超过6秒”约束仍需单独澄清，本报告不宣称部署合规。', '', '复现：`.venv\\Scripts\\python.exe -B test/73_fixed_fusion/e73.py`。无模型训练；CPU仅处理小型分数序列，复用原参考后处理以避免GPU重写的数值差异。']
    save('protocol.json', protocol)
    save('e73_results.json', result)
    (HERE / 'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'missing_verified_lr_folds': len(missing), 'available57': result['available57_endpoints_not_same_fit']['aggregate'], 'diagnostic6': result['six_subject_diagnostic_only']['aggregate']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
