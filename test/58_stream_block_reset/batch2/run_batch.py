from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('batch1', HERE.parent / 'batch1' / 'run_batch.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
np, torch = b.np, b.torch


def persist(result):
    b.save(HERE / 'results.json', result)
    rows = []
    for sid, sub in result['subjects'].items():
        for mode, value in sub['metrics'].items():
            for part, m in value['parts'].items():
                base = sub['metrics']['baseline']['parts'][part]
                rows.append(dict(subject=sid, mode=mode, part=part, **m,
                                 **{'delta_' + k: m[k] - base[k] for k in ('bacc', 'auc', 'f1')}))
    fields = ['subject', 'mode', 'part', 'bacc', 'auc', 'f1', 'n', 'delta_bacc', 'delta_auc', 'delta_f1']
    with (HERE / 'results.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def report(result):
    lines = ['# E58 batch2：真实BDF端到端block重置验证', '',
             '复用batch1的run/evaluate/postprocess；每人完整原始分数仅计算一次（chunk=137），三方案共享原始分数。追加运行保留已完成被试；完整分数保存后可复用，不重复完整推理。', '',
             '基线为dt1200+HMM；keep为每block重置detrend但保留HMM；reset为每block重置detrend+HMM。后段主口径为严格block4/5，另列block4plus。指标以score>0判断task，AUC使用连续分数。', '',
             '## 指标与相对基线变化', '',
             '|被试|方案|范围|n|bACC|AUC|F1|ΔbACC|ΔAUC|ΔF1|',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for sid, sub in result['subjects'].items():
        for mode, value in sub['metrics'].items():
            for part, m in value['parts'].items():
                base = sub['metrics']['baseline']['parts'][part]
                vals = [m[k] for k in ('bacc', 'auc', 'f1')]
                delta = [m[k] - base[k] for k in ('bacc', 'auc', 'f1')]
                lines.append(f"|{sid}|{mode}|{part}|{m['n']}|" + '|'.join(f'{v:.4f}' for v in vals) + '|' + '|'.join(f'{v:+.4f}' for v in delta) + '|')
    lines += ['', '## 运行与前缀一致性', '']
    for sid, sub in result['subjects'].items():
        lines.append(f"- {sid}: 记录{sub['duration_s']:.1f}s，完整推理{sub['inference_seconds']:.1f}s，{sub['n_outputs']}帧；训练名单内={sub['in_training']}；真实trial_start={sub['trial_starts_s']}。")
        for chunk, check in sub['checks'].items():
            lines.append(f"  - 替代chunk={chunk}，前{check['prefix_seconds']}秒新建引擎重新提取特征：时间一致={check['times_identical']}，原始分数一致={check['raw_identical']}，最大差={check['raw_max_abs']}，三后处理一致={all(x['identical'] for x in check['post'].values())}。")
    lines += ['', '## 边界限制与解释', '',
              '- 重置依赖真实事件CSV的trial_start（oracle），不使用固定300秒边界；仅到达边界后第一帧重置。首个trial不额外重置，沿用batch1；不是无标签自动边界检测，也不是可直接部署方案。',
              '- BAD_ACQ_SKIP标记输入为NaN；流中无标签前后段和间隔继续更新状态，但不计入指标；标签窗口按start<t<=end。未进行按边界纯净窗口筛选，与batch1一致。',
              '- 模型推理沿用GPU路径，SciPy/NumPy特征提取仍在CPU；仅前400秒检查替代chunk，不宣称全记录多chunk一致性。',
              '- 模型训练名单中的被试属于回代端到端核验，并非LOSO泛化；oracle边界和训练重叠使指标不能支持正式部署收益结论。',
              '- 长期running统计、dt1200和HMM的有效感受野≤6秒合规性仍未证明；没有修改正式src。',
              '- 检查点仅包含已输出raw与samples_fed，不包含流式引擎内部状态，不能无损中途恢复；完整scores缓存可直接复用。',
              f"- 错误/未完成：{json.dumps(result['errors'], ensure_ascii=False)}。", '',
              '## 是否改善', '']
    for sid, sub in result['subjects'].items():
        for mode in b.MODES[1:]:
            details = []
            for part in ('full', 'block45', 'block4plus'):
                m = sub['metrics'][mode]['parts'].get(part)
                if m is None:
                    continue
                base = sub['metrics']['baseline']['parts'][part]
                delta = {k: m[k] - base[k] for k in ('bacc', 'auc', 'f1')}
                details.append(part + '：' + '，'.join(f'{k} {v:+.4f}' for k, v in delta.items()))
            lines.append(f"- {sid} {mode}：" + '；'.join(details) + '。')
    (HERE / '报告_batch2.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subjects', default='009,010,050')
    ap.add_argument('--subject-budget', type=float, default=240)
    args = ap.parse_args()
    torch.set_num_threads(1)
    bundle = torch.load(b.streaming.BUNDLE, map_location='cpu', weights_only=False)
    path = HERE / 'results.json'
    result = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {
        'subjects': {}, 'errors': {}, 'model': str(b.streaming.BUNDLE),
        'training_subjects': bundle.get('subjects'),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'protocol': 'batch1 shared raw full BDF; oracle actual event trial-start reset; chunk137 versus chunk250 prefix400'}
    for sid in args.subjects.split(','):
        sid = sid.zfill(3)
        result['errors'].pop(str(int(sid)), None)
        started = time.perf_counter()
        deadline = started + args.subject_budget
        try:
            bdf = next((b.e58.ROOT / 'EEG_dataset' / sid).glob('*.bdf'))
            rec = b.e58.infer.load_bdf(bdf)
            x = np.asarray(rec['signal_uv'], dtype=float) * 1e-6
            x[~rec['valid']] = np.nan
            fs = rec['fs']
            trials = b.e58.load_trials(str(bdf.parent / f'{sid}_events.csv'), str(bdf))
            cache = HERE / f'{sid}_scores.npz'
            if cache.exists():
                with np.load(cache) as data:
                    raw = np.column_stack((data['time_s'], data['raw']))
                sub = result['subjects'][sid]
            else:
                raw, seconds = b.run(x, fs, bundle, 137, deadline, HERE / f'{sid}_checkpoint.npz')
                metrics, arrays = b.evaluate(raw, trials)
                np.savez(cache, **arrays)
                sub = dict(duration_s=len(x)/fs, invalid_samples=int((~rec['valid']).sum()),
                           n_outputs=len(raw), inference_seconds=seconds,
                           trial_starts_s=[float(t.start) for t in trials], metrics=metrics,
                           checks={}, in_training=sid in bundle.get('subjects', []))
                result['subjects'][sid] = sub
                b.save(HERE / f'{sid}_result.json', sub)
                persist(result)
            if '250' not in sub['checks']:
                n = min(len(x), int(400 * fs))
                q, seconds = b.run(x[:n], fs, bundle, 250, deadline)
                ref = raw[raw[:, 0] <= n/fs]
                same = len(q) == len(ref) and np.array_equal(q[:, 0], ref[:, 0])
                check = dict(prefix_seconds=n/fs, n_outputs=len(q), seconds=seconds,
                             times_identical=same, raw_identical=np.array_equal(q, ref),
                             raw_max_abs=float(np.max(np.abs(q[:, 1]-ref[:, 1]))) if same and len(q) else None,
                             post={})
                for mode in b.MODES:
                    a, _ = b.postprocess(q, trials, mode)
                    z, _ = b.postprocess(ref, trials, mode)
                    check['post'][mode] = dict(identical=np.array_equal(a, z),
                                              max_abs=float(np.max(np.abs(a-z))) if same and len(a) else None)
                sub['checks']['250'] = check
            sub['total_seconds'] = time.perf_counter() - started
            result['errors'].pop(sid, None)
            b.save(HERE / f'{sid}_result.json', sub)
            print(sid, json.dumps(sub), flush=True)
        except Exception as exc:
            result['errors'][sid] = f'{type(exc).__name__}: {exc}'
            print(sid, result['errors'][sid], flush=True)
        persist(result)
    report(result)


if __name__ == '__main__':
    main()
