from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import stream_block_reset as e58
import streaming

MODES = ('baseline', 'dt_reset_hmm_keep', 'dt_reset_hmm_reset')


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def postprocess(raw, trials, mode):
    post = e58.ResetPost(mode)
    scores, resets = [], []
    boundary = 1
    # Advance starts even across unlabeled gaps: the old prev=None logic missed these resets.
    for t, s in raw:
        while boundary < len(trials) and t >= trials[boundary].start:
            post.reset()
            resets.append(float(t))
            boundary += 1
        scores.append(post.update(float(s)))
    return np.asarray(scores), resets


def evaluate(raw, trials):
    t = raw[:, 0]
    y = np.full(len(t), -1)
    blocks = np.zeros(len(t), dtype=int)
    for i, tr in enumerate(trials, 1):
        mask = (t > tr.start) & (t <= tr.end + 1e-9)
        y[mask], blocks[mask] = tr.label, i
    result, arrays = {}, {'time_s': t, 'raw': raw[:, 1], 'label': y, 'block': blocks}
    for mode in MODES:
        scores, resets = postprocess(raw, trials, mode)
        arrays[mode] = scores
        parts = {'full': y >= 0, 'first600': (y >= 0) & (t < 600),
                 'block45': (y >= 0) & np.isin(blocks, (4, 5)),
                 'block4plus': (y >= 0) & (blocks >= 4)}
        result[mode] = {'reset_output_times_s': resets if mode != 'baseline' else [],
                        'parts': {p: dict(e58.metrics(y[m], scores[m]), n=int(m.sum()))
                                  for p, m in parts.items() if m.any()}}
    return result, arrays


def run(signal, fs, bundle, chunk, deadline, checkpoint=None):
    decoder = streaming.StreamingInference(bundle, fs=fs, step=1.0)
    out = []
    started = time.perf_counter()
    last = started
    for start in range(0, len(signal), chunk):
        if time.perf_counter() > deadline:
            if checkpoint:
                np.savez(checkpoint, raw=np.asarray(out).reshape(-1, 2), samples_fed=start)
            raise TimeoutError(f'EEG feature/scoring stream exceeded budget at sample {start}/{len(signal)}; outputs={len(out)}')
        out.extend((t, s) for t, s, _ in decoder.update(signal[start:start + chunk]))
        now = time.perf_counter()
        if checkpoint and now - last >= 15:
            np.savez(checkpoint, raw=np.asarray(out).reshape(-1, 2), samples_fed=min(start + chunk, len(signal)))
            print(f'{checkpoint.stem}: {len(out)} frames, {now-started:.1f}s', flush=True)
            last = now
    return np.asarray(out).reshape(-1, 2), time.perf_counter() - started


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subjects', default='018,004')
    ap.add_argument('--budget', type=float, default=480)
    args = ap.parse_args()
    torch.set_num_threads(1)
    bundle = torch.load(streaming.BUNDLE, map_location='cpu', weights_only=False)
    total_start = time.perf_counter()
    result = {'subjects': {}, 'errors': {}, 'device': 'cuda' if torch.cuda.is_available() else 'cpu',
              'model': str(streaming.BUNDLE), 'training_subjects': bundle.get('subjects'),
              'protocol': 'Real full BDF, NaN invalid samples, continuous E17 6s/1s stream; oracle trial-start reset; shared raw inference for three postprocessing modes.'}
    for sid in args.subjects.split(','):
        started = time.perf_counter()
        deadline = min(total_start + args.budget, started + 180)
        if started >= deadline:
            result['errors'][sid] = 'Not started: total budget exhausted'
            continue
        try:
            bdf = next((e58.ROOT / 'EEG_dataset' / sid).glob('*.bdf'))
            rec = e58.infer.load_bdf(bdf)
            x = np.asarray(rec['signal_uv'], dtype=float) * 1e-6
            x[~rec['valid']] = np.nan
            fs = rec['fs']
            trials = e58.load_trials(str(bdf.parent / f'{sid}_events.csv'), str(bdf))
            raw, elapsed = run(x, fs, bundle, 137, deadline, HERE / f'{sid}_checkpoint.npz')
            metrics, arrays = evaluate(raw, trials)
            np.savez(HERE / f'{sid}_scores.npz', **arrays)
            subject = {'duration_s': len(x) / fs, 'invalid_samples': int((~rec['valid']).sum()),
                       'n_outputs': len(raw), 'inference_seconds': elapsed,
                       'trial_starts_s': [float(tr.start) for tr in trials], 'metrics': metrics,
                       'checks': {}, 'in_training': sid in bundle.get('subjects', [])}
            result['subjects'][sid] = subject
            save(HERE / f'{sid}_result.json', subject)
            save(HERE / 'results.json', result)
            # Fresh end-to-end runs on a 400-second prefix, not cached feature replay.
            # Includes the initial trial transition on typical records; report actual reset times.
            for chunk in (92, 250, 1250):
                n = min(len(x), int(400 * fs))
                q, seconds = run(x[:n], fs, bundle, chunk, deadline)
                ref = raw[raw[:, 0] <= n / fs]
                same_times = len(q) == len(ref) and np.array_equal(q[:, 0], ref[:, 0])
                check = {'prefix_seconds': n / fs, 'n_outputs': len(q), 'seconds': seconds,
                         'times_identical': same_times, 'raw_identical': np.array_equal(q, ref),
                         'raw_max_abs': float(np.max(np.abs(q[:, 1] - ref[:, 1]))) if same_times and len(q) else None,
                         'post': {}}
                for mode in MODES:
                    a, _ = postprocess(q, trials, mode)
                    b, _ = postprocess(ref, trials, mode)
                    check['post'][mode] = {'identical': np.array_equal(a, b),
                                          'max_abs': float(np.max(np.abs(a-b))) if same_times and len(a) else None}
                subject['checks'][str(chunk)] = check
                save(HERE / f'{sid}_result.json', subject)
                save(HERE / 'results.json', result)
            subject['total_seconds'] = time.perf_counter() - started
            save(HERE / f'{sid}_result.json', subject)
            print(sid, json.dumps(metrics), flush=True)
        except Exception as exc:
            result['errors'][sid] = f'{type(exc).__name__}: {exc}'
            print(sid, result['errors'][sid], flush=True)
        save(HERE / 'results.json', result)
    result['elapsed_seconds'] = time.perf_counter() - total_start
    save(HERE / 'results.json', result)
    lines = ['# E58 batch1 真实BDF端到端验证', '',
             '完整BDF逐被试运行一次137点chunk；同一原始分数复用到三个后处理，逐被试及每15秒保存中间数据。模型推理使用GPU，原有SciPy/NumPy特征提取使用CPU，未改正式src。', '',
             '重置使用事件CSV的真实trial_start，属于已知边界（oracle）消融，不代表可无标签自动部署。测试被试可能在模型训练名单中，下表为回代端到端验证，不是LOSO泛化结果。', '',
             '|被试|方案|片段|窗口数|bACC|AUC|F1|', '|---|---|---|---:|---:|---:|---:|']
    for sid, sub in result['subjects'].items():
        for mode, value in sub['metrics'].items():
            for part, m in value['parts'].items():
                lines.append(f"|{sid}|{mode}|{part}|{m['n']}|{m['bacc']:.4f}|{m['auc']:.4f}|{m['f1']:.4f}|")
    lines += ['', '## 一致性与运行范围', '',
              '不同chunk检查使用新建引擎，重新提取真实BDF前400秒特征，与完整137点chunk运行的同时间前缀比较；不是全记录多chunk验证。']
    for sid, sub in result['subjects'].items():
        lines.append(f"- {sid}：完整记录{sub['duration_s']:.1f}s，推理{sub['inference_seconds']:.1f}s，输出{sub['n_outputs']}帧；训练名单内={sub['in_training']}。")
        for chunk, check in sub['checks'].items():
            lines.append(f"  - chunk={chunk}：前缀{check['prefix_seconds']}s，原始分数完全一致={check['raw_identical']}，max_abs={check['raw_max_abs']}；三种后处理完全一致={all(v['identical'] for v in check['post'].values())}。")
    lines += ['', '## 瓶颈与局限', '',
              '- 原实现按后处理模式反复提取特征、并在无标签间隔将prev清为None，可能漏掉后续block重置。本脚本只提取一次完整分数，按到达trial_start触发重置，跨间隔不漏重置；首个trial不额外重置，以保持E58原设计。',
              '- 原实现block45实际为block>=4；本批分别报告严格block4/5与block4plus，避免混淆。first600仍为BDF绝对前600秒。',
              '- 完整流包含前后无标签EEG与间隔，它们更新无监督统计及后处理。标签仅用于边界消融和指标计算。',
              '- 每被试180秒、总预算受参数限制；超时不重试，已完成完整流结果及中间检查均保留。',
              f"- 实际总耗时：{result['elapsed_seconds']:.1f}秒。", f"- 未完成/错误：{json.dumps(result['errors'], ensure_ascii=False)}", '',
              '尚不能据此认定满足有效感受野≤6秒：原有长时running统计、dt1200和HMM合规解释仍需独立确认。小样本回代及oracle边界不足以支持正式部署收益结论。']
    (HERE / '报告_batch1.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
