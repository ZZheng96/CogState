"""构建去时序化的数据库：把每个受试者的 6 秒窗按标签切片，去掉一切时间/trial 信息。

输出结构：
    database/windows/relax/sub-001.npz
    database/windows/task/sub-001.npz
每个文件只含 x(1500 原始采样)、f(858 维 EEG 特征)、e(20 维肌电/眨眼特征)、subject、label。
窗内顺序用固定种子打乱，且不保存任何时间戳、trial 编号或块位置，
因此训练端无法还原"这是第几个窗口"。
"""
import json, sys, time
from pathlib import Path
import numpy as np
import mne

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT/'test'
DB = ROOT/'database'
sys.path[:0] = [str(ROOT/'src'), str(OUT)]
from cogstate import load_trials, bad_intervals, valid_window
from highdim_deep_ablation import highdim_features
from emg_v2 import features_v2, NAMES as EMG_NAMES

FS = 250.
WIN = 6.


def main():
    tick = time.perf_counter()
    (DB/'windows'/'relax').mkdir(parents=True, exist_ok=True)
    (DB/'windows'/'task').mkdir(parents=True, exist_ok=True)
    paths = sorted((ROOT/'EEG_dataset').glob('*/*_events.csv'))
    feat_names = json.loads((OUT/'highdim_deep_results.json').read_text(encoding='utf-8'))['feature_names']
    manifest = {'fs': FS, 'window_seconds': WIN, 'stride_seconds': WIN,
                'note': 'windows are grouped by label; no timestamp, trial id or stream position is stored; '
                        'order within each shard is shuffled with a fixed seed',
                'feature_names': feat_names, 'emg_feature_names': EMG_NAMES, 'subjects': {}}

    for i, csv in enumerate(paths, 1):
        bdf = next(csv.parent.glob('*.bdf'))
        raw = mne.io.read_raw_bdf(str(bdf), preload=True, verbose='ERROR')
        fs, bad = float(raw.info['sfreq']), bad_intervals(raw)
        buckets = {0: [], 1: []}
        for tr in load_trials(str(csv), str(bdf)):
            end = tr.start+WIN
            while end <= tr.end+1e-6:
                if valid_window(end-WIN, end, bad, raw.n_times/fs):
                    x = raw.get_data(start=round((end-WIN)*fs), stop=round(end*fs))[0]
                    if len(x) == round(WIN*fs) and np.isfinite(x).all():
                        buckets[tr.label].append(x.astype(np.float32))
                end += WIN
        raw.close()
        subject = csv.parent.name
        counts = {}
        for label, name in ((0, 'relax'), (1, 'task')):
            arr = np.array(buckets[label], dtype=np.float32)
            if not len(arr):
                continue
            np.random.default_rng(0).shuffle(arr)
            F = np.array([highdim_features(x.astype(float), fs) for x in arr], dtype=np.float32)
            E = np.array([features_v2(x.astype(float)) for x in arr], dtype=np.float32)
            np.savez_compressed(DB/'windows'/name/f'sub-{subject}.npz',
                                x=arr, f=F, e=E,
                                subject=np.array([subject]*len(arr)),
                                label=np.full(len(arr), label, dtype=np.int8))
            counts[name] = int(len(arr))
        manifest['subjects'][subject] = counts
        print(f'[build {i}/{len(paths)}] sub-{subject} relax={counts.get("relax", 0)} '
              f'task={counts.get("task", 0)} elapsed={time.perf_counter()-tick:.0f}s', flush=True)

    manifest['totals'] = {'relax': int(sum(v.get('relax', 0) for v in manifest['subjects'].values())),
                          'task': int(sum(v.get('task', 0) for v in manifest['subjects'].values())),
                          'n_subjects': len(manifest['subjects'])}
    (DB/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'[done] totals={manifest["totals"]} elapsed={time.perf_counter()-tick:.0f}s', flush=True)


if __name__ == '__main__':
    main()
