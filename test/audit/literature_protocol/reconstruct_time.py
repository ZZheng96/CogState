"""Recover the real recording time of every database window (the database shards are shuffled, no timestamps stored).

Replays the window loop of test/build_database.py (load_trials / bad_intervals / valid_window, 6 s windows, 6 s stride,
channel 0) on EEG_dataset/<subject>/*.bdf + *_events.csv, then matches every stored window x (database/windows/<label>/
sub-<id>.npz) to a replayed window by exact float32 bytes (asserted: bijection, every row matched). Output:
real_time.npz with t_end [n_rows] (seconds from recording start) and trial index, in canonical order (load_db order:
per subject concat(relax, task)). Mapping: stored = replayed[default_rng(0).permutation(n)] (asserted bytewise for
every shard; some shards contain byte-identical duplicate windows, so content hashing alone is ambiguous).
Run: python test/audit/literature_protocol/reconstruct_time.py [--smoke]   (smoke: 3 subjects, no npz written)
"""
import argparse, sys, time
from pathlib import Path

import mne
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import TEST, Tee, el  # noqa: E402
ROOT = TEST.parent
sys.path.insert(0, str(ROOT / 'src'))
from cogstate import bad_intervals, load_trials, valid_window  # noqa: E402

WIN = 6.


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); args = ap.parse_args()
    stem = 'reconstruct_time' + ('_smoke' if args.smoke else '')
    tee = Tee(sys.stdout, HERE / (stem + '.log')); sys.stdout = tee
    DB = ROOT / 'database'
    subs = sorted(p.stem for p in (DB / 'windows' / 'relax').glob('*.npz'))
    if args.smoke: subs = subs[:3]
    T, TR, Y, G = [], [], [], []; perm_ok = 0
    for i, s in enumerate(subs, 1):
        d = ROOT / 'EEG_dataset' / s.replace('sub-', '')
        csv = next(d.glob('*_events.csv')); bdf = next(d.glob('*.bdf'))
        raw = mne.io.read_raw_bdf(str(bdf), preload=True, verbose='ERROR')
        fs, bad = float(raw.info['sfreq']), bad_intervals(raw)
        buckets = {0: [], 1: []}
        for tr in load_trials(str(csv), str(bdf)):
            end = tr.start + WIN
            while end <= tr.end + 1e-6:
                if valid_window(end - WIN, end, bad, raw.n_times / fs):
                    x = raw.get_data(start=round((end - WIN) * fs), stop=round(end * fs))[0]
                    if len(x) == round(WIN * fs) and np.isfinite(x).all():
                        buckets[tr.label].append((x.astype(np.float32), end, tr.trial_id))
                end += WIN
        raw.close()
        for lab, name in ((0, 'relax'), (1, 'task')):
            X = np.load(DB / 'windows' / name / f'{s}.npz')['x']
            assert len(X) == len(buckets[lab]), (s, name, len(X), len(buckets[lab]))
            idx = np.random.default_rng(0).permutation(len(X))  # == build_database's rng(0).shuffle on the array
            Xr = np.stack([buckets[lab][j][0] for j in idx])
            assert np.array_equal(Xr, X.astype(np.float32)), ('stored x != replayed x[perm]', s, name)
            n_dup = len(X) - len({b[0].tobytes() for b in buckets[lab]})
            if n_dup: print(f'   note: {s} {name} has {n_dup} byte-identical duplicate windows', flush=True)
            perm_ok += 1
            T.append(np.array([buckets[lab][j][1] for j in idx])); TR.append(np.array([buckets[lab][j][2] for j in idx]))
            Y.append(np.full(len(X), lab)); G.append(np.full(len(X), s))
        print(f'[subject {i}/{len(subs)}] {s} relax={len(buckets[0])} task={len(buckets[1])} {el()}', flush=True)
    t_end, trial = np.concatenate(T), np.concatenate(TR)
    print(f'[check] all {len(t_end)} stored windows == replayed windows[default_rng(0).permutation] (exact float32) '
          f'in {perm_ok}/{2 * len(subs)} shards', flush=True)
    if not args.smoke:
        np.savez_compressed(HERE / 'real_time.npz', t_end=t_end, trial=trial, y=np.concatenate(Y), groups=np.concatenate(G))
        print('[done] -> real_time.npz', flush=True)
    sys.stdout = tee.stream; tee.f.close()


if __name__ == '__main__':
    main()
