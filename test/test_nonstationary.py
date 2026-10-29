import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nonstationary_ablation import feature_blocks, wavelet_features, timefreq_features, spectrogram, WAVE_NAMES, TF_NAMES


class TestNonstationary(unittest.TestCase):
    def setUp(self):
        self.fs = 250.
        t = np.arange(1500) / self.fs
        self.x = np.sin(2*np.pi*10*t) * (1+t/6)

    def test_dimensions_and_finite(self):
        for x in (self.x, np.zeros(1500), np.random.default_rng(0).normal(size=1500)):
            blocks = feature_blocks(x)
            self.assertEqual({k: len(v) for k, v in blocks.items()},
                             {'raw44': 44, 'quality53': 53, 'wavelet': 23, 'timefreq': 50, 'quality': 9})
            self.assertTrue(all(np.isfinite(v).all() for v in blocks.values()))
        self.assertEqual(len(WAVE_NAMES), 23)
        self.assertEqual(len(TF_NAMES), 50)

    def test_exact_window(self):
        for bad in (self.x[:-1], np.r_[self.x, 0], np.full(1500, np.nan), np.zeros((1, 1500))):
            with self.assertRaises(ValueError):
                feature_blocks(bad)

    def test_future_independence(self):
        stream = np.r_[self.x, np.zeros(1500)]
        before = feature_blocks(stream[:1500])
        stream[1500:] = 1e8
        after = feature_blocks(stream[:1500])
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])

    def test_stft_support(self):
        f, t, z = spectrogram(self.x, self.fs)
        self.assertEqual(len(t), 11)
        self.assertGreaterEqual(t.min()-.5, 0)
        self.assertLessEqual(t.max()+.5, 6)

    def test_relative_energy(self):
        wave = wavelet_features(self.x)
        self.assertAlmostEqual(float(wave[6:12].sum()), 1.)
        self.assertTrue(np.all((wave[12:18] >= 0) & (wave[12:18] <= 1)))
        tf = timefreq_features(self.x)
        self.assertAlmostEqual(float(tf[25::5].sum()), 1.)

    def test_subject_disjoint(self):
        groups = np.repeat(np.arange(10), 4)
        labels = np.tile([0, 1, 0, 1], 10)
        for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros((40, 1)), labels, groups):
            self.assertFalse(set(groups[tr]) & set(groups[va]))


if __name__ == '__main__':
    unittest.main()
