import io
import sys
import unittest
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from highdim_deep_ablation import highdim_features, WindowFeatures, FeatureTransform, network, predict
from nonstationary_ablation import feature_blocks


class HighdimTests(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(42).normal(0, 1e-5, 1500)

    def test_features_and_constant(self):
        f, names = highdim_features(self.x, return_names=True)
        b = feature_blocks(self.x)
        np.testing.assert_array_equal(f[:82], np.r_[b['wavelet'], b['timefreq'], b['quality']])
        self.assertGreater(len(f), 600)
        self.assertEqual(len(f), len(names))
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(np.isfinite(highdim_features(np.zeros(1500))).all())

    def test_chunking_and_outside_window(self):
        record = np.r_[np.ones(500), self.x, np.ones(400)]
        reference = highdim_features(record[500:2000])
        record[:500] = -1e9; record[2000:] = 1e9
        np.testing.assert_array_equal(reference, highdim_features(record[500:2000]))
        stream = WindowFeatures()
        self.assertIsNone(stream.update(record[:500]))
        for chunk in np.array_split(self.x, 17):
            value = stream.update(chunk)
        np.testing.assert_array_equal(value, reference)
        np.testing.assert_array_equal(highdim_features(self.x), reference)

    def test_validation_and_selection(self):
        with self.assertRaises(ValueError):
            highdim_features(self.x[:-1])
        bad = self.x.copy(); bad[0] = np.nan
        with self.assertRaises(ValueError):
            highdim_features(bad)
        rng = np.random.default_rng(8)
        X = rng.normal(size=(80, 20)); X[:, 0] = 0; X[:, 2] = X[:, 1]
        y = np.tile([0, 1], 40)
        for mode in ('kbest', 'corr', 'variance', 'prefix'):
            transform = FeatureTransform(10, mode).fit(X, y)
            mean = transform.scaler.mean_.copy(); indices = transform.indices.copy()
            transform.transform(X+1e6)
            np.testing.assert_array_equal(mean, transform.scaler.mean_)
            np.testing.assert_array_equal(indices, transform.indices)
            self.assertEqual(transform.transform(X).shape[1], 10)
            if mode != 'prefix':
                self.assertNotIn(0, indices)
            if mode == 'corr':
                self.assertFalse(1 in indices and 2 in indices)

    def test_subject_splits(self):
        from sklearn.model_selection import StratifiedGroupKFold
        groups = np.repeat(np.arange(20), 20)
        y = np.tile([0, 1], 200)
        X = np.zeros((400, 1))
        seen = np.zeros(400, dtype=int)
        for fold, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, groups), 1):
            seen[te] += 1
            self.assertFalse(set(groups[tr]) & set(groups[te]))
            a, b = next(StratifiedGroupKFold(4, shuffle=True, random_state=fold).split(X[tr], y[tr], groups[tr]))
            self.assertFalse(set(groups[tr[a]]) & set(groups[tr[b]]))
            self.assertFalse(set(groups[tr[b]]) & set(groups[te]))
        np.testing.assert_array_equal(seen, np.ones(400))

    def test_transform_roundtrip(self):
        rng = np.random.default_rng(11)
        X = rng.normal(size=(50, 30)); y = np.tile([0, 1], 25)
        transform = FeatureTransform(10).fit(X, y)
        data = io.BytesIO()
        torch.save({'indices': transform.indices, 'scaler': transform.scaler}, data)
        data.seek(0)
        saved = torch.load(data, weights_only=False)
        np.testing.assert_array_equal(transform.transform(X), saved['scaler'].transform(X[:, saved['indices']]).astype(np.float32))

    def test_model_roundtrip(self):
        torch.set_num_threads(1)
        for kind, dim in (('mlp', 82), ('large', 300), ('cnn', 1500)):
            net = network(kind, dim).eval()
            X = np.random.default_rng(3).normal(size=(3, dim)).astype(np.float32)
            if kind == 'cnn':
                X = X[:, None, :]
            expected = predict(net, X, 'cpu')
            data = io.BytesIO(); torch.save(net.state_dict(), data); data.seek(0)
            restored = network(kind, dim)
            restored.load_state_dict(torch.load(data, weights_only=True))
            np.testing.assert_array_equal(expected, predict(restored, X, 'cpu'))


if __name__ == '__main__':
    unittest.main()
