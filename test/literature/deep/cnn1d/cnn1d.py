"""Light-weight 1D-CNN on raw single-channel EEG (Saini et al., arXiv 2012.06782 style).
Stacked Conv1d+BN+ReLU+MaxPool blocks, flatten, dense head with dropout. Input: 4 s crop (1000 samples @ 250 Hz).
Run: python test/literature/deep/cnn1d/cnn1d.py [--smoke] [--seeds 0,1,2] [--protocols STRICT,LIT_WINDOW,LIT_SESSION_NORM]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import run, CROP  # noqa: E402
from torch import nn  # noqa: E402


class CNN1d(nn.Module):
    def __init__(self, chans=(16, 32, 64, 64, 128), k=7, pool=3):
        super().__init__()
        layers, cin, L = [], 1, CROP
        for c in chans:
            layers += [nn.Conv1d(cin, c, k, padding=k // 2), nn.BatchNorm1d(c), nn.ReLU(), nn.MaxPool1d(pool)]
            cin, L = c, L // pool
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(.5), nn.Linear(cin * L, 64), nn.ReLU(), nn.Dropout(.5),
                                  nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.features(x))


if __name__ == '__main__':
    run('cnn1d', CNN1d)
