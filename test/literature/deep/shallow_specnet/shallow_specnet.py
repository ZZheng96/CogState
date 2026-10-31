"""Shallow CNN on spectro-temporal input (Ajra et al., arXiv 2209.11767 style: two conv layers on a time-frequency map).
Input 4 s crop -> STFT (Hann 1 s / hop 31 samples ~0.124 s, 250 Hz) -> log power 1-45 Hz (45 bins x 33 frames) -> BatchNorm over
frequency bins (training statistics only) -> 2 x [Conv2d 3x3 + BN + ReLU + MaxPool2d 2] -> dropout -> linear.
Run: python test/literature/deep/shallow_specnet/shallow_specnet.py [--smoke] [--seeds 0,1,2]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import run, CROP, FS  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

NFFT, HOP, F_LO, F_HI = 250, 31, 1, 45  # 1 Hz bins


class ShallowSpec(nn.Module):
    def __init__(self, F1=16, F2=32):
        super().__init__()
        self.register_buffer('win', torch.hann_window(NFFT), persistent=False)
        nf = F_HI - F_LO + 1; nt = CROP // HOP + 1
        self.inorm = nn.BatchNorm1d(nf)
        self.conv = nn.Sequential(nn.Conv2d(1, F1, 3, padding=1), nn.BatchNorm2d(F1), nn.ReLU(), nn.MaxPool2d(2),
                                  nn.Conv2d(F1, F2, 3, padding=1), nn.BatchNorm2d(F2), nn.ReLU(), nn.MaxPool2d(2))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(.5), nn.Linear(F2 * (nf // 4) * (nt // 4), 1))

    def forward(self, x):
        S = torch.stft(x[:, 0], NFFT, HOP, NFFT, self.win, center=True, return_complex=True)
        S = torch.log(S[:, F_LO * NFFT // FS:F_HI * NFFT // FS + 1].abs() ** 2 + 1e-6)
        return self.head(self.conv(self.inorm(S)[:, None]))


if __name__ == '__main__':
    run('shallow_specnet', ShallowSpec)
