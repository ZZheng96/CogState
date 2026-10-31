"""EEGNet (Lawhern et al. 2018) with C=1. Block 1: temporal Conv2d(1,F1,(1,K1)) + BN; depthwise Conv2d(F1,D*F1,(C,1),
groups=F1) with max-norm 1 (degenerates to a per-filter scaling for C=1, kept for faithfulness) + BN + ELU + AvgPool(1,4)
+ Dropout; Block 2: separable conv (depthwise (1,K2) + pointwise) + BN + ELU + AvgPool(1,8) + Dropout; dense with
max-norm .25. Kernel lengths scaled from 128 Hz to 250 Hz (K1 = fs/2 = 125, K2 = 32). F1=8, D=2, F2=16, dropout .25
(cross-subject setting of the paper). Single logit (BCE) instead of softmax.
Run: python test/literature/deep/eegnet/eegnet.py [--smoke] [--seeds 0,1,2]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import run, CROP  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

C = 1


class EEGNet(nn.Module):
    def __init__(self, F1=8, D=2, F2=16, K1=125, K2=32, p=.25):
        super().__init__()
        self.temporal = nn.Sequential(nn.ZeroPad2d((K1 // 2, K1 - 1 - K1 // 2, 0, 0)),
                                      nn.Conv2d(1, F1, (1, K1), bias=False), nn.BatchNorm2d(F1))
        self.depthwise = nn.Conv2d(F1, D * F1, (C, 1), groups=F1, bias=False)
        self.block1 = nn.Sequential(nn.BatchNorm2d(D * F1), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(p))
        self.block2 = nn.Sequential(nn.ZeroPad2d((K2 // 2, K2 - 1 - K2 // 2, 0, 0)),
                                    nn.Conv2d(D * F1, D * F1, (1, K2), groups=D * F1, bias=False),
                                    nn.Conv2d(D * F1, F2, 1, bias=False), nn.BatchNorm2d(F2), nn.ELU(),
                                    nn.AvgPool2d((1, 8)), nn.Dropout(p))
        self.fc = nn.Linear(F2 * (CROP // 4 // 8), 1)

    @torch.no_grad()
    def constrain(self):
        self.depthwise.weight.data = torch.renorm(self.depthwise.weight.data, 2, 0, 1.)
        self.fc.weight.data = torch.renorm(self.fc.weight.data, 2, 0, .25)

    def forward(self, x):
        x = self.block1(self.depthwise(self.temporal(x[:, :, None, :])))  # (N,1,C=1,T)
        return self.fc(self.block2(x).flatten(1))


if __name__ == '__main__':
    run('eegnet', EEGNet)
