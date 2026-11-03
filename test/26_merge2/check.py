"""E26 评估器自检：与 E23 cands_eval（q300_.6 / q300_.5≈m300）对比，并计时。"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e26lib import THR2, metrics, subj_data  # noqa: E402
from clib import load  # noqa: E402

R = load()
q = R["subs"][0]
C, cal_t, tst, yy = subj_data(R, q)
torch.cuda.synchronize()
t = time.time()
out = metrics(C, tst, yy).cpu().numpy()
torch.cuda.synchronize()
print("time", time.time() - t, out.shape)
names = [n for n, _, _ in THR2]
for mine, ref in (("q300_0.6", 5), ("q300_0.5", 2), ("q600_0.4", 6)):
    a, b = out[:, names.index(mine), 0], R["ev"][0, 0, :, ref]
    m = ~np.isnan(a) & ~np.isnan(b)
    print(mine, "max|diff|", np.abs(a[m] - b[m]).max(), "mean", np.abs(a[m] - b[m]).mean())
print("oracle E23", np.nanmax(R["ev"][0, 0]), "E26", np.nanmax(out[:, :, 0]))
