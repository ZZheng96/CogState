import sys
import numpy as np
import torch

sys.path.insert(0, r"d:\WORK\BCIC2026\CogState\test\17_compliant")
sys.path.insert(0, r"d:\WORK\BCIC2026\CogState\src")
import loso as L
import infer

subs, D = L.load()
d = np.load(r"d:\WORK\BCIC2026\CogState\test\53_e17_post_loso\e53_scores.npz")
s = d["018"].astype(float)
y = D["018"][1]

pp = infer.PostProcessor(detrend_halflife=1200)
out = np.array([pp.update(float(x)) for x in s])
print("out tail", out[-3:])
print("bacc", L.subj_metrics(y, torch.tensor(out), torch.zeros(1)))
print("bacc flipped", L.subj_metrics(y, torch.tensor(-out), torch.zeros(1)))
for k in range(6):
    m = D["018"][2] == k
    print("blk", k, "y", y[m][0], "mean", round(float(out[m].mean()), 2))
