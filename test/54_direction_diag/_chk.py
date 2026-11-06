import json
import sys

import numpy as np

sys.path.insert(0, r"d:\WORK\BCIC2026\CogState\test\17_compliant")
import loso as L

subs, D = L.load()
d = np.load(r"d:\WORK\BCIC2026\CogState\test\53_e17_post_loso\e53_scores.npz")
for q in subs:
    y = D[q][1]
    b = D[q][2]
    for name, m in (("b01", b <= 1), ("b23", (b >= 2) & (b <= 3)), ("b45", b >= 4)):
        yy = y[m]
        if len(set(yy.tolist())) < 2:
            print("single-class", q, name, "n", int(m.sum()), "y", np.unique(yy, return_counts=True))
