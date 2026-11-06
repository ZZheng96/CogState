import json
import sys

import numpy as np

sys.path.insert(0, r"d:\WORK\BCIC2026\CogState\test\17_compliant")
import loso as L

subs, D = L.load()
a = np.load(r"d:\WORK\BCIC2026\CogState\test\42_e17_loso_hmm\e42_scores.npz")
b = np.load(r"d:\WORK\BCIC2026\CogState\test\53_e17_post_loso\e53_scores.npz")
diffs = []
for q in subs:
    d = float(np.abs(a[q] - b[q]).max())
    diffs.append((d, q))
diffs.sort(reverse=True)
print("max diff over subjects:", diffs[0])
print("top5:", [(q, round(d, 6)) for d, q in diffs[:5]])
print("n subjects with diff>1e-4:", sum(d > 1e-4 for d, _ in diffs))
