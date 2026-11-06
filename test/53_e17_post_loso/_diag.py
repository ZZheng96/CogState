"""诊断：E17 分数方向从何而来（训练集 vs LOSO）。"""
import sys
import numpy as np
import torch

sys.path.insert(0, r"d:\WORK\BCIC2026\CogState\test\17_compliant")
sys.path.insert(0, r"d:\WORK\BCIC2026\CogState\src")
import loso as L
import infer

subs, D = L.load()
d = np.load(r"d:\WORK\BCIC2026\CogState\test\53_e17_post_loso\e53_scores.npz")
raw = {q: d[q].astype(float) for q in subs}

# 1) 原始分数（未后处理）逐人 AUC：方向是否已经错
auc_raw = np.array([L.subj_metrics(D[q][1], torch.tensor(raw[q]), torch.zeros(1))["auc"] for q in subs])
print("raw AUC: mean %.4f  <0.5: %d  <0.45: %d" % (auc_raw.mean(), (auc_raw < .5).sum(), (auc_raw < .45).sum()))
print("raw AUC<0.5:", [q for q, a in zip(subs, auc_raw) if a < .5])

# 2) 后处理（部署口径）逐人 AUC / bACC
pp = {q: np.array([infer.PostProcessor(detrend_halflife=1200).update(float(s)) for s in raw[q]]) for q in subs}
auc_pp = np.array([L.subj_metrics(D[q][1], torch.tensor(pp[q]), torch.zeros(1))["auc"] for q in subs])
bac_pp = np.array([L.subj_metrics(D[q][1], torch.tensor(pp[q]), torch.zeros(1))["bacc"] for q in subs])
print("post AUC: mean %.4f  <0.5: %d" % (auc_pp.mean(), (auc_pp < .5).sum()))
print("post bACC: mean %.4f" % bac_pp.mean())

# 3) 分数均值 vs 标签：训练集里 task 分数是否系统性更高
for name, sc in (("raw", raw), ("post", pp)):
    mt = np.mean([sc[q][D[q][1] == 1].mean() for q in subs])
    mr = np.mean([sc[q][D[q][1] == 0].mean() for q in subs])
    print(f"{name}: mean(task)={mt:+.3f} mean(rest)={mr:+.3f}")

# 4) 块均值（6 块）与标签，看漂移
for q in ("018", "004", "010", "050", "024"):
    b = D[q][2]
    print(q, "y", D[q][1][0], "raw blk", [round(float(raw[q][b == k].mean()), 2) for k in range(6)],
          "post blk", [round(float(pp[q][b == k].mean()), 2) for k in range(6)])
