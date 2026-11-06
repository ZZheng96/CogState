"""诊断2：真实 BDF 流式（部署口径）逐人方向，与 npz 分数口径对比。"""
import sys
import numpy as np
import torch

ROOT = r"d:\WORK\BCIC2026\CogState"
sys.path.insert(0, ROOT + r"\test\17_compliant")
sys.path.insert(0, ROOT + r"\src")
sys.path.insert(0, ROOT)
import loso as L
import infer
import streaming as S
from system import truth

subs, D = L.load()
bundle = torch.load(S.BUNDLE, map_location="cpu", weights_only=False)
rows = []
for q in subs:
    bdf = next((__import__("pathlib").Path(ROOT) / "EEG_dataset" / q).glob("*.bdf"))
    rec = infer.load_bdf(bdf)
    fs = rec["fs"]
    sig = np.asarray(rec["signal_uv"], dtype=float) * 1e-6
    sig[~rec["valid"]] = np.nan
    dec = S.StreamingInference(bundle, fs=fs, step=1.0)
    outs = infer.stream_outputs(dec, infer.PostProcessor(detrend_halflife=1200), sig, fs, 250)
    t = np.array([o["time_s"] for o in outs])
    post = np.array([o["score_task"] for o in outs])
    y = truth.labels("builtin", q, t)
    m = y >= 0
    if m.sum() < 10 or len(set(y[m])) < 2:
        print(q, "skip", m.sum())
        continue
    r = L.subj_metrics(torch.tensor(y[m]), torch.tensor(post[m]), torch.zeros(1))
    rows.append((q, r["bacc"], r["auc"], float(post[m].mean())))
    print(q, {k: round(v, 3) for k, v in r.items()}, "mean", round(float(post[m].mean()), 3), flush=True)

b = np.array([r[1] for r in rows]); a = np.array([r[2] for r in rows])
print("N", len(rows), "bACC %.4f AUC %.4f  AUC<0.5: %d  bACC<0.5: %d" % (b.mean(), a.mean(), (a < .5).sum(), (b < .5).sum()))
print("bad:", [r[0] for r in rows if r[2] < .5])
