"""E14：窗内 19 维「新判别特征」——跨带协方差（log-Euclidean）、相位-幅值耦合、相位一致性。

与已有 878 维特征（功率/谱形/肌电）互补，考察相位与跨带耦合是否携带新的状态信息。
窗口枚举与 `build_sliding.py`（6 s 窗、1 s 步长）**完全一致**，并与其缓存逐元素对账。

流程（每个被试）：
  1. 全记录只做一次因果滤波：`preprocessing.causal_filter`（4 阶 1–45 Hz 带通 + 50 Hz notch）；
  2. 用 sosfilt 拆 5 个带：1–4 / 4–8 / 8–13 / 13–30 / 30–45 Hz；
  3. 逐窗（只用窗内样本）：
     a) 5 带信号 5×5 协方差（+1e-6·I）→ 对称特征分解 logm → 上三角 15 维；
     b) Tort MI：α 相位 ↔ β 幅值、θ 相位 ↔ γ 幅值，18 个相位区间 → 2 维；
     c) α、β 相位 resultant vector length → 2 维。

产物：test/11_individual_20261102/newfeat/sub-XXX.npz
      F (n,19) float32, y int8, blk int16, t float32
      以及 newfeat/manifest.json、newfeat.log

运行：py -3.11 test/11_individual_20261102/newfeat.py [--subjects 036,054] [--workers 4]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from scipy.signal import sosfilt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "newfeat"
LOG = HERE / "newfeat.log"
SLID1 = HERE / "sliding1"
WIN = 6.0
STRIDE = 1.0
N_BINS = 18
BAND_LABELS = ("delta_1_4", "theta_4_8", "alpha_8_13", "beta_13_30", "gamma_30_45")


def _paths():
    p = str(ROOT / "src")
    if p not in sys.path:
        sys.path.append(p)


def _feature_names():
    names = [f"cov_logm_{BAND_LABELS[i]}_{BAND_LABELS[j]}"
             for i in range(5) for j in range(i, 5)]
    names += ["pac_alpha_beta", "pac_theta_gamma", "rvl_alpha", "rvl_beta"]
    return names


def logm_upper(c):
    """对称矩阵的 logm 上三角（15 维）；非正定回退为 log(I)=0。"""
    c = 0.5 * (c + c.T)
    w, v = np.linalg.eigh(c)
    if not np.isfinite(w).all() or w[0] <= 1e-12:
        return np.zeros(15)
    lo = (v * np.log(w)) @ v.T
    return lo[np.triu_indices(5)]


def tort_mi(phase, amp, edges):
    idx = np.clip(np.digitize(phase, edges) - 1, 0, N_BINS - 1)
    m = np.bincount(idx, weights=amp, minlength=N_BINS)
    tot = m.sum()
    if not np.isfinite(tot) or tot <= 0:
        return 0.0
    p = m / tot
    nz = p > 0
    h = -float(np.sum(p[nz] * np.log(p[nz])))
    return float((np.log(N_BINS) - h) / np.log(N_BINS))


def window_features(bands, edges):
    from scipy.signal import hilbert

    c = np.cov(np.vstack(bands))
    c = c + 1e-6 * float(np.mean(np.diag(c))) * np.eye(5)  # 相对正则，见 manifest 说明
    feats = list(logm_upper(c))
    ana = [hilbert(b) for b in bands]
    ph = [np.angle(a) for a in ana]
    am = [np.abs(a) for a in ana]
    feats.append(tort_mi(ph[2], am[3], edges))  # alpha phase -> beta amplitude
    feats.append(tort_mi(ph[1], am[4], edges))  # theta phase -> gamma amplitude
    feats.append(float(abs(np.mean(np.exp(1j * ph[2])))))
    feats.append(float(abs(np.mean(np.exp(1j * ph[3])))))
    return np.asarray(feats, dtype=np.float32)


def build_one(args):
    sid, outdir = args
    _paths()
    import mne
    from preprocessing import causal_filter
    from cogstate import FS, bad_intervals, load_trials, valid_window

    t0 = time.perf_counter()
    csv = next((ROOT / "EEG_dataset").glob(f"{sid}/*_events.csv"))
    bdf = next(csv.parent.glob("*.bdf"))
    raw = mne.io.read_raw_bdf(str(bdf), preload=True, verbose="ERROR")
    fs, rec_end = float(raw.info["sfreq"]), raw.n_times / float(raw.info["sfreq"])
    assert abs(fs - FS) < 1e-6, (sid, fs)
    bad = bad_intervals(raw)
    trials = load_trials(str(csv), str(bdf))

    x = raw.get_data()[0].astype(float)
    raw.close()
    yf = causal_filter(x)
    nyq = fs / 2.0
    from scipy.signal import butter
    bands_sig = [sosfilt(butter(4, [lo / nyq, hi / nyq], btype="band", output="sos"), yf)
                 for lo, hi in ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))]
    edges = np.linspace(-np.pi, np.pi, N_BINS + 1)

    F, Y, T, B = [], [], [], []
    for bi, tr in enumerate(trials):
        end = tr.start + WIN
        while end <= tr.end + 1e-6:
            if valid_window(end - WIN, end, bad, rec_end):
                s, e = int(round((end - WIN) * fs)), int(round(end * fs))
                if e - s > 0 and np.isfinite(yf[s:e]).all():
                    F.append(window_features([b[s:e] for b in bands_sig], edges))
                    Y.append(tr.label)
                    T.append(end)
                    B.append(bi)
            end += STRIDE

    F = np.asarray(F, dtype=np.float32)
    y = np.asarray(Y, dtype=np.int8)
    t = np.asarray(T, dtype=np.float32)
    blk = np.asarray(B, dtype=np.int16)
    sec = time.perf_counter() - t0

    ref = np.load(SLID1 / f"sub-{sid}.npz")
    ref_y, ref_blk, ref_t = ref["y"], ref["blk"], ref["t"]
    if not (np.array_equal(y, ref_y) and np.array_equal(blk, ref_blk)
            and np.array_equal(t, ref_t)):
        msg = (f"ALIGN MISMATCH rows new={len(y)} ref={len(ref_y)}; "
               f"y_eq={np.array_equal(y, ref_y)} blk_eq={np.array_equal(blk, ref_blk)} "
               f"t_eq={np.array_equal(t, ref_t)}")
        if np.array_equal(t, ref_t) and np.array_equal(y, ref_y):
            d = np.flatnonzero(blk != ref_blk)
            msg += f"; blk diff idx={d[:10].tolist()}"
        return sid, F.shape[0], sec, False, msg

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"sub-{sid}.npz", F=F, y=y, blk=blk, t=t)
    return sid, F.shape[0], sec, True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    _paths()
    if a.subjects:
        subs = [s.strip().zfill(3) for s in a.subjects.split(",")]
    else:
        subs = sorted(p.name for p in (ROOT / "EEG_dataset").iterdir() if p.is_dir())
    Path(a.out).mkdir(parents=True, exist_ok=True)
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    t0 = time.perf_counter()
    log(f"[build] {len(subs)} subjects win={WIN}s stride={STRIDE}s workers={a.workers} out={a.out}")
    ok_subs, failed = [], []
    with Pool(a.workers) as pool:
        for i, (sid, n, sec, ok, msg) in enumerate(pool.imap_unordered(
                build_one, [(s, a.out) for s in subs]), 1):
            if ok:
                ok_subs.append(sid)
                log(f"[{i}/{len(subs)}] {sid} rows={n} {sec:.1f}s "
                    f"elapsed={time.perf_counter() - t0:.0f}s")
            else:
                failed.append(sid)
                log(f"[{i}/{len(subs)}] {sid} FAILED {msg}")
    total = time.perf_counter() - t0
    log(f"[done] ok={len(ok_subs)} failed={len(failed)} total={total:.0f}s")
    if failed:
        log(f"[failed] {','.join(failed)}")

    meta = {
        "window_s": WIN, "stride_s": STRIDE, "sfreq": 250.0,
        "filter": "causal_filter(4th-order 1-45 Hz bandpass + 50 Hz notch), partial state, full record once",
        "bands_hz": [[1, 4], [4, 8], [8, 13], [13, 30], [30, 45]],
        "cov": ("5x5 covariance of band signals per window + 1e-6*mean(diag)*I (relative ridge; "
                "BDF 单位为 V，δ 方差 2.4e-6、γ 方差 4.6e-9，绝对 1e-6*I 会盖过 θ/α/β/γ 使 logm 退化为常数)；"
                "symmetric eigendecomposition logm; upper triangle 15"),
        "pac": "Tort MI, alpha(8-13) phase -> beta(13-30) amp, theta(4-8) phase -> gamma(30-45) amp, 18 bins",
        "phase_consistency": "resultant vector length of alpha and beta",
        "n_features": 19,
        "feature_names": _feature_names(),
        "subjects": sorted(ok_subs),
        "failed": failed,
        "total_seconds": round(total, 1),
        "notes": "y/blk/t 与 sliding1/sub-XXX.npz 逐元素一致（已 assert）",
    }
    (Path(a.out) / "manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                                encoding="utf-8")
    LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
