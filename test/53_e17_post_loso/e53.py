"""E53：E17+后处理（dt1200 + 前向 HMM d300/t1）的 57 人 LOSO 重做与改进候选。

stage train：57 折 LOSO 重训 E17（LR + 3 种子 MLP，融合尺度只用训练折），分数存 e53_scores.npz。
stage eval ：在分数上做后处理候选（全部严格因果），输出全帧/块4–5 指标、分组分解与嵌套 LOSO 选择。
stage verify：用 src/infer.py 的 PostProcessor 逐帧复现所选后处理，并做前缀/分块一致性检查。

运行：
  .venv\\Scripts\\python.exe test/53_e17_post_loso/e53.py train
  .venv\\Scripts\\python.exe test/53_e17_post_loso/e53.py eval
  .venv\\Scripts\\python.exe test/53_e17_post_loso/e53.py verify --subject 018
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent
ROOT = TESTS.parent
sys.path.insert(0, str(TESTS / "17_compliant"))
import loso as L  # noqa: E402
from multiscale import fit_mlp  # noqa: E402

DEV = "cuda"
SCALES = (60, 120, 300, 600, 99999)
SCORES = HERE / "e53_scores.npz"
RESULT = HERE / "e53.json"


def _imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


E42 = _imp("e42lib", TESTS / "42_e17_loso_hmm" / "e42.py")
E44 = _imp("e44lib", TESTS / "44_late_drift" / "e44.py")
metrics, summary = E42.metrics, E42.summary


# ---------------------------------------------------------------- 训练
def train_scores():
    subs, D = L.load()
    F = E42.build_features(subs, D)
    scores = {}
    for i, q in enumerate(subs):
        t0 = time.time()
        tr = [p for p in subs if p != q]
        X = torch.cat([F[p] for p in tr])
        y = np.concatenate([D[p][1] for p in tr])
        flr = L.fit_lr(X, y, 0.001)
        fms = [fit_mlp(X, y, seed=s) for s in range(3)]
        lr_parts, mlp_parts = [], []
        for p in tr:
            lr_parts.append(flr(F[p]))
            mlp_parts.append(torch.stack([f(F[p]) for f in fms]).mean(0))
        std_lr = float(torch.cat(lr_parts).std())
        std_mlp = float(torch.cat(mlp_parts).std())
        del X, lr_parts, mlp_parts
        torch.cuda.empty_cache()
        a = flr(F[q])
        m = torch.stack([f(F[q]) for f in fms]).mean(0)
        scores[q] = (a / std_lr + m / std_mlp).cpu().numpy()
        print(f"[{i + 1}/{len(subs)}] {q} {metrics(D[q][1], scores[q])} {time.time() - t0:.0f}s", flush=True)
        del flr, fms
        torch.cuda.empty_cache()
    np.savez_compressed(SCORES, subjects=np.array(subs), **scores)
    return subs, D, scores


# ---------------------------------------------------------------- 后处理候选
def const(B, T, d):
    return torch.full((B, T), -math.log(d), device=DEV)


def hmm(S, dwell=300, temp=1.0):
    return E44.hmm(S / temp, const(*S.shape, dwell))


def ema_std(S, hl):
    m = E44.ema_mean(S, hl)
    v = (E44.ema_mean(S ** 2, hl) - m ** 2).clamp_min(1e-6)
    return v.sqrt()


def causal_quantile(s, W):
    T = s.shape[0]
    i = torch.arange(T, device=DEV)
    mask = (i[None, :] <= i[:, None]) & (i[:, None] - i[None, :] < W)
    less = ((s[None, :] < s[:, None]) & mask).sum(1).float()
    eq = ((s[None, :] == s[:, None]) & mask).sum(1).float()
    n = mask.sum(1).float()
    return torch.logit((less + 0.5 * eq + 0.5) / (n + 1))


def post_variants(subs, s42):
    """返回 {名称: {被试: 分数}}。所有变换只用当前及过去分数。"""
    S = E44.pad(subs, s42)
    B, T = S.shape
    lens = [len(s42[q]) for q in subs]
    Z = S - E44.ema_mean(S, 1200)
    V = {"raw": S, "hmm_d300": hmm(S), "dt1200_hmm": hmm(Z)}
    # 1) 去趋势后按因果 EMA 尺度归一化（低组后段分数方差收缩）
    for hl in (600, 1200):
        V[f"dt1200_z{hl}_hmm"] = hmm(Z / ema_std(Z, hl))
    # 2) 去趋势 + 因果分位数归一化（对单调漂移不敏感）
    for W in (600, 1200):
        Q = torch.zeros_like(S)
        for i in range(len(subs)):
            Q[i, :lens[i]] = causal_quantile(S[i, :lens[i]], W)
        V[f"dt1200_q{W}_hmm"] = hmm(0.5 * (Z + Q))
    # 3) 温度：低组后段分数被压缩，降温放大后验
    for t in (0.5, 0.7):
        V[f"dt1200_hmm_t{t}"] = hmm(Z, temp=t)
    # 4) 驻留时间
    for d in (150, 600):
        V[f"dt1200_hmm_d{d}"] = hmm(Z, dwell=d)
    # 5) 去趋势强度（混合原始分数）
    for r in (0.5, 0.75):
        V[f"mix_dt1200_r{r}_hmm"] = hmm(S - r * E44.ema_mean(S, 1200))
    # 6) 双半衰期去趋势
    V["dt600_1200_hmm"] = hmm(0.5 * ((S - E44.ema_mean(S, 600)) + Z))
    return {k: {q: v[i, :lens[i]].cpu().numpy() for i, q in enumerate(subs)} for k, v in V.items()}


def rows_of(D, qs, sc):
    full = [metrics(D[q][1], sc[q]) for q in qs]
    late = [metrics(D[q][1][D[q][2] >= 4], sc[q][D[q][2] >= 4]) for q in qs]
    return full, late


def decompose(subs, rows, grp):
    out = {"all": summary(rows)}
    for name, keep in (("high", lambda q: grp[q] == "high"), ("low", lambda q: grp[q] == "low"),
                       ("rev4", lambda q: q in E44.REV)):
        out[name] = summary([r for q, r in zip(subs, rows) if keep(q)])
    return out


def stage_eval(subs, D):
    grp = E44.groups()
    data = np.load(SCORES)
    s42 = {q: data[q] for q in subs}
    cands = post_variants(subs, s42)
    res = {}
    for k, sc in cands.items():
        full, late = rows_of(D, subs, sc)
        res[k] = {"full": decompose(subs, full, grp), "late45": decompose(subs, late, grp),
                  "per_full_bacc": [r["bacc"] for r in full], "per_late_bacc": [r["bacc"] for r in late]}
        f, l = res[k]["full"]["all"], res[k]["late45"]["all"]
        print(f"{k:22s} full {f['bacc']:.4f}/{f['auc']:.4f}/{f['f1']:.4f} n90={f['n90']} "
              f"late {l['bacc']:.4f}/{l['auc']:.4f}/{l['f1']:.4f} "
              f"H={res[k]['full']['high']['bacc']:.3f} L={res[k]['full']['low']['bacc']:.3f} "
              f"Llate={res[k]['late45']['low']['bacc']:.3f}", flush=True)
    # 嵌套 LOSO：目标排除本人，用其余 56 人 (全帧 bACC + 块4–5 bACC)/2 选变体
    keys = list(res)
    E = np.array([[0.5 * (a + b) for a, b in zip(res[k]["per_full_bacc"], res[k]["per_late_bacc"])] for k in keys])
    chosen = [keys[int(np.argmax(np.delete(E, i, axis=1).mean(1)))] for i in range(len(subs))]
    nf, nl = [], []
    for i, q in enumerate(subs):
        sc = cands[chosen[i]][q]
        y, b = D[q][1], D[q][2]
        nf.append(metrics(y, sc))
        nl.append(metrics(y[b >= 4], sc[b >= 4]))
    res["nested_select"] = {"full": decompose(subs, nf, grp), "late45": decompose(subs, nl, grp),
                            "chosen_counts": {k: chosen.count(k) for k in set(chosen)}}
    for k in res:
        res[k].pop("per_full_bacc", None)
        res[k].pop("per_late_bacc", None)
    RESULT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("nested", res["nested_select"]["chosen_counts"], res["nested_select"]["full"]["all"],
          res["nested_select"]["late45"]["all"], flush=True)


# ---------------------------------------------------------------- 部署核验
def stage_verify(sid):
    sys.path.insert(0, str(ROOT / "src"))
    import infer
    import streaming as S

    subs, D = L.load()
    data = np.load(SCORES)
    scores = {q: data[q] for q in subs}
    ref = post_variants(subs, scores)["dt1200_hmm"]
    mine = {}
    for q in subs:
        post = infer.PostProcessor(detrend_halflife=1200)
        mine[q] = np.array([post.update(float(s)) for s in scores[q]])
    diff = max(float(np.abs(mine[q] - ref[q]).max()) for q in subs)
    sign = float(np.mean(np.concatenate([(mine[q] > 0) == (ref[q] > 0) for q in subs])))
    grp = E44.groups()
    full, late = rows_of(D, subs, mine)
    pick = lambda r: {k: round(r[k], 4) for k in ("bacc", "auc", "f1")}  # noqa: E731
    out = {"reproduce": {"max_abs_vs_e44": diff, "sign_agree": sign,
                         "full": pick(decompose(subs, full, grp)["all"]),
                         "late45": pick(decompose(subs, late, grp)["all"])}}
    print("[reproduce]", out["reproduce"], flush=True)

    # 分数方向核验：同一批分数分别按 +s / -s 送入部署后处理，比较宏 bACC
    def macro_bacc(sc):
        return float(np.mean([metrics(D[q][1], sc[q])["bacc"] for q in subs]))

    out["direction"] = {"plus": macro_bacc(mine),
                        "minus": macro_bacc({q: -mine[q] for q in subs})}
    print("[direction]", out["direction"], flush=True)

    # 流式：真实 BDF，前缀不变性与分块一致性
    bundle = torch.load(S.BUNDLE, map_location="cpu", weights_only=False)
    bdf = next((ROOT / "EEG_dataset" / sid).glob("*.bdf"))
    rec = infer.load_bdf(bdf)
    fs = rec["fs"]
    sig = np.asarray(rec["signal_uv"], dtype=float) * 1e-6
    sig[~rec["valid"]] = np.nan

    def stream(x, chunk):
        dec = S.StreamingInference(bundle, fs=fs, step=1.0)
        return infer.stream_outputs(dec, infer.PostProcessor(detrend_halflife=1200), x, fs, chunk)

    ref_out = stream(sig, len(sig))
    s_ref = np.array([r["score_task"] for r in ref_out])
    pre = stream(sig[:int(1500 * fs)], 250)
    out["streaming"] = {"n_frames": len(ref_out),
                        "prefix": {"n": len(pre), "identical": pre == ref_out[:len(pre)],
                                   "max_abs": float(np.abs(np.array([r["score_task"] for r in pre])
                                                           - s_ref[:len(pre)]).max())},
                        "chunks": {}}
    for name, c in (("250pt", 250), ("137pt", 137), ("0.37s", 92), ("5s", 1250)):
        r = stream(sig, c)
        out["streaming"]["chunks"][name] = {"n": len(r), "identical": r == ref_out,
                                            "max_abs": float(np.abs(np.array([p["score_task"] for p in r])
                                                                    - s_ref).max()) if len(r) == len(ref_out) else None}
    print("[streaming]", json.dumps(out["streaming"], ensure_ascii=False), flush=True)
    (HERE / "verify.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("train", "eval", "verify"))
    ap.add_argument("--subject", default="018")
    a = ap.parse_args()
    t0 = time.time()
    if a.stage == "train":
        train_scores()
    elif a.stage == "eval":
        subs, D = L.load()
        stage_eval(subs, D)
    else:
        stage_verify(a.subject.zfill(3))
    print(f"elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
