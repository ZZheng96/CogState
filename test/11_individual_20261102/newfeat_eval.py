"""E14 评估：19 维新特征（跨带 log-Euclidean 协方差 + PAC + 相位一致性）在逐人协议下的 bACC。

协议与 `subject_multi1.py` / `workup2.py` 一致（关键：不另创新协议）：
  - 每个被试独立，按 blk（trial）留一块（LOBO）：训练块 = 其余 5 块，被评块不参与标准化/选特征/拟合；
  - 特征级因果标准化 `running_stats(F, w=300)`（只用过去 300 帧）；
  - 模型：StandardScaler + SelectKBest(f_classif, k∈{5,10,19}) + LogisticRegression(C=0.01)；
  - 每折决策分数用**训练折**的均值/标准差标准化；极性用训练折分数判定
    （task 均值 > relax 均值则 +1，否则 −1）后再作用到被评块；
  - 个体记忆 ema(s, h)，h∈{2,4,8,12,20,30,40} 秒；阈值 0。

对比两套特征：
  new19      仅 19 维新特征
  new19+pow  19 维 + 已有 5 个频带 log 功率指数（δ/θ/α/β/γ，取自 sliding1 缓存）
诊断用（同一协议，判断"新信息"）：
  pow5       仅 5 个频带 log 功率（新特征的对角线与其近似重复）
  nonpow14   去掉 5 个对角线后的 14 维（10 个跨带协方差 + 2 PAC + 2 相位一致性）
基线：workup2.json 的逐人被试最优 bACC（23 人 ≥0.90）。

产物：newfeat_eval.json / newfeat_eval.log
运行：py -3.11 test/11_individual_20261102/newfeat_eval.py
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import lfilter
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from feat_cstd import running_stats

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NEW = HERE / "newfeat"
SLID1 = HERE / "sliding1"
BASE = HERE / "workup2.json"
KS = (5, 10, 19)
HS = (2, 4, 8, 12, 20, 30, 40)
C_REG = 0.01
CSTD_W = 300


def ema(x, h):
    if h <= 1:
        return x.copy()
    a = 0.5 ** (1.0 / h)
    return lfilter([1 - a], [1, -a], x, zi=[a * x[0]])[0]


def lobo_scores(Xc, y, blk, k):
    """留一块：训练折拟合，训练折分数定标定极性，作用到被评块。"""
    s = np.full(len(y), np.nan)
    for b in np.unique(blk):
        te, tr = blk == b, blk != b
        if len(np.unique(y[tr])) < 2:
            s[te] = 0.0
            continue
        sc = StandardScaler().fit(Xc[tr])
        A, B = sc.transform(Xc[tr]), sc.transform(Xc[te])
        sel = SelectKBest(f_classif, k=min(k, A.shape[1])).fit(A, y[tr])
        A2, B2 = sel.transform(A), sel.transform(B)
        m = LogisticRegression(C=C_REG, max_iter=3000).fit(A2, y[tr])
        dtr, dte = m.decision_function(A2), m.decision_function(B2)
        mu, sd = dtr.mean(), dtr.std() + 1e-12
        ztr = (dtr - mu) / sd
        sign = 1.0 if ztr[y[tr] == 1].mean() > ztr[y[tr] == 0].mean() else -1.0
        s[te] = sign * (dte - mu) / sd
    return s


def eval_set(X, y, blk):
    X = np.nan_to_num(X.astype(np.float32))
    m, sd = running_stats(X, w=CSTD_W)
    Xc = (X - m) / sd
    rec, best = {}, None
    for k in KS:
        s = lobo_scores(Xc, y, blk, k)
        for h in HS:
            e = ema(s, h)
            p = (e >= 0).astype(int)
            r = {"bACC": float(balanced_accuracy_score(y, p)),
                 "AUC": float(roc_auc_score(y, e)),
                 "F1": float(f1_score(y, p))}
            rec[f"k{k}_h{h}"] = r
            if best is None or r["bACC"] > best["bACC"]:
                best = {**r, "k": k, "h": h, "key": f"k{k}_h{h}"}
    return best, rec


def main():
    man = json.loads((ROOT / "database" / "manifest.json").read_text(encoding="utf-8"))
    names = man["feature_names"] + man["emg_feature_names"]
    pow_ix = [names.index(f"stft_log_power_{b}_mean") for b in ("1_4", "4_8", "8_13", "13_30", "30_45")]
    base = json.loads(BASE.read_text(encoding="utf-8"))

    subs = sorted(p.stem.replace("sub-", "") for p in NEW.glob("sub-*.npz"))
    out, rows = {}, []
    lines = []
    t0 = time.perf_counter()

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    log(f"[eval] subjects={len(subs)} cstd_w={CSTD_W} ks={KS} hs={HS} C={C_REG}")
    for q in subs:
        S = np.load(NEW / f"sub-{q}.npz")
        B = np.load(SLID1 / f"sub-{q}.npz")
        assert np.array_equal(S["y"], B["y"]) and np.array_equal(S["blk"], B["blk"]) \
            and np.array_equal(S["t"], B["t"]), f"{q}: alignment broken"
        o = np.argsort(S["t"])
        X19 = S["F"][o].astype(np.float32)
        y, blk = S["y"][o].astype(int), S["blk"][o].astype(int)
        P = np.nan_to_num(B["F"][o].astype(np.float32))[:, pow_ix]
        if len(np.unique(y)) < 2:
            log(f"[skip] {q} single class")
            continue

        b19, r19 = eval_set(X19, y, blk)
        b_all, rall = eval_set(np.hstack([X19, P]), y, blk)
        b_pow, rpow = eval_set(P, y, blk)
        b_np, rnp = eval_set(X19[:, np.r_[5:15, 15:19]], y, blk)
        base_b = float(base[q]["best_bACC"]) if q in base else float("nan")
        out[q] = {"new19": b19, "grid19": r19, "new19_pow": b_all, "grid19_pow": rall,
                  "pow5": b_pow, "nonpow14": b_np, "baseline_bACC": base_b}
        rows.append((q, b19["bACC"], b_all["bACC"], base_b, b_pow["bACC"], b_np["bACC"]))
        log(f"{q:4s} new19 {b19['bACC']:.3f} ({b19['key']}) | +pow {b_all['bACC']:.3f} "
            f"({b_all['key']}) | pow5 {b_pow['bACC']:.3f} | nonpow14 {b_np['bACC']:.3f} "
            f"| base {base_b:.3f}")

    a19 = np.array([r[1] for r in rows])
    aal = np.array([r[2] for r in rows])
    ab = np.array([r[3] for r in rows])
    apw = np.array([r[4] for r in rows])
    anp = np.array([r[5] for r in rows])

    def stats(a):
        return {"n": int(len(a)), "mean": float(np.mean(a)), "median": float(np.median(a)),
                "ge90": int((a >= .90).sum()), "ge85": int((a >= .85).sum())}

    s19, sall, sbase = stats(a19), stats(aal), stats(ab)
    spw, snp = stats(apw), stats(anp)
    new90 = lambda i: [r[0] for r in rows if r[i] >= .90 and r[3] < .90]
    lost90 = lambda i: [r[0] for r in rows if r[i] < .90 and r[3] >= .90]
    new85 = lambda i: [r[0] for r in rows if r[i] >= .85 and r[3] < .85]
    changes = {"new19_ge90_new": new90(1), "new19_ge90_lost": lost90(1),
               "new19_pow_ge90_new": new90(2), "new19_pow_ge90_lost": lost90(2),
               "new19_ge85_new": new85(1), "new19_pow_ge85_new": new85(2)}

    log("")
    for tag, s in (("new19", s19), ("new19+pow", sall), ("pow5(诊断)", spw),
                   ("nonpow14(诊断)", snp), ("baseline(workup2)", sbase)):
        log(f"[{tag:17s}] mean={s['mean']:.3f} median={s['median']:.3f} "
            f"n>=.90={s['ge90']} n>=.85={s['ge85']}")
    log(f"[new19 >=.90 新过线] {','.join(changes['new19_ge90_new']) or '-'}")
    log(f"[new19 >=.90 掉线]   {','.join(changes['new19_ge90_lost']) or '-'}")
    log(f"[new19+pow >=.90 新过线] {','.join(changes['new19_pow_ge90_new']) or '-'}")
    log(f"[new19+pow >=.90 掉线]   {','.join(changes['new19_pow_ge90_lost']) or '-'}")

    out["summary"] = {"new19": s19, "new19_pow": sall, "pow5": spw, "nonpow14": snp,
                      "baseline": sbase,
                      "baseline_claim_ge90": 23, "changes": changes,
                      "runtime_s": round(time.perf_counter() - t0, 1),
                      "params": {"ks": list(KS), "hs": list(HS), "C": C_REG, "cstd_w": CSTD_W,
                                 "protocol": "LOBO by blk; train-fold mean/std + train-fold polarity"}}
    log(f"[done] {out['summary']['runtime_s']:.0f}s -> newfeat_eval.json")
    (HERE / "newfeat_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
    (HERE / "newfeat_eval.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[done] newfeat_eval.json")


if __name__ == "__main__":
    main()
