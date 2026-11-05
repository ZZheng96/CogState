"""E1 / E46：LOSO 下的因果在线被试归一化。

主结果固定 C=.001，不使用目标被试标签。score 越高越倾向 task。
同时保留原始分数和固定 center300 后处理，供 E2/E3 复用。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "test" / "17_compliant"))
import loso as L  # noqa: E402

SUBS = []
LATE = 300  # 前 45 个 6 s 窗 = 270 s；按窗索引 45*6+30 严格等价于 300。
VARIANTS = ("raw", "cumulative", "ema600", "w600")


def feature_set(D, subs, variant):
    if variant == "raw":
        return {q: D[q][0].clone() for q in subs}
    if variant in {"cumulative", "w600"}:
        return L.feats(D, subs, 600 if variant == "w600" else 99999)
    if variant == "ema600":
        out = {}
        a = 1 - 0.5 ** (1 / 600)
        for q in subs:
            x = D[q][0].float()
            z = torch.empty_like(x)
            mean = x[0].clone()
            mean2 = x[0].square().clone()
            z[0] = 0
            for t in range(1, len(x)):
                mean = mean + a * (x[t] - mean)
                mean2 = mean2 + a * (x[t].square() - mean2)
                var = (mean2 - mean.square()).clamp_min(0)
                z[t] = (x[t] - mean) / torch.sqrt(var + 1e-8)
            out[q] = z
        return out
    raise ValueError(variant)


def causal_center(x, width=300):
    """只用当前及过去 width 帧的中位数；GPU 计算，CPU 返回。"""
    a = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=L.dev)
    n = len(a)
    pad = torch.full((width,), float("nan"), device=L.dev)
    z = torch.cat([pad, a])
    win = z.unfold(0, width + 1, 1)
    center = torch.nanquantile(win, 0.5, dim=1)
    return (a - center).cpu().numpy()


def metrics(y, score, late=False):
    y = np.asarray(y, dtype=int)
    s = np.asarray(score, dtype=float)
    p = (s > 0).astype(int)
    out = {
        "bACC": float(balanced_accuracy_score(y, p)),
        "AUC": float(roc_auc_score(y, s)),
        "F1": float(f1_score(y, p, zero_division=0)),
        "accuracy": float((p == y).mean()),
        "n": int(len(y)),
    }
    if late:
        return out
    return out


def evaluate_scores(D, scores, key):
    rows = []
    all_y, all_s, all_yl, all_sl = [], [], [], []
    for q in SUBS:
        y = D[q][1]
        s = scores[f"{key}__{q}"]
        rows.append({"subject": q, **metrics(y, s), "late": metrics(y[LATE:], s[LATE:])})
        all_y.append(y)
        all_s.append(s)
        all_yl.append(y[LATE:])
        all_sl.append(s[LATE:])
    y = np.concatenate(all_y)
    s = np.concatenate(all_s)
    yl = np.concatenate(all_yl)
    sl = np.concatenate(all_sl)
    return {
        "macro": {
            "bACC": float(np.mean([r["bACC"] for r in rows])),
            "AUC": float(np.mean([r["AUC"] for r in rows])),
            "F1": float(np.mean([r["F1"] for r in rows])),
        },
        "macro_late45": {
            "bACC": float(np.mean([r["late"]["bACC"] for r in rows])),
            "AUC": float(np.mean([r["late"]["AUC"] for r in rows])),
            "F1": float(np.mean([r["late"]["F1"] for r in rows])),
        },
        "pooled": metrics(y, s),
        "pooled_late45": metrics(yl, sl),
        "per_subject": rows,
    }


def causal_prefix_check(D, q=None, prefix=64):
    q = sorted(D)[0] if q is None else q
    entry = D[q]
    original = entry[0]
    changed = original.clone()
    changed[prefix:] = changed[prefix:] * 7.0 + 123.0
    result = {}
    for variant in VARIANTS[1:]:
        D[q] = (changed,) + entry[1:]
        a = feature_set(D, [q], variant)[q].cpu().numpy()
        D[q] = (original,) + entry[1:]
        b = feature_set(D, [q], variant)[q].cpu().numpy()
        result[variant] = float(np.max(np.abs(a[:prefix] - b[:prefix])))
    D[q] = entry
    return result


def write_report(result):
    lines = [
        "# E1 / E46 因果在线被试归一化",
        "",
        "## 结论",
        "",
        "57 人真 LOSO、纯他人训练、6 s 有效窗；累计/EMA/滑窗统计均只用当前及过去帧。主指标采用逐被试宏平均，同时给出汇总帧结果。",
        "",
        "| 特征 | 后处理 | 全段 bACC | 全段 AUC | 全段 F1 | 晚段 bACC | 晚段 AUC | 晚段 F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in result["variants"].items():
        for post in ("raw", "center300"):
            m = r[post]
            full, late = m["macro"], m["macro_late45"]
            lines.append(f"| {name} | {post} | {full['bACC']:.4f} | {full['AUC']:.4f} | {full['F1']:.4f} | {late['bACC']:.4f} | {late['AUC']:.4f} | {late['F1']:.4f} |")
    best = max(result["variants"], key=lambda k: result["variants"][k]["center300"]["macro"]["bACC"])
    bm = result["variants"][best]["center300"]["macro"]
    lines += [
        "",
        f"固定 center300 口径下，全段 bACC 最高为 **{best} = {bm['bACC']:.4f}**（AUC {bm['AUC']:.4f}，F1 {bm['F1']:.4f}）。该选择是描述性比较，不作为目标标签上的部署调参。",
        "",
        "E39 的 w600 晚段 bACC 0.7270 使用了旧阈值口径，本实验不将其当作同口径全段对照；E40 的累计特征结果可作为外部复核。",
        "",
        "## 泄漏核验",
        "",
        f"篡改第 64 帧之后的特征，前 64 帧最大变化：`{json.dumps(result['causal_prefix_check'])}`。目标被试标签从未进入训练、统计或阈值拟合。",
        "",
        "## 复现",
        "",
        "```powershell",
        "& .\\.venv\\Scripts\\python.exe test\\46_causal_online_norm\\e46.py",
        "```",
        "",
        "数值结果见 `e46.json`；逐人逐帧分数见 `e46_scores.npz`。",
    ]
    (HERE / "报告_E46_因果在线归一化.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    global SUBS
    start = time.time()
    SUBS, D = L.load()
    prefix = causal_prefix_check(D)
    scores = {"subjects": np.asarray(SUBS)}
    result = {"experiment": "E1/E46", "protocol": "57-subject LOSO", "C": 0.001,
              "causal_prefix_check": prefix, "variants": {}}
    for variant in VARIANTS:
        print(f"\n=== {variant} ===", flush=True)
        F = feature_set(D, SUBS, variant)
        for i, q in enumerate(SUBS):
            train = [s for s in SUBS if s != q]
            x_train = torch.cat([F[s] for s in train])
            y_train = np.concatenate([D[s][1] for s in train])
            model = L.fit_lr(x_train, y_train, C=0.001, iters=150)
            s = model(F[q]).cpu().numpy()
            scores[f"{variant}__{q}"] = s
            scores[f"{variant}_center300__{q}"] = causal_center(s)
            if (i + 1) % 10 == 0:
                print(f"{i + 1}/57, elapsed={time.time() - start:.1f}s", flush=True)
        for post in ("raw", "center300"):
            key = variant if post == "raw" else f"{variant}_center300"
            result["variants"].setdefault(variant, {})[post] = evaluate_scores(D, scores, key)
        print(json.dumps({k: v["macro"] for k, v in result["variants"][variant].items()}), flush=True)
        np.savez_compressed(HERE / "e46_scores.npz", **scores)
        (HERE / "e46.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["elapsed_s"] = time.time() - start
    (HERE / "e46.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(result)
    print(f"done in {result['elapsed_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
