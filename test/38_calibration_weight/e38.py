"""E38：扩展本人校准权重 alpha>0.75 的固定通用池化实验。

模型、特征和因果去均值与 E31 完全一致，只扩展 alpha；不使用目标测试标签训练。
配置/阈值的折外评价由其余56人的测试bACC决定。034 回退为纯池化。

运行：.venv\Scripts\python.exe test/38_calibration_weight/e38.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
T = HERE.parent
sys.path.insert(0, str(T / "17_compliant"))


def imp(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


G = imp("e31general", T / "31_general_algorithm" / "general.py")
E26 = G.E26
E27 = G.E27
E28 = imp("e28lib", T / "28_block_invariant" / "e28.py")
ALPHAS = (0.75, 0.85, 0.9, 0.95, 1.0)
M300 = [x[0] for x in E26.THR2].index("m300")


def summary(a: np.ndarray) -> dict:
    return {"bacc": float(np.nanmean(a[:, 0])), "auc": float(np.nanmean(a[:, 1])),
            "f1": float(np.nanmean(a[:, 2])), "n90": int(np.nansum(a[:, 0] >= .9)),
            "min_bacc": float(np.nanmin(a[:, 0]))}


def main() -> None:
    subs, raw, features = G.prepare_features()
    pool = G.prepare_pool(subs, raw, features)
    ev = np.full((len(subs), len(ALPHAS), len(E26.THR2), 3), np.nan, np.float32)
    scores = {}
    for i, q in enumerate(subs):
        calibration = torch.tensor(raw[q][2] == 3 if q == G.SINGLE_CLASS_CAL else raw[q][2] <= 3,
                                   device=G.DEV)
        other = pool["base_weight"] * (pool["sid"] != i).float() / (len(subs) - 1)
        own = torch.zeros_like(other)
        start, end = int(pool["offsets"][i]), int(pool["offsets"][i + 1])
        if q != G.SINGLE_CLASS_CAL:
            own[start:end] = E27.bal_w(raw[q][1], calibration)
        if q == G.SINGLE_CLASS_CAL:
            # 034 没有 relax 校准，所有 alpha 必须完全相同地回退为纯池化；
            # 不能乘以 1-alpha，否则会改变有效 L2 正则强度，alpha=1 更会得到零模型。
            sw = other[:, None].repeat(1, len(ALPHAS))
        else:
            sw = torch.stack([(1 - a) * other + a * own for a in ALPHAS], 1)
        w, b = E28.lbfgs_multi(pool["x"], pool["y"], sw, G.LAM, iters=150)
        s = features[q] @ w.T + b
        s = torch.stack([E27.zc(s[:, j], calibration) for j in range(len(ALPHAS))])
        block = raw[q][2]
        tst = torch.tensor(np.where(block >= 4)[0], device=G.DEV)
        yy = torch.tensor(raw[q][1][block >= 4], device=G.DEV).bool()
        ev[i] = E26.metrics(s, tst, yy).cpu().numpy()
        scores[f"s{q}"] = s.half().cpu().numpy()
        print(f"[{i + 1:02d}/57] {q} " + " ".join(
            f"a{a:g}={ev[i, j, M300, 0]:.3f}" for j, a in enumerate(ALPHAS)), flush=True)
        del sw, w, b, s
        torch.cuda.empty_cache()
    np.savez_compressed(HERE / "e38_ev.npz", EV=ev, subs=subs, alphas=ALPHAS)
    np.savez_compressed(HERE / "e38_scores.npz", subs=subs, alphas=ALPHAS, **scores)

    fixed_m300 = ev[:, :, M300]
    mean = np.nanmean(fixed_m300[..., 0], 0)
    best_alpha = int(np.nanargmax(mean))
    nested_alpha = []
    nested_all = []
    for i in range(len(subs)):
        obj = np.nanmean(np.delete(fixed_m300[..., 0], i, 0), 0)
        nested_alpha.append(fixed_m300[i, int(np.nanargmax(obj))])
        obj_all = np.nanmean(np.delete(ev[..., 0], i, 0), 0)
        a, t = np.unravel_index(int(np.nanargmax(obj_all)), obj_all.shape)
        nested_all.append(ev[i, a, t])
    oracle_flat = np.nanargmax(np.where(np.isnan(ev[..., 0]), -np.inf, ev[..., 0]).reshape(len(subs), -1), 1)
    oracle = ev.reshape(len(subs), -1, 3)[np.arange(len(subs)), oracle_flat]
    result = {
        "protocol": "E31特征和池化LR；扩展alpha；目标测试标签不参与训练，LOSO选择由其余56人完成",
        "alphas": ALPHAS,
        "fixed_m300_all57_selection": {"alpha": ALPHAS[best_alpha], **summary(fixed_m300[:, best_alpha])},
        "loso_alpha_fixed_m300": summary(np.asarray(nested_alpha)),
        "nested_loso_alpha_threshold": summary(np.asarray(nested_all)),
        "test_oracle": summary(oracle),
        "per_subject_m300": {
            q: {f"a{a:g}": fixed_m300[i, j].tolist() for j, a in enumerate(ALPHAS)}
            for i, q in enumerate(subs)
        },
    }
    (HERE / "e38.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for key in ("fixed_m300_all57_selection", "loso_alpha_fixed_m300", "nested_loso_alpha_threshold", "test_oracle"):
        v = result[key]
        print(f"{key}: bACC={v['bacc']:.4f} AUC={v['auc']:.4f} F1={v['f1']:.4f} "
              f"n90={v['n90']} min={v['min_bacc']:.4f}")


if __name__ == "__main__":
    main()
