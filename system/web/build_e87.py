"""训练集内口径的 E76：E87 全量 57 人基座分量 + 部署门控（deploy_gate.json，嵌套 LOSO 折外分数拟合）→ EMA1200 + HMM300。

输出 system/web/cache/e87_deploy/pred_{sid}.npz（t, deploy_shrink），供 schemes 训练集内对比。
"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
E87 = ROOT / "test" / "87_full_train_inference"
OUT = Path(__file__).resolve().parent / "cache" / "e87_deploy"
NESTED = ROOT / "test" / "76_conditioned_fusion" / "nested_loso" / "cache"


def main():
    spec = importlib.util.spec_from_file_location("e87", E87 / "e87.py")
    e87 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e87)
    g = json.loads((ROOT / "test" / "76_conditioned_fusion" / "deploy_gate.json").read_text(encoding="utf-8"))
    coef, b = np.asarray(g["coef"]), g["intercept"]
    OUT.mkdir(parents=True, exist_ok=True)
    for f in sorted(E87.glob("sub_*.npz")):
        d = np.load(f)
        lr, mlp = d["lr_scaled"], d["mlp_scaled"]
        raw = .5 * (e87.gate_features(lr, mlp) @ coef + b) + .5 * (lr + mlp)
        sid = f.stem[4:]
        t = np.load(next(NESTED.glob(f"fold_*_{sid}.npz")))["t"]  # 与嵌套 LOSO 缓存同一时间轴
        assert len(t) == len(raw)
        np.savez_compressed(OUT / f"pred_{sid}.npz", t=t, deploy_shrink=e87.stage(raw),
                            e87_shrink=d["score_conditioned_shrink"], w050=d["score_w0.50"])
    print("done", len(list(OUT.glob("*.npz"))))


if __name__ == "__main__":
    main()
