"""构建 E17 跨被试 5 折（out-of-fold）预测，供查看器的 "e17_cv_*" 方案使用。

外层划分与 test/17_compliant/multiscale.py 相同：perm = RandomState(0).permutation(subs)，
outer = perm[k::5]。每折用其余 4 折训练 bundle（与 model_bundle.pt 同配方），再对本折被试
按 src/infer.py 口径流式推理，结果写入 system/web/cache/e17_cv/pred_<sid>.npz。

用法：python system/web/build_oof.py
"""
from __future__ import annotations

import json

import numpy as np

import model_eval as ME

OUT = ME.CACHE / "e17_cv"
BUNDLES = OUT / "bundles"


def main():
    import streaming as ST
    subs = sorted(p.stem[4:] for p in ST.SLID.glob("sub-*.npz"))
    perm = list(np.random.RandomState(0).permutation(subs))
    outer = [perm[k::5] for k in range(5)]
    BUNDLES.mkdir(parents=True, exist_ok=True)
    folds = {}
    for k, test in enumerate(outer):
        bp = BUNDLES / f"fold{k}.pt"
        if not bp.exists():
            ST.train_bundle(out_path=bp, subs=[s for s in subs if s not in test])
        for sid in test:
            folds[sid] = k
            f = OUT / f"pred_{sid}.npz"
            if not f.exists():
                d = ME.compute(sid, bundle_path=bp, out=f)
                print(k, sid, json.dumps({"raw": ME.metrics(d["y"], d["raw"]), "post": ME.metrics(d["y"], d["post"])}), flush=True)
    (OUT / "folds.json").write_text(json.dumps(folds, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
