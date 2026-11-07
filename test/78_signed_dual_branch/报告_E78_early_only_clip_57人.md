# E78 early-only clip ratio 57人验证

严格复用 E76 逐折缓存。仅对 early block<=2 使用 clip ratio 幅度调节；late block>=4 及其他 block 保留 w=.5 原始 score，统一因果 EMA1200 + forward HMM300。

## 协议

- raw = `0.5 * (lr_scaled + mlp_scaled)`，全程固定 raw 符号。
- early block<=2：`|conditioned|/|raw|` 裁剪到 `[0.75,1.25]`。
- block>=3 不调节，特别是 late block>=4 严格保留基线。

## 宏观指标

|方法|阶段|AUC|F1|bACC|
|---|---|---:|---:|---:|
|w0.50|full|0.835171|0.776439|0.784831|
|w0.50|early|0.849689|0.729766|0.797063|
|w0.50|late|0.759572|0.712027|0.726379|
|e78_early_only_clip|full|0.839223|0.779609|0.786471|
|e78_early_only_clip|early|0.854412|0.732305|0.799938|
|e78_early_only_clip|late|0.759159|0.713208|0.724465|

## 相对 w=.5 逐被试改善/退步/持平

|阶段|AUC|F1|bACC|
|---|---|---|---|
|full|36/20/1|34/20/3|31/23/3|
|early|31/25/0|27/26/3|29/24/3|
|late|16/39/2|28/16/13|25/21/11|

## 结论

early-only clip 的 full AUC/F1/bACC = 0.839223/0.779609/0.786471；late AUC/F1/bACC = 0.759159/0.713208/0.724465。
由于 late 区间逐点保持 raw score，理论上其未经过分块调节；最终指标仍统一经过同一因果后处理。

逐折产物见 `e78_early_only_57v1_fold_*.npz`，完整JSON见 `e78_early_only_57_results.json`。