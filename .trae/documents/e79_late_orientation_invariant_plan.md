# E79 后段方向不变专家实验计划

## Context

E74/E53 固定 LR/MLP 融合是当前稳定基线，但部分被试在后段的原始模型输出已经反向。E76/E77 的自由条件化门控虽提升 full AUC，却反复损害 late AUC；E78 的有界幅度调节能改善 full 指标，但 late AUC 仍未稳定超过基线。因此下一步不再让目标被试的早期表现决定后段方向，而是冻结早期输出，在后段独立验证“有符号方向失效但输出幅度仍保留状态信息”的假设。

目标：严格 57 人 LOSO 下，验证一个低复杂度、固定融合、严格因果的后段方向不变专家；若不能同时改善后段和全段指标，则终止 score 后处理/门控路线，转向训练阶段的方向反转增强。

## 推荐方案

### 1. 新实验目录与产物

创建：`test/79_late_orientation_invariant/`

产物：
- `e79.py`
- `protocol.json`
- `e79_results.json`
- `报告_E79_后段方向不变专家.md`
- 每折 `fold_*.npz` 与必要的逐折 JSON

不修改 `src`、原始数据和既有 E74-E78 文件。

### 2. 数据和基线口径

复用 `test/76_conditioned_fusion/fold_*.npz` 的逐折组件缓存，检查目标被试不在训练集合中，并保持时间排序。

主基线直接使用 E76 缓存中的 `score_w0.50`，不对已后处理分数重复执行后处理。报告中明确基线来源，避免与重新对 `.5*(lr_scaled+mlp_scaled)` 后处理得到的另一口径混用。统一报告 full/early/late 的 AUC、F1、bACC。

### 3. 严格 LOSO 的 late expert

每个目标被试使用其余 56 人，仅取 `block >= 4` 的训练帧，构造方向不变特征：

`[abs(lr_scaled), abs(mlp_scaled), abs(lr_scaled - mlp_scaled)]`

在训练折上估计均值和标准差，并用相同参数变换目标被试；不得使用目标被试任何标签、全段统计或未来统计量。使用固定低容量 LogisticRegression：`C=0.1`、`class_weight=balanced`、`solver=liblinear`、`max_iter=300`、`random_state=0`。输出训练折标准差归一化的连续 `u_abs_scaled`。

### 4. 预注册比较方案

- `fixed_w05`：完整复用 E76 的 `score_w0.50`。
- `late_mlp`：early 和 block 3 使用 fixed_w05，late 使用 `2*mlp_scaled`，仅作组件抵消诊断。
- `late_abs_only`：early 和 block 3 使用 fixed_w05，late 使用 `u_abs_scaled`。
- `late_abs_blend025`：early 和 block 3 使用 fixed_w05，late 使用 `0.75*fixed_w05 + 0.25*u_abs_scaled`，作为主候选。

只允许使用预先固定的方案，不在 57 人标签结果上再搜索融合比例。

### 5. 因果约束

先拼接每折的整段候选分数，再统一复用 E76 的因果 EMA1200 + forward HMM300 实现。不得使用双向滤波、完整目标序列归一化或目标标签。明确 block 边界：`block<=2` 冻结基线，`block==3` 保持基线，`block>=4` 才启用 late expert。

实现前缀不变性检查：修改目标序列后缀时，前缀输出只能有浮点误差范围内变化；记录训练标准化参数和目标被试排除检查。

## 验证与停止标准

按逐被试宏平均报告：
- full/early/late AUC、F1、bACC；
- 相对 fixed_w05 的逐被试改善/退步/持平；
- 重点被试 004、009、010、018、034、050 的 late 指标；
- `u_abs_scaled` 与 signed score 的 late 相关性；
- block 3 边界是否出现异常跳变。

只有同时满足以下条件才保留为候选：
- late AUC、F1、bACC 相对基线均不下降（允许不超过 0.001 的浮点误差）；
- 且 `late AUC >= 0.765`，或 full AUC 至少达到 0.836 且 late AUC 不下降；
- 个体改善不能仅由少数被试贡献，重点反转被试至少两个获得明显 late AUC 改善，或 late AUC 改善人数多于退步人数。

若后段任一主要指标下降、绝对专家接近随机、结果高度依赖融合比例，或因果检查失败，则停止 score 后处理/门控方向，不继续增加阈值和比例网格，转向训练阶段方向反转增强。

## 实施与验证命令

```powershell
D:\WORK\BCIC2026\CogState\.venv\Scripts\python.exe -B D:\WORK\BCIC2026\CogState\test\79_late_orientation_invariant\e79.py
```

执行后检查 `protocol.json`、逐折缓存、汇总 JSON 与中文报告，确认 57 折完整、无目标泄漏、指标口径一致。