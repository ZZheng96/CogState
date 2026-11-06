# E59 后段性能软适应改进计划

## Context

当前严格 LOSO 主链为 E17 融合分数 → causal detrend（halflife=1200）→ forward HMM（dwell=300），全段 bACC/AUC/F1 为 0.7824/0.8315/0.7742，块 4–5 为 0.7282/0.7601/0.7130，低组后段明显偏低。已有 E57/E58 表明 block 边界硬重置收益高度个体依赖，因此本次仅在已有 E53 LOSO 分数上验证连续、严格因果的软适应，不修改模型训练和正式主链。

## 推荐实现

1. 在 [late_solution.py](file:///d:/WORK/BCIC2026/CogState/test/59_late_performance_solution/late_solution.py) 新增独立实验脚本，复用：
   - [e53.py](file:///d:/WORK/BCIC2026/CogState/test/53_e17_post_loso/e53.py) 的缓存、指标和分组口径；
   - [e44.py](file:///d:/WORK/BCIC2026/CogState/test/44_late_drift/e44.py) 的因果 EMA/HMM；
   - [loso.py](file:///d:/WORK/BCIC2026/CogState/test/17_compliant/loso.py) 的标签与 block 数据。
2. 保留 raw、hmm_d300、dt1200_hmm 作为对照。新增双时间尺度候选：长期半衰期 1200，短期半衰期 60/120/300/600，短期权重 0.10/0.25/0.50，correction 限幅 0.10/0.25/0.50；所有候选之后统一进入 HMM dwell=300。
3. 新增持续漂移触发候选：使用当前及过去分数计算 short-long gap 与因果方差归一化，阈值 1.0/1.5/2.0；连续满足 60 帧后才触发，权重缓慢上升/下降（alpha_up=1/60、alpha_down=1/120），最大权重 0.25/0.50，限幅同上。触发只改变短期基线权重，不能重置 detrend/HMM、切换方向或读取 block/标签/未来。
4. 处理时每个被试独立维护状态，避免 padding 污染；优先使用 CUDA，CPU 作为回退。记录触发率、平均/最大权重和最大修正幅度。
5. 输出 `late_solution.json`、`late_solution.csv` 和 `报告_E59_后段性能软适应实验.md`。报告必须包含全段、块 4–5、低组/高组、指定个体 004/009/010/018/050、最差个体、触发率，以及每项 bACC/AUC/F1。
6. 同时提供固定候选结果和排除目标被试后的嵌套 LOSO 选择结果；块号仅用于离线评价/选择，绝不进入在线门控。

## 筛选标准

- 全段 bACC ≥ 0.7774、AUC ≥ 0.8265、F1 ≥ 0.7692；
- 块 4–5 bACC 至少提升 0.01，且低组后段确有改善；
- 不能出现 010 型灾难性退化，且保护 018、004、009、050；
- 触发率不能接近常开或几乎为零；
- 不满足全部标准则不修改正式主链。

## 验证

运行：

```powershell
.venv\Scripts\python.exe test/59_late_performance_solution/late_solution.py
```

检查 JSON/CSV 和 Markdown 中的指标、候选配置、因果约束、触发率与指定个体结果。若出现通过筛选的候选，再修改 [infer.py](file:///d:/WORK/BCIC2026/CogState/src/infer.py) 实现对应在线状态机，并使用 E53/E58 的前缀一致性、不同 chunk 大小和真实 BDF 流式验证；否则维持现有 E17→dt1200→HMM 主链。
