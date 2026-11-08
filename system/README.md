# system/ 验证系统

自包含的流式推理与评估系统。算法代码全部位于 `system/algorithms/`，系统只从这里导入，不依赖 `src/` 或 `test/`。

## 目录

| 路径 | 内容 |
|---|---|
| `algorithms/io.py` | BDF 读取、trial/BAD 区间解析（复制自 src/eeg_decoder.py、src/cogstate.py） |
| `algorithms/features.py` | E17 的 878 维窗口特征（复制自 src/preprocessing.py 与 test/01、03、08、17） |
| `algorithms/train_utils.py` | GPU 上的 LR / MLP 训练（复制自 test/17_compliant） |
| `algorithms/e17.py` | E17 bundle 训练与流式推理 StreamingInference（复制自 test/17_compliant/streaming.py） |
| `algorithms/postprocess.py` | 因果 EMA 去趋势 + 前向 HMM（复制自 src/infer.py） |
| `algorithms/fit_e76_gate.py` | 在 57 人训练集内拟合 E76 门控，写 `models/e76_gate.json` |
| `algorithms/models/` | `e17_bundle.pt`（复制自 test/17_compliant/model_bundle.pt）、`e76_gate.json` |
| `runner.py` | 方案注册（RUNNABLE）、流式回放、运行缓存 `cache/runs/` |
| `truth.py` / `evaluate.py` | 标签生成与黑盒评估（bACC/AUC/F1，阈值 0） |
| `train.py` | 用内置或导入数据重训 bundle（特征缓存 `cache/feat/`） |
| `recompute_metrics.py` | 57 人指标重算，结果 `cache/metrics_57.json` |
| `compliance.py` / `export.py` / `jobs.py` / `report.py` | 合规检查、导出提交包、后台任务、报告 |
| `web/` | 查看器服务 `viewer_server.py`、方案注册表 `schemes.py`、`model_eval.py` |

## 方案

均为 57 人全量训练、对同一批被试推理（训练集内指标，偏乐观，不代表泛化成绩）：

- `current`：E17 + EMA1200 去趋势 + HMM300
- `e17_raw`：E17 原始分数
- `e76_gated`：E17 的 LR/MLP 分量 + 因果条件化半收缩门控 + EMA1200 + HMM300

缓存用 bundle/参数文件的 SHA1 内容哈希做版本，模型文件变化后自动重算。

## 用法

```
.venv\Scripts\python.exe -m system.algorithms.fit_e76_gate --workers 4   # 拟合门控（首次会流式推理 57 人）
.venv\Scripts\python.exe -m system.recompute_metrics                     # 重算三方案指标
.venv\Scripts\python.exe system\web\viewer_server.py                     # 启动查看器
```

`export.py` 导出的提交包仍以 `src/infer.py` 为入口（赛题提交格式），它不属于系统内部导入。

指标见 `报告_57人全量训练指标.md`。
