# `LARA/utils/` 深入总结

## 目录职责

这个目录提供训练与评估所需的通用工具：

- `tools.py`：early stopping、learning-rate adjustment、可视化等。
- `metrics.py`：MSE、MAE、RMSE、MAPE、MSPE。
- `timefeatures.py`：时间特征编码。
- `augmentation.py`：时间序列增强。
- `dtw_metric.py`：DTW 评估。
- `print_args.py`：打印实验参数。

## 与 LARA 的关系

LARA 的核心 diagnostics 不在 `utils/`，而在 `LARA/LARA/modules.py` 和 `exp/exp_long_term_forecasting.py` 中打印。`utils` 主要保证普通 TSF-Lib 训练流程能运行。

## 需要注意

正式 LARA 实验应至少记录：

- final MSE/MAE：来自 `metrics.py`。
- LARA rank/gate/retrieval diagnostics：当前只打印，不保存。
- 可视化 PDF：`visual()` 会在 test results 中保存预测图，但这不是 retrieval 证据。

## 建议补强

后续可以在 `utils/` 新增一个 `lara_logging.py`，统一写入 per-epoch diagnostics 和 per-query retrieval analysis。这样实验报告才能稳定复现 help/harm split、gate calibration、active K distribution。
