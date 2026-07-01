# `LARA/data_provider/` 深入总结

## 目录职责

这个目录负责把标准 long-term forecasting 数据集转换成 TSF-Lib batch。LARA 的关键改动是 `IndexedDataset`：当模型名以 `LARA` 开头或包含 `GTR` 时，batch 会额外返回样本 index。

## 支持的数据

`data_factory.py` 注册了：

- `ETTh1`, `ETTh2`
- `ETTm1`, `ETTm2`
- `custom`，用于 electricity/traffic/weather 等 CSV
- `Solar`
- `PEMS`

各 Dataset 类都返回：

`seq_x, seq_y, seq_x_mark, seq_y_mark`

LARA/GTR 模型包装后返回：

`seq_x, seq_y, seq_x_mark, seq_y_mark, index`

## 为什么 sample index 关键

LARA 需要 sample index 做 temporal overlap mask。如果没有 index，训练 query 可能检索到自己或高度重叠窗口，导致 retrieval branch 得到不公平的近邻 future。GTR 也使用 sample index 计算 cycle position。

## split 与 scaling

ETT 使用固定边界；custom/Solar/PEMS 使用比例 split。scaler 只在 train 区间 fit，这是标准做法。LARA memory 构建时从 `train_data` 读取已 scale 后的 windows，因此 retrieved futures 与 host 输入输出在同一归一化空间。

## 对 LARA 实验的影响

- 如果 `shuffle=True`，batch 顺序变化不影响 index，因为 index 来自底层 dataset。
- 如果使用 augmentation，train memory 会基于 augmentation 后的 train data；这可能改变 retrieval distribution，论文主实验建议先关闭 augmentation。
- PEMS/Solar 的 `data_stamp` 是零向量，所以 LARA 当前 `time_bonus` 在这些数据上基本无信息。

## 后续应补

为 LARA diagnostics 保存 query index、candidate indices、weights、utility、gate，可以在 data/exp 层增加可选 logging hook。这样才能离线画 retrieval help/harm 和 gate calibration。
