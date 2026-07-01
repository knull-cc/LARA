# `LARA/exp/` 深入总结

## 目录职责

`exp/` 是训练、验证、测试循环。LARA 相关改动主要在 `exp_long_term_forecasting.py`，其职责是让普通 TSF-Lib 模型与 LARA 模型共用同一训练入口。

## LARA 接入点

关键 helper：

- `_model_ref()`：兼容 DataParallel，拿到真实模型对象。
- `_is_lara_model()`：检查 `supports_lara_context`。
- `_maybe_prepare_lara_memory(train_data)`：训练前为 LARA 构建 train-only memory。
- `_unpack_batch(batch)`：支持普通 4 元 batch 和带 sample index 的 5 元 batch。
- `_forward_model(...)`：若是 LARA，则额外传入 `sample_index`、`mode`、`y_true`。
- `_add_lara_aux_loss(loss)`：把模型内部 `aux_loss` 加到 forecast loss。
- `_print_lara_diagnostics()` 与 `_print_lara_average_diagnostics()`：打印 LARA 诊断指标。

## 训练流程

`train(setting)` 中先获取 train/val/test dataloader，然后调用 `_maybe_prepare_lara_memory(train_data)`。这一步非常重要：LARA memory 只来自 train split。随后每个 batch 正常构造 decoder input，LARA forward 会接收 `target_for_model = batch_y[:, -pred_len:, :]` 用于 utility label。

训练 loss 是：

`MSE(outputs, batch_y) + model.aux_loss`

其中 `aux_loss` 来自 rank loss、pair loss、score loss、sparse loss、gate loss 等。

## 验证与测试

validation/test 也会把 `y_true` 传给 LARA，但模型在 `eval/no_grad` 下只用它记录 diagnostics，不反向更新。测试前如果发现 memory 还没准备，会重新从 train split 构建 memory。

## 防 leakage 设计

leakage 防线有两层：

1. memory 只从 train split 构建。
2. train mode 检索时用 sample index 做 temporal overlap mask。

这两层都必须在论文实验协议里写清楚。否则 retrieval-augmented forecasting 很容易被质疑把目标 future 间接塞进 memory。

## 当前不足

- diagnostics 只打印到 stdout，没有稳定写入 CSV/JSON，后续做 help/harm split 和 gate calibration 不方便。
- validation/test 传入 `y_true` 记录 oracle 指标是合理的分析行为，但论文中必须明确这些值只用于 evaluation diagnostics，推理输出不依赖未来标签。
- early stopping 仍以普通 validation loss 为准，没有单独考虑 retrieval rank quality。
