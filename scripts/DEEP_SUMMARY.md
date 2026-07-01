# `LARA/scripts/` 深入总结

## 目录职责

这个目录放每个数据集的运行脚本。它们把 dataset path、channel 数、horizon、模型名和 LARA 超参传给 `run.py`。

## MVP 入口

`LARA_DLinear_ETTh1_96.sh` 是最直接的 smoke run：

- dataset: ETTh1
- seq_len: 96
- pred_len: 96
- model: `LARA_DLinear`
- batch_size: 32
- train_epochs: 10
- LARA top-M: 16
- rank loss: 0.3
- sparse entropy loss: 0.01
- sparse mode: softmax
- gate: scalar

`run_main.sh` 当前只调用 `scripts/ETTh1.sh`，其他数据集被注释掉。这说明仓库处于 MVP 验证阶段，不是完整论文 sweep 阶段。

## 对实验的启发

第一阶段不要直接跑全部数据集。更合理的顺序是：

1. ETTh1/96 验证训练是否稳定。
2. 同设置跑 DLinear baseline。
3. 加 `--lara_freeze_host --lara_host_ckpt <path>` 做 cleaner Stage-C。
4. 对 softmax/sparsemax、scalar/horizon gate、offset alignment 做小规模 ablation。
5. 再扩到 ETTh2、ETTm1、Electricity、Traffic、Weather。

## 需要注意

脚本里默认 `--num_workers 0` 是为了减少 dataloader 问题，适合调试。正式实验可根据机器情况提高。所有 LARA 实验必须保留 stdout 或保存 diagnostics，否则只能看到最终 MSE/MAE，无法证明 loss-aware retrieval 真的生效。
