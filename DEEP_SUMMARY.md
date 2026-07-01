# LARA 文件夹深入总结

## 这个文件夹是什么

`LARA/` 是当前工作区的核心实现：一个裁剪后的 TSF-Lib 长期时间序列预测框架，外加 LARA MVP adapter。它的目的不是重新发明一个 forecasting backbone，而是在 DLinear/GTR 等 host forecaster 外部增加 loss-aware retrieval adaptation。

本地来源：

- `LARA/README.md`
- `LARA/LARA/README.md`
- `LARA/run.py`
- `LARA/exp/exp_long_term_forecasting.py`
- `LARA/models/LARA_DLinear.py`
- `LARA/LARA/memory.py`
- `LARA/LARA/retrieval.py`
- `LARA/LARA/modules.py`

## 核心机制

LARA 的预测流程是：

1. host forecaster 先输出 `y_host`。
2. `TimeSeriesMemory` 从 train split 构建 memory，保存 pasts、futures、marks、indices 和 normalized keys。
3. `CandidateRetriever` 对 query window 做 cosine retrieval，训练时用 temporal overlap mask 避免检索自身或重叠片段。
4. retrieved candidate futures 先做 offset alignment：使用 `query_last + (candidate_future - candidate_past_last)`，减少绝对尺度偏移。
5. `UtilityReranker` 根据 6 个候选特征输出 candidate score。
6. `aggregate_candidates` 用 softmax 或 sparsemax 聚合 candidate futures 得到 `y_ret`。
7. `FusionGate` 按 scalar 或 horizon gate 混合 `y_host` 与 `y_ret`。
8. 训练时 `UtilityLabeler` 用真实 `y_true` 构造 oracle utility，并把 listwise KL、sparse entropy、可选 pair/score/gate loss 加到 forecast loss 上。

## 当前实现已经有的东西

- train-only memory：`prepare_memory(train_data)` 只从训练集构造，不直接把 validation/test 写入 memory。
- overlap mask：训练模式会按 `seq_len + pred_len + overlap_margin` 屏蔽邻近样本。
- utility label：用 alpha grid 搜索候选 future 与 host prediction 的最优线性插值收益。
- reranker 特征：similarity、time_bonus、rank、score_margin、future_dispersion、past_scale。
- aggregation：softmax 与 sparsemax 两种权重函数。
- gate：支持 scalar 与 horizon gate，但不是完整 channel-aware gate。
- diagnostics：输出 gate、active_k、entropy、host/final/oracle MSE、oracle_gain、positive_utility、helpful_query、rank_corr 等。

## 与设计文档的差距

`idea_optimization/04_LARA_mvp_implementation_spec.md` 中的完整 MVP 目标比当前实现更强。当前代码满足了第一版 DLinear/GTR adapter，但还没覆盖以下点：

- candidate pool 还不是多 view retrieval；主要是 normalized key cosine。
- dynamic K 不是显式 hard K，而是通过 sparsemax/softmax 权重和 active K 统计体现。
- channel-aware 或 channel-group gate 还没实现。
- LARA_PatchTST 和 LARA_iTransformer wrapper 尚未看到。
- help/harm split、utility NDCG、gate calibration 还没有形成独立分析脚本。

## 读代码时的关键入口

- 训练入口：`run.py -> Exp_Long_Term_Forecast.train()`
- LARA memory 注入：`Exp_Long_Term_Forecast._maybe_prepare_lara_memory()`
- LARA forward：`models/LARA_DLinear.py::Model.forward`
- memory 构造：`LARA/memory.py::TimeSeriesMemory.from_dataset`
- retrieval：`LARA/retrieval.py::CandidateRetriever.retrieve`
- utility/gate：`LARA/modules.py`

## 对论文写作的意义

这份代码已经能支撑“loss-aware retrieval utility learning”的最小实验，但还不能自然支撑“channel-aware retrieval trust”这种强 claim。写论文时应该把主贡献收敛到：

- candidate-level final-loss utility supervision；
- query-level sparse reranking；
- host-retrieval gate；
- train-only memory 与 leakage control；
- retrieval diagnostics。

## 最危险的问题

如果实验只显示平均 MSE/MAE 提升，LARA 会显得像 RAFT + SARAF + PFRP gate 的拼装。必须用诊断实验证明：相似度排名高但 oracle utility 低的候选被 reranker 降权，gate 在 harmful/ambiguous query 上降低 retrieval trust。
