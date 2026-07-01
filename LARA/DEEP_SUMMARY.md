# `LARA/LARA/` 深入总结

## 目录职责

这个目录是 LARA adapter 的核心算法层，包含三个文件：

- `memory.py`：构建 train-only memory bank 与 retrieval keys。
- `retrieval.py`：执行 top-M candidate retrieval 并生成候选特征。
- `modules.py`：utility label、reranker、aggregation、gate、loss 和 diagnostics helper。

## `memory.py`

`TimeSeriesMemory` 保存：

- `pasts`: historical input windows `[N, L, C]`
- `futures`: paired future horizons `[N, H, C]`
- `marks`: 每个窗口最后时刻的 time mark
- `indices`: 样本索引，用于 temporal overlap mask
- `keys`: 标准化后的 retrieval keys

key 构造逻辑是先减去窗口最后值，再按时间维标准差归一化，最后 flatten 并做 L2 normalize。`key_mode=auto` 会在 `L*C <= max_key_dim` 时用 flatten，否则退化为 channel mean，以避免高维 key 过大。

关键判断：这是一种保守、可解释的 morphology key，不是 learned encoder。优点是实现简单、不会额外引入训练不稳定；缺点是无法像 PFRP 那样学习“future-similar”的表示，也无法天然区分不同变量组的重要性。

## `retrieval.py`

`CandidateRetriever` 做三件事：

1. 分块计算 query keys 与 memory keys 的 cosine similarity。
2. 训练模式下按 sample index 应用 temporal overlap mask。
3. 取 top-M 候选并返回 candidate futures、candidate pasts 与 6 维候选特征。

候选特征包括：

- `values`: past-window cosine similarity
- `time_bonus`: query/candidate time mark 接近程度
- `rank_feature`: top-M 内部排名
- `score_margin`: top-1 相似度与当前候选相似度的差
- `future_dispersion`: 候选 future 相对候选均值的离散度
- `past_scale`: 候选 past 的尺度

关键判断：LARA 当前不是完全抛弃 similarity，而是把 similarity 降级为 reranker 输入特征。真正的 novelty 必须由 utility label 和 reranker 体现。

## `modules.py`

`UtilityLabeler` 是 LARA 的核心：对每个候选 future，搜索 `alpha in [0, 1]`，计算 `(1-alpha)*y_host + alpha*y_cand` 的最小 MSE，并定义：

`delta = MSE(y_host, y_true) - min_alpha MSE(mixed, y_true)`

代码中对 `delta` 做了 `clamp_min(0.0)`，所以负收益候选被压成 0。这对 listwise target 稳定有好处，但会丢失“有害程度”的强弱信息。后续若要更强地学习 harmful retrieval，可以考虑保留 signed utility 或单独建 harmful classification label。

其他模块：

- `UtilityReranker`：6 维特征到 scalar score 的 MLP。
- `sparsemax`：可让部分候选权重精确为 0。
- `aggregate_candidates`：score -> weights -> retrieval forecast。
- `FusionGate`：从 score/weight/scale stats 预测 scalar 或 horizon gate。
- `listwise_kl_loss`：让 score distribution 对齐 utility distribution。
- `rank_correlation`：诊断 utility 与 score 的排序一致性。

## 当前强点

- utility label 明确连接 final forecast loss，而不是只用 candidate future similarity。
- overlap mask 是训练期防 leakage 的必要设计。
- offset alignment 让 retrieved future 更像 query-relative continuation。
- diagnostics 比普通 MVP 更完整，方便快速判断 adapter 是否真学到了 retrieval usefulness。

## 当前薄弱点

- reranker 只看 6 个手工特征，没有 query summary、host uncertainty 或 channel-group 信息。
- candidate utility 是单候选插值收益，不是候选集合收益；这可能让多个互补候选的价值被低估。
- clamp 到非负 utility 后，模型只能区分 useful vs useless，不能区分 mildly useless 与 actively harmful。
- `time_bonus` 使用 mark 均值差的指数形式，和 SARAF 的 hour/day/week/month 显式时间对齐相比更弱。

## 读懂 LARA 的关键问题

1. `rank_corr` 是否显著大于 0？
2. `positive_utility` 是否足够高，还是大多数候选都没有收益？
3. `gate` 是否随着 `weighted_alpha` 或 oracle gain 变化？
4. sparsemax 下 `active_k` 是否小于 top-M，并且没有塌缩到总是 1 或总是 top-M？
5. offset alignment 是否在非平稳或不同尺度数据集上稳定提升？
