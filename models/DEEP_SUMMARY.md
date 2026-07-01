# `LARA/models/` 深入总结

## 目录职责

这个目录包含 host forecasters 与 LARA wrappers。TSF-Lib 的 `LazyModelDict` 会自动扫描 `models/*.py`，所以新增模型文件只要提供 `class Model` 就能通过 `--model` 调用。

## 关键文件

- `DLinear.py`：强简单 baseline，按 seasonal/trend decomposition 做线性预测。
- `PatchTST.py`：patch embedding + Transformer encoder 的经典 multivariate baseline。
- `iTransformer.py`：变量维 token 化的强 multivariate baseline。
- `GTR.py`：在本地改造成 TSF-Lib 四参数接口的 Global Temporal Retrieval host。
- `LARA_DLinear.py`：LARA 的主要 MVP wrapper。
- `LARA_GTR.py`：继承 LARA_DLinear 的 adapter 框架，但把 host 替换成 GTR。

## LARA_DLinear 的结构

`LARA_DLinear.Model` 内部包含：

- `self.host`: DLinear host forecaster。
- `self.labeler`: `UtilityLabeler`。
- `self.reranker`: `UtilityReranker(feature_dim=6)`。
- `self.gate`: scalar 或 horizon `FusionGate`。
- `self.memory/self.retriever`: 训练前由 `prepare_memory()` 注入。

forward 逻辑是：

1. host 预测 `y_host`；
2. 若 memory 未准备，直接 fallback 到 host；
3. 检索 candidate futures；
4. offset align；
5. rerank + aggregate；
6. gate fusion；
7. 若有 `y_true`，构造 oracle utility 和 aux loss；
8. 返回 `y_final[:, -pred_len:, :]`。

## LARA_GTR 的意义

`LARA_GTR.py` 说明 LARA 的 wrapper 设计具备一定 host-agnostic 潜力：它先复用 LARA_DLinear 初始化，再替换 `self.host = GTRModel(configs)`。这支持论文中的 plug-and-play 叙事，但目前代码只显式给出 DLinear 和 GTR 两个 host。若要写“任意 backbone”，还需要 PatchTST/iTransformer wrapper 或统一 host factory。

## Host checkpoint 与 freezing

`--lara_host_ckpt` 可以加载 host checkpoint；`--lara_freeze_host` 可以冻结 host。研究上最干净的 Stage-C 实验应该使用 frozen host，否则 oracle utility label 会随着 host 参数变化而漂移，难以解释 reranker 学到的到底是什么。

## 论文风险

当前最能防守的 claim 是“LARA can be attached to DLinear and GTR in this codebase”。若没有 PatchTST/iTransformer 的实测结果，不应写成全面 model-agnostic。更稳的表述是“the adapter interface is host-agnostic, and we validate it on selected hosts”。
