# `LARA/layers/` 深入总结

## 目录职责

这个目录主要是从 TSF-Lib/Autoformer/PatchTST 继承来的通用神经网络层，不是 LARA 的核心创新点。

关键文件：

- `Autoformer_EncDec.py`
- `Transformer_EncDec.py`
- `SelfAttention_Family.py`
- `Embed.py`

## 在 LARA 中的角色

- `DLinear.py` 依赖 `series_decomp` 做 trend/seasonal decomposition。
- `PatchTST.py` 依赖 patch embedding、Transformer encoder、attention layers。
- LARA adapter 本身不直接在这里实现；它在 `LARA/LARA/` 和 `models/LARA_*.py`。

## 阅读建议

如果目标是理解 LARA，先不用深读这个目录。只有当你要比较 host backbone 差异，或解释 DLinear/PatchTST/iTransformer 作为 host 的 inductive bias 时，再回来看这里。

## 论文写作边界

不要把这些 inherited layers 写成 LARA 的贡献。它们只是 host model 和 baseline 的依赖。LARA 的贡献应聚焦 retrieval utility、candidate reranking、aggregation 和 fusion。
