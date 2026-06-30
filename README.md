# TSF-Lib

一个**简洁、通用、可扩展**的长序列时间序列预测(long-term forecasting)框架。

设计目标:严格遵循清华 [Time-Series-Library](https://github.com/thuml/Time-Series-Library) (TSLib) 的模型与配置约定,
使其模型可以**直接拖进 `models/` 即用**;同时裁剪为单任务、扁平脚本的精简版,方便快速落地新 idea。

## 特性

- 一键运行:`bash run_main.sh` 串行跑通常见七大类 + Solar + PEMS。
- 模型自动注册:把任意 `models/X.py`(内含 `class Model`)放进去,即可 `--model X` 运行,**无需改注册表**(见 `exp/exp_basic.py` 的 `LazyModelDict`)。
- 通用参数先行:`run.py` 提供所有模型共享的通用超参,模型只从 `configs` 读自己需要的。
- 内置 baseline:`DLinear`、`iTransformer`、`PatchTST`(均原样来自 TSLib)。

## 目录结构

```
TSF-Lib/
  run.py                          # 统一入口(单一共享 argparse)
  run_main.sh                     # 串行跑所有数据集脚本,透传额外参数
  exp/
    exp_basic.py                  # 设备 + models/ 目录自动扫描(懒加载)
    exp_long_term_forecasting.py  # train / vali / test + 指标
  models/                         # DLinear / iTransformer / PatchTST(放新模型即生效)
  layers/                         # 模型依赖的子模块
  data_provider/                  # data_factory + data_loader(ETT/custom/PEMS/Solar)
  utils/                          # tools / metrics / timefeatures / augmentation / ...
  scripts/                        # 每个数据集一个脚本
```

## 安装

```bash
pip install -r requirements.txt
```

## 数据放置约定

数据不随框架附带,按如下结构放在 `./dataset/`(与脚本里的 `--root_path` 一致):

```
dataset/
  ETT-small/   ETTh1.csv  ETTh2.csv  ETTm1.csv  ETTm2.csv
  electricity/ electricity.csv
  traffic/     traffic.csv
  weather/     weather.csv
  Solar/       solar_AL.txt
  PEMS/        PEMS03.npz  PEMS04.npz  PEMS07.npz  PEMS08.npz
```

数据集来源同 TSLib(可从其 README 提供的链接下载)。

## 使用

跑单个数据集(默认 `DLinear`,在 96/192/336/720 上循环):

```bash
bash scripts/ETTh1.sh
```

换模型 / 改超参(额外参数会透传给 `run.py`):

```bash
bash scripts/ETTh1.sh --model iTransformer
bash scripts/ETTh1.sh --model PatchTST --train_epochs 5 --learning_rate 0.0005
MODEL=iTransformer bash scripts/ETTh1.sh
```

一键跑全部数据集:

```bash
bash run_main.sh                 # 全部数据集 + 默认 DLinear
bash run_main.sh --model iTransformer
```

直接调用:

```bash
python run.py \
  --task_name long_term_forecast --is_training 1 \
  --data ETTh1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --model_id ETTh1_96_96 --model iTransformer \
  --features M --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7
```

支持的 `--data`:`ETTh1 / ETTh2 / ETTm1 / ETTm2 / custom(electricity, traffic, weather) / Solar / PEMS`。

## 如何加一个新 idea(新模型)

1. 在 `models/` 新建 `MyIdea.py`,实现统一接口(可直接抄 `models/DLinear.py` 当模板):

```python
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        # ... 从 configs 读你需要的超参(seq_len / enc_in / d_model ...)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: [B, seq_len, enc_in]
        # 返回 [B, pred_len, enc_in]
        ...
```

2. 直接运行,无需任何注册:

```bash
python run.py --model MyIdea --data ETTh1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --seq_len 96 --pred_len 96 --enc_in 7
# 或
bash scripts/ETTh1.sh --model MyIdea
```

3. 需要专属超参时,在 `run.py` 的 `build_parser()` 里加一行 `parser.add_argument(...)`,模型里用 `configs.your_arg` 读取即可。

> 提示:TSLib 的任意模型文件(如 `TimesNet.py`、`TimeMixer.py`)也可以直接拷进 `models/`;
> 若它依赖额外的 `layers/` 子模块或额外超参,把对应文件一起拷来、并在 `run.py` 补上缺失参数即可。

## 结果

- 指标追加写入 `result_long_term_forecast.txt`;
- 每次实验的 `pred.npy / true.npy / metrics.npy` 存到 `results/<setting>/`;
- checkpoint 存到 `checkpoints/<setting>/`。
