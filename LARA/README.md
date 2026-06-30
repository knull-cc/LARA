# LARA MVP in TSF-Lib

This folder contains a minimal Loss-Aware Retrieval Adaptation (LARA) adapter for
quick validation on top of `models/DLinear.py`.

## What is implemented

- Train-only memory bank built from the training split.
- Top-M cosine retrieval with temporal-overlap masking during training.
- Candidate features: past similarity, time-mark bonus, rank, score margin,
  candidate future dispersion, and candidate past scale.
- Forecast-loss oracle utility labels:
  `MSE(y_host, y_true) - min_alpha MSE((1-alpha)y_host + alpha y_cand, y_true)`.
- Utility reranker, softmax/sparsemax aggregation, and scalar/horizon fusion gate.

## Quick smoke run

From `TSF-Lib/`:

```bash
bash scripts/LARA_DLinear_ETTh1_96.sh
```

Run the matched DLinear baseline with:

```bash
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --data ETTh1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --freq h \
  --model_id ETTh1_DLinear_96_96 --model DLinear \
  --features M --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --e_layers 2 --d_layers 1 --factor 3 --d_model 512 --d_ff 512 \
  --des Exp --itr 1 --batch_size 32 --learning_rate 0.0001 --train_epochs 3 --num_workers 0
```

## Closer Stage-C run

Train DLinear first, then pass its checkpoint:

```bash
python -u run.py ... --model LARA_DLinear \
  --lara_host_ckpt ./checkpoints/<DLinear-setting>/checkpoint.pth \
  --lara_freeze_host
```

The Stage-C variant is the cleaner test of whether loss-aware retrieval helps
over a fixed host.
