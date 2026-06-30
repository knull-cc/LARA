#!/usr/bin/env bash
set -euo pipefail

python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --data ETTh1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --freq h \
  --model_id ETTh1_LARA_DLinear_96_96 --model LARA_DLinear \
  --features M --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --e_layers 2 --d_layers 1 --factor 3 --d_model 512 --d_ff 512 \
  --des LARA_MVP --itr 1 --batch_size 32 --learning_rate 0.0003 \
  --train_epochs 3 --patience 2 --num_workers 0 \
  --lara_top_m 16 --lara_lambda_rank 0.3 --lara_lambda_sparse 0.01 \
  --lara_sparse_mode softmax --lara_gate scalar "$@"
