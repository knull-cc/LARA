#!/usr/bin/env bash
# ETTm2 (7 variates, 15-min). Usage: bash scripts/ETTm2.sh [--model X ...]
model_name=${MODEL:-LARA_DLinear}
extra_args="$@"

seq_len=96
for pred_len in 96; do
  python -u run.py \
    --task_name long_term_forecast --is_training 1 \
    --data ETTm2 --root_path ./dataset/ETT-small/ --data_path ETTm2.csv --freq t \
    --model_id ETTm2_LARA_${seq_len}_${pred_len} --model "$model_name" \
    --features M --seq_len $seq_len --label_len 48 --pred_len $pred_len \
    --enc_in 7 --dec_in 7 --c_out 7 \
    --e_layers 2 --d_layers 1 --factor 3 --d_model 512 --d_ff 512 \
    --des LARA_ETTm2_96 --itr 1 --batch_size 32 --learning_rate 0.0003 \
    --train_epochs 10 --patience 3 --num_workers 0 \
    --lara_top_m 16 --lara_lambda_rank 0.3 --lara_lambda_sparse 0.01 \
    --lara_sparse_mode softmax --lara_gate scalar \
    $extra_args
done
