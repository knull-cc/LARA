#!/usr/bin/env bash
# ETTh1 (7 variates, hourly). Default host is DLinear through LARA_DLinear.
model_name=${MODEL:-LARA_DLinear}
extra_args="$@"

seq_len=96
for pred_len in 96; do
  python -u run.py \
    --task_name long_term_forecast --is_training 1 \
    --data ETTh1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --freq h \
    --model_id ETTh1_LARA_sparsemax_gate5_${seq_len}_${pred_len} --model "$model_name" \
    --features M --seq_len $seq_len --label_len 48 --pred_len $pred_len \
    --enc_in 7 --dec_in 7 --c_out 7 \
    --e_layers 2 --d_layers 1 --factor 3 --d_model 512 --d_ff 512 \
    --des LARA_sparsemax_gate5_ETTh1_96_ep10 --itr 1 --batch_size 32 --learning_rate 0.0003 \
    --train_epochs 10 --patience 3 --num_workers 0 \
    --lara_top_m 50 --lara_lambda_rank 0.3 --lara_lambda_sparse 0.01 \
    --lara_sparse_mode sparsemax --lara_lambda_gate 5.0 --lara_gate scalar \
    --lara_oracle_ms 1,3,5,10,20,50 --lara_oracle_topk 1,3,5 \
    $extra_args
done
