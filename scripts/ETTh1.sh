#!/usr/bin/env bash
# ETTh1 (7 variates, hourly). One-click two-stage run:
# 1) train a DLinear host, 2) freeze it and train the LARA adapter.
extra_args="$@"

seq_len=96
for pred_len in 96; do
  host_model_id=ETTh1_DLinear_host_${seq_len}_${pred_len}
  host_des=DLinear_host_ETTh1_96_ep10
  host_setting=long_term_forecast_${host_model_id}_DLinear_ETTh1_ftM_sl${seq_len}_ll48_pl${pred_len}_dm512_nh8_el2_dl1_df512_fc3_ebtimeF_${host_des}_0
  host_ckpt=./checkpoints/${host_setting}/checkpoint.pth

  if [ -f "$host_ckpt" ]; then
    echo "[LARA] using existing frozen host checkpoint: $host_ckpt"
  else
    python -u run.py \
      --task_name long_term_forecast --is_training 1 \
      --data ETTh1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --freq h \
      --model_id "$host_model_id" --model DLinear \
      --features M --seq_len $seq_len --label_len 48 --pred_len $pred_len \
      --enc_in 7 --dec_in 7 --c_out 7 \
      --e_layers 2 --d_layers 1 --factor 3 --d_model 512 --d_ff 512 \
      --des "$host_des" --itr 1 --batch_size 32 --learning_rate 0.0003 \
      --train_epochs 10 --patience 3 --num_workers 0
  fi

  python -u run.py \
    --task_name long_term_forecast --is_training 1 \
    --data ETTh1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --freq h \
    --model_id ETTh1_LARA_hwise_raw_oracle_gate1_${seq_len}_${pred_len} --model LARA_DLinear \
    --features M --seq_len $seq_len --label_len 48 --pred_len $pred_len \
    --enc_in 7 --dec_in 7 --c_out 7 \
    --e_layers 2 --d_layers 1 --factor 3 --d_model 512 --d_ff 512 \
    --des LARA_hwise_raw_oracle_gate1_ETTh1_96_ep10 --itr 1 --batch_size 32 --learning_rate 0.0003 \
    --train_epochs 10 --patience 3 --num_workers 0 \
    --lara_top_m 50 --lara_lambda_rank 0.3 --lara_lambda_sparse 0.01 \
    --lara_score_mode horizon --lara_sparse_mode sparsemax \
    --lara_lambda_gate 1.0 --lara_gate horizon --lara_gate_target oracle \
    --lara_host_ckpt "$host_ckpt" --lara_freeze_host \
    --lara_oracle_ms 1,3,5,10,20,50 --lara_oracle_topk 1,3,5 \
    $extra_args
done
