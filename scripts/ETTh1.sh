#!/usr/bin/env bash
# ETTh1 (7 variates, hourly). One-click residual-amplified LARA_DLinear run.
extra_args="$@"

seq_len=96
for pred_len in 96; do
  python -u run.py \
    --task_name long_term_forecast --is_training 1 \
    --data ETTh1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --freq h \
    --model_id ETTh1_LARA_resamp_unionK300_${seq_len}_${pred_len} --model LARA_DLinear \
    --features M --seq_len $seq_len --label_len 48 --pred_len $pred_len \
    --enc_in 7 --dec_in 7 --c_out 7 \
    --e_layers 2 --d_layers 1 --factor 3 --d_model 512 --d_ff 512 \
    --des LARA_resamp_unionK300_ETTh1_96_ep10 --itr 1 --batch_size 32 --learning_rate 0.0003 \
    --train_epochs 10 --patience 3 --num_workers 0 \
    --lara_phase_top_k 300 --lara_phase_rerank_mode add --lara_phase_weight 0.08 \
    --lara_pibr_period 24 --lara_pibr_weight 0.12 --lara_pibr_delta_weight 0.5 \
    --lara_retrieval_mode union --lara_extra_m 20 \
    --lara_top_m 50 --lara_lambda_rank 0.4 --lara_lambda_sparse 0.005 \
    --lara_sparse_mode sparsemax --lara_score_mode horizon \
    --lara_fusion residual_amp --lara_max_amp 2.0 --lara_alpha_max 2.0 --lara_alpha_step 0.25 \
    --lara_lambda_amp 1.0 --lara_lambda_risk 1.0 --lara_risk_margin 0.0 \
    --lara_oracle_ms 1,3,5,10,20,50 --lara_oracle_topk 1,3,5 \
    $extra_args
done
