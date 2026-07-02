#!/usr/bin/env bash
# ETTh1 (7 variates, hourly). One-click two-stage LARA_DLinear run.
extra_args="$@"

seq_len=96
for pred_len in 96; do
  host_model_id=ETTh1_DLinear_host_${seq_len}_${pred_len}
  host_des=DLinear_host_ETTh1_96_ep10
  host_setting=long_term_forecast_${host_model_id}_DLinear_ETTh1_ftM_sl${seq_len}_ll48_pl${pred_len}_dm512_nh8_el2_dl1_df512_fc3_ebtimeF_${host_des}_0
  host_ckpt=./checkpoints/${host_setting}/checkpoint.pth

  if [ ! -f "$host_ckpt" ]; then
    python -u run.py \
      --task_name long_term_forecast --is_training 1 \
      --data ETTh1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --freq h \
      --model_id $host_model_id --model DLinear \
      --features M --seq_len $seq_len --label_len 48 --pred_len $pred_len \
      --enc_in 7 --dec_in 7 --c_out 7 \
      --e_layers 2 --d_layers 1 --factor 3 --d_model 512 --d_ff 512 \
      --des $host_des --itr 1 --batch_size 32 --learning_rate 0.0003 \
      --train_epochs 10 --patience 3 --num_workers 0
  fi

  if [ ! -f "$host_ckpt" ]; then
    echo "Missing host checkpoint: $host_ckpt"
    exit 1
  fi

  python -u run.py \
    --task_name long_term_forecast --is_training 1 \
    --data ETTh1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --freq h \
    --model_id ETTh1_LARA_resamp_unionK500_distill5_${seq_len}_${pred_len} --model LARA_DLinear \
    --features M --seq_len $seq_len --label_len 48 --pred_len $pred_len \
    --enc_in 7 --dec_in 7 --c_out 7 \
    --e_layers 2 --d_layers 1 --factor 3 --d_model 512 --d_ff 512 \
    --des LARA_resamp_unionK500_distill5_ETTh1_96_ep15 --itr 1 --batch_size 32 --learning_rate 0.0003 \
    --train_epochs 15 --patience 5 --num_workers 0 \
    --lara_phase_top_k 500 --lara_phase_rerank_mode add --lara_phase_weight 0.08 \
    --lara_pibr_period 24 --lara_pibr_weight 0.15 --lara_pibr_delta_weight 0.5 \
    --lara_retrieval_mode union --lara_extra_m 50 \
    --lara_top_m 50 --lara_temperature 0.03 --lara_rank_temperature 0.03 \
    --lara_lambda_rank 1.0 --lara_lambda_pair 0.5 --lara_pair_margin 0.5 \
    --lara_lambda_score 0.2 --lara_score_loss corr --lara_lambda_sparse 0.001 \
    --lara_sparse_mode sparsemax --lara_score_mode horizon \
    --lara_fusion residual_amp --lara_max_amp 2.0 --lara_alpha_max 2.0 --lara_alpha_step 0.25 \
    --lara_distill_topk 5 --lara_distill_mode horizon --lara_distill_temperature 0.03 \
    --lara_lambda_teacher 2.0 --lara_lambda_weight 1.0 --lara_teacher_gate \
    --lara_lambda_amp 1.0 --lara_lambda_risk 0.2 --lara_risk_margin 0.0 \
    --lara_host_ckpt "$host_ckpt" --lara_freeze_host \
    --lara_oracle_ms 1,3,5,10,20,50 --lara_oracle_topk 1,3,5 \
    $extra_args
done
