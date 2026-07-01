#!/usr/bin/env bash
# One-click entry for the current LARA experiment.
# Default: ETTh1-96 two-stage run: train DLinear host, then freeze it and train LARA.
# Run directly with:
#   bash run_main.sh
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Selected small multivariate dataset for fast LARA validation.
bash "$DIR"/scripts/ETTh1.sh "$@"

# Other datasets are intentionally disabled for the MVP sweep.
# bash "$DIR"/scripts/ETTm2.sh "$@"
# bash "$DIR"/scripts/ETTh2.sh "$@"
# bash "$DIR"/scripts/ETTm1.sh "$@"
# bash "$DIR"/scripts/Electricity.sh "$@"
# bash "$DIR"/scripts/Traffic.sh "$@"
# bash "$DIR"/scripts/Weather.sh "$@"
# bash "$DIR"/scripts/Solar.sh "$@"
# bash "$DIR"/scripts/PEMS03.sh "$@"
# bash "$DIR"/scripts/PEMS04.sh "$@"
# bash "$DIR"/scripts/PEMS07.sh "$@"
# bash "$DIR"/scripts/PEMS08.sh "$@"
