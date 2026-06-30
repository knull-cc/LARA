#!/usr/bin/env bash
# One-click entry for the current LARA MVP experiment.
# Default: Weather-96 with LARA_DLinear. Any extra args are passed through
# to scripts/Weather.sh and then to run.py, e.g.:
#   bash run_main.sh --des offset_align
#   MODEL=DLinear bash run_main.sh --des dlinear_baseline
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Selected small multivariate dataset for fast LARA validation.
bash "$DIR"/scripts/Weather.sh "$@"

# Other datasets are intentionally disabled for the MVP sweep.
# bash "$DIR"/scripts/ETTh1.sh "$@"
# bash "$DIR"/scripts/ETTh2.sh "$@"
# bash "$DIR"/scripts/ETTm1.sh "$@"
# bash "$DIR"/scripts/ETTm2.sh "$@"
# bash "$DIR"/scripts/Electricity.sh "$@"
# bash "$DIR"/scripts/Traffic.sh "$@"
# bash "$DIR"/scripts/Solar.sh "$@"
# bash "$DIR"/scripts/PEMS03.sh "$@"
# bash "$DIR"/scripts/PEMS04.sh "$@"
# bash "$DIR"/scripts/PEMS07.sh "$@"
# bash "$DIR"/scripts/PEMS08.sh "$@"
