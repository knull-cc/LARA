#!/usr/bin/env bash
# One-click entry for the current LARA MVP experiment.
# Default: ETTm2-96 with LARA_DLinear. Any extra args are passed through
# to scripts/ETTm2.sh and then to run.py, e.g.:
#   bash run_main.sh --des LARA_offset_ETTm2_96
#   MODEL=DLinear bash run_main.sh --des DLinear_ETTm2_96
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Selected small multivariate dataset for fast LARA validation.
bash "$DIR"/scripts/ETTm2.sh "$@"

# Other datasets are intentionally disabled for the MVP sweep.
# bash "$DIR"/scripts/ETTh1.sh "$@"
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
