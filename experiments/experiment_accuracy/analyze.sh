#!/bin/bash

NAME=$1

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SRC_PATH="${SCRIPT_DIR}/../../.."
MODEL_PATH="${SRC_PATH}/saved_models/${NAME}_model"

python table.py \
    --id "${NAME}_experiment" \
    --save_latex True \

python supplemental_plots.py \
    --id "${NAME}_experiment" \
    --filter_outliers True \
