#!/bin/bash
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SRC_PATH="${SCRIPT_DIR}/../.."
MODEL_PATH="${SRC_PATH}/saved_models"

export PYTORCH_ENABLE_MPS_FALLBACK=1
# copy the model and training code if new
if [ -d "${MODEL_PATH}" ]
then
    echo "Directory already exists, using existing model."
else
    echo "Saved models not found!"
fi

python plot_training_stats.py    \
    --model_path "${MODEL_PATH}" \
    --script_path "${SCRIPT_DIR}" 
