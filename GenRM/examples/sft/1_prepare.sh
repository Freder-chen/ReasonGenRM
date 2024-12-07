#!/bin/bash

SCRIPT_DIR=$(dirname "$0")
WORK_DIR=$(realpath "$SCRIPT_DIR/../..")

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1

DATASET_DIR="Skywork/Skywork-Reward-Preference-80K-v0.2"
DATA_DIR="${WORK_DIR}/data/Skywork-Reward-Preference-80K-v0.2"
BASE_FILE="${DATA_DIR}/sft.jsonl"

echo "Preparing reasoning dataset..."
python ${WORK_DIR}/src/prepare_sft_dataset.py \
    --dataset_dir "${DATASET_DIR}" \
    --save_filename "${BASE_FILE}"
