#!/bin/bash

SCRIPT_DIR="$(dirname "$0")"
WORK_DIR="$(realpath "$SCRIPT_DIR/../..")"

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONPATH=$PYTHONPATH:$WORK_DIR

MODEL_PATH="jiulaikankan/Qwen2.5-14B-ReasonGenRM"

python ${WORK_DIR}/src/demo/web_cli.py \
  --model_path ${MODEL_PATH}\
  --cuda_visible_devices 0,1 \
  --port 7860 \
  --max_new_tokens 8192 \
  --temperature 0.5
