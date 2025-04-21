#!/bin/bash

SCRIPT_DIR="$(dirname "$0")"
WORK_DIR="$(realpath "$SCRIPT_DIR/../..")"

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

MODEL_NAME_OR_PATH="Qwen/QwQ-32B"

# Greedy
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python ${WORK_DIR}/src/eval/rewardbench.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --record_filename "${WORK_DIR}/rewardbench_eval.txt" \
  --tensor_parallel_size 8 \
  --num_repeats 1 \
  --temperature 0.0 \
  --top_p 0.95 \
  --max_tokens 64000

# Major Vote
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python ${WORK_DIR}/src/eval/rewardbench.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --record_filename "${WORK_DIR}/rewardbench_eval.txt" \
  --tensor_parallel_size 8 \
  --num_repeats 16 \
  --temperature 0.6 \
  --top_p 0.95 \
  --max_tokens 64000
