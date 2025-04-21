#!/bin/bash

SCRIPT_DIR="$(dirname "$0")"
WORK_DIR="$(realpath "$SCRIPT_DIR/../..")"

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONPATH=$PYTHONPATH:$WORK_DIR

# Dataset and Model Paths
DATASET_DIR="Skywork/Skywork-Reward-Preference-80K-v0.2"
MODEL_PATH="${WORK_DIR}/exp/ReasonGenRM-QwQ-32B/zero"

# Generated Files
BASE_DIR="${WORK_DIR}/data/Skywork-Reward-Preference-80K-v0.2/sft"
PROMPT_FILE="${BASE_DIR}/dataset.jsonl"
GENERATED_REASON_DIR="${BASE_DIR}/generated_reason"

# vLLM Config
NUM_PROCESSES=4 # NOTE: Modify according to the model size
TENSOR_PARALLEL_SIZE=2 # NOTE: Modify according to the model size
NUM_RESPONSE_PER_PROMPT=16 # NOTE: Modify according to the model size
BASE_SEED=42

###########################
# Preprepare Prompt
###########################
if [ ! -f "${PROMPT_FILE}" ]; then
  python ${WORK_DIR}/src/sft/prepare_prompt.py \
    --dataset_dir ${DATASET_DIR} \
    --save_filename ${PROMPT_FILE} || {
    echo "Error: Failed to prepare prompt dataset"
    exit 1
  }
else
    echo "Prompt file already exists: $PROMPT_FILE"
fi

###########################
# Prepare Reason
###########################
declare -a SCRIPT_PIDS

for (( i=0; i<NUM_PROCESSES; i++ ))
do
  generated_file="${GENERATED_REASON_DIR}/${i}.jsonl"
  echo "Starting generation for process $i (output: $generated_file)"

  start_gpu=$((i * TENSOR_PARALLEL_SIZE))
  end_gpu=$((start_gpu + TENSOR_PARALLEL_SIZE - 1))
  cuda_devices=$(seq -s, "$start_gpu" "$end_gpu")

  n=$((NUM_RESPONSE_PER_PROMPT / NUM_PROCESSES))
  remainder=$((NUM_RESPONSE_PER_PROMPT % NUM_PROCESSES))
  if ((i < remainder)); then
    n=$((n + 1))
  fi

  CUDA_VISIBLE_DEVICES="$cuda_devices" \
  python ${WORK_DIR}/src/sft/generate_reason.py \
    --dataset_dir "${PROMPT_FILE}" \
    --save_filename "${generated_file}" \
    --model_name_or_path "${MODEL_PATH}" \
    --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
    --temperature 1.0 \
    --top_p 1.0 \
    --max_tokens 64000 \
    --seed $(($BASE_SEED + $i)) \
    --N "$n" &
    
  SCRIPT_PIDS[$i]=$!
done

# Wait for all generate scripts to complete
for pid in "${SCRIPT_PIDS[@]}"; do
    wait "$pid" || {
        echo "Process $pid failed"
        exit 1
    }
done

echo "All processes completed successfully"
