#!/bin/bash

SCRIPT_DIR="$(dirname "$0")"
WORK_DIR="$(realpath "$SCRIPT_DIR/../..")"

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONPATH=$PYTHONPATH:$WORK_DIR

MODEL_NAME_OR_PATH="${WORK_DIR}/exp/Llama3.1-8B-ReasonRM/sft"

DATASET_DIR=()
while IFS= read -r file; do
    DATASET_DIR+=("$file")
done < <(find "${WORK_DIR}/data/Skywork-Reward-Preference-80K-v0.2/dpo/internal_generated_reason" -type f -name "*.jsonl")

OUTPUT_DIR="${WORK_DIR}/data/Skywork-Reward-Preference-80K-v0.2/dpo/Llama3.1-8B-16sample"
NUM_PROCESSES=8 # NOTE: Modify according to the size of the model
GPUS_PER_PROCESS=1 # NOTE: Modify according to the size of the model
THRESH=0.1

# Create necessary directories
mkdir -p "${OUTPUT_DIR}/tmp_reward" "${OUTPUT_DIR}/logs"

# Function to start a process
start_process() {
    local start_gpu=$(($1 * $GPUS_PER_PROCESS))
    local end_gpu=$(($start_gpu + $GPUS_PER_PROCESS - 1))
    local devices=$(seq -s, $start_gpu $end_gpu)
    
    local worker_index=$2
    local save_filename="${OUTPUT_DIR}/tmp_reward/reward_${worker_index}.jsonl"
    local log_filename="${OUTPUT_DIR}/logs/process_${worker_index}.log"

    echo "Starting process ${worker_index} on GPU ${devices}..."
    CUDA_VISIBLE_DEVICES=${devices} python "${WORK_DIR}/src/dpo/reward_dpo_dataset.py" \
        --model_name_or_path "${MODEL_NAME_OR_PATH}" \
        --dataset_dir "${DATASET_DIR[@]}" \
        --save_filename "${save_filename}" \
        --thresh "${THRESH}" \
        --num_workers "${NUM_PROCESSES}" --worker_index "${worker_index}" \
        > "${log_filename}" 2>&1 &
}

# Start parallel processes
for i in $(seq 0 $((NUM_PROCESSES - 1))); do
    start_process $i $i
done

# Wait for all processes to finish
wait
echo "All processes have completed."

# Combine the output files into a single file
COMBINED_REWARD_FILENAME="${OUTPUT_DIR}/reward.jsonl"
cat "${OUTPUT_DIR}/tmp_reward/reward_"*.jsonl > "${COMBINED_REWARD_FILENAME}" 2>/dev/null

if [[ $? -eq 0 ]]; then
    echo "Combined all reward files into ${COMBINED_REWARD_FILENAME}"
else
    echo "Error combining reward files."
    exit 1
fi
