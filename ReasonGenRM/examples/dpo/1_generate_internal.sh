#!/bin/bash

SCRIPT_DIR="$(dirname "$0")"
WORK_DIR="$(realpath "$SCRIPT_DIR/../..")"

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONPATH=$PYTHONPATH:$WORK_DIR

# Dataset and Model Paths
dataset_name_or_path="Skywork/Skywork-Reward-Preference-80K-v0.2"
internal_model_name_or_path="${WORK_DIR}/exp/Llama3.1-8B-ReasonRM/sft"

# Generated Files
base_dir="${WORK_DIR}/data/Skywork-Reward-Preference-80K-v0.2/dpo"
prompt_filename="${base_dir}/../sft/source.jsonl"
internal_generated_reason_dir="${base_dir}/internal_generated_reason"

# vLLM Config
NUM_PROCESSES=8 # NOTE: Modify according to the size of the model
GPUS_PER_PROCESS=1 # NOTE: Modify according to the size of the model
NUM_WORKER_PER_PROCESS=6 # NOTE: Modify according to the size of the model
NUM_RESPONSE_PER_REQUEST=2 # NOTE: Modify according to the size of the model
LOG_DIR="$WORK_DIR/logs"
BASE_PORT=8009
BASE_SEED=42

mkdir -p "$LOG_DIR"

###########################
# Preprepare Prompt
###########################
if [ ! -f "${prompt_filename}" ]; then
  python ${WORK_DIR}/src/sft/prepare_prompt.py \
    --dataset_dir ${dataset_name_or_path} \
    --save_filename ${prompt_filename}
fi

###########################
# Prepare Internal Reason
###########################
declare -a SERVER_URLS

for ((i = 0; i < NUM_PROCESSES; i++)); do
  PORT=$(($BASE_PORT + $i))
  START_GPU=$(($i * $GPUS_PER_PROCESS))
  END_GPU=$(($START_GPU + $GPUS_PER_PROCESS - 1))
  CUDA_VISIBLE_DEVICES=$(seq -s, $START_GPU $END_GPU)
  echo "Starting vLLM server $i on port $PORT using GPUs $CUDA_VISIBLE_DEVICES"

  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
  nohup python -m vllm.entrypoints.api_server \
    --port $PORT \
    --model ${internal_model_name_or_path} \
    --tensor-parallel-size ${GPUS_PER_PROCESS} \
    --gpu-memory-utilization 0.95 \
    --max-model-len 32768 \
    --max-num-batched-tokens 131072 \
    --max-num-seqs $(($NUM_WORKER_PER_PROCESS * $NUM_RESPONSE_PER_REQUEST)) \
    --disable-log-requests \
  > "${LOG_DIR}/vllm_internal_server${i}.log" 2>&1 &

  SERVER_URLS[$i]="http://0.0.0.0:${PORT}/generate"
done

# Wait for vLLM servers to be fully up
echo "Waiting for vLLM servers to start..."
for ((i = 0; i < NUM_PROCESSES; i++)); do
  PORT=$(($BASE_PORT + $i))
  while ! curl -s http://localhost:${PORT}/health > /dev/null; do
    sleep 1  # Check every second
  done
done
echo "vLLM servers have started, proceeding with the next steps..."

# Start scripts
declare -a SCRIPT_PIDS
for (( i=0; i<$NUM_PROCESSES; i++ ))
do
  internal_generated_reason_filename="${internal_generated_reason_dir}/${i}.jsonl"
  echo "Starting generation script for server $i, output to $internal_generated_reason_filename"

  python ${WORK_DIR}/src/dpo/generate_reason_internal.py \
    --dataset_dir "${prompt_filename}" \
    --save_filename "${internal_generated_reason_filename}" \
    --server_url "${SERVER_URLS[$i]}" \
    --model_name_or_path ${internal_model_name_or_path} \
    --max_workers ${NUM_WORKER_PER_PROCESS} \
    --temperature 1.0 \
    --top_p 1.0 \
    --seed $(($BASE_SEED + $i)) \
    --N ${NUM_RESPONSE_PER_REQUEST} &
    
  SCRIPT_PIDS[$i]=$!
done

# Wait for all generate scripts to complete
for pid in "${SCRIPT_PIDS[@]}"; do
  wait $pid
done

# Stop Servers
pkill -f "vllm.entrypoints.api_server"
echo "vllm server has been stopped."
