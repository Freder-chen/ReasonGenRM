#!/bin/bash

SCRIPT_DIR="$(dirname "$0")"
WORK_DIR="$(realpath "$SCRIPT_DIR/../..")"

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONPATH=$PYTHONPATH:$WORK_DIR

# Dataset and Model Paths
dataset_name_or_path="Skywork/Skywork-Reward-Preference-80K-v0.2"
external_model_name_or_path="meta-llama/Llama-3.1-70B-Instruct"

# Generated Files
base_dir="${WORK_DIR}/data/Skywork-Reward-Preference-80K-v0.2/sft"
prompt_filename="${base_dir}/source.jsonl"
external_generated_reason_dir="${base_dir}/external_generated_reason"

# vLLM Config
NUM_PROCESSES=2 # NOTE: Modify according to the size of the model
GPUS_PER_PROCESS=4 # NOTE: Modify according to the size of the model
VLLM_MODEL_NAME="reason_model"
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
# Prepare External Reason
###########################
declare -a SERVER_URLS

for ((i = 0; i < NUM_PROCESSES; i++)); do
  PORT=$(($BASE_PORT + $i))
  START_GPU=$(($i * $GPUS_PER_PROCESS))
  END_GPU=$(($START_GPU + $GPUS_PER_PROCESS - 1))
  CUDA_VISIBLE_DEVICES=$(seq -s, $START_GPU $END_GPU)
  echo "Starting vLLM server $i on port $PORT using GPUs $CUDA_VISIBLE_DEVICES"

  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
  nohup python -m vllm.entrypoints.openai.api_server \
    --port $PORT \
    --served-model-name ${VLLM_MODEL_NAME} \
    --model ${external_model_name_or_path} \
    --tensor-parallel-size ${GPUS_PER_PROCESS} \
    --gpu-memory-utilization 0.95 \
    --max-model-len 32768 \
    --max-num-batched-tokens 131072 \
    --max-num-seqs 32 \
    --disable-log-requests \
  > "${LOG_DIR}/vllm_external_server${i}.log" 2>&1 &

  SERVER_URLS[$i]="http://0.0.0.0:${PORT}/v1/chat/completions"
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
  external_generated_reason_filename="${external_generated_reason_dir}/${i}.jsonl"
  echo "Starting generation script for server $i, output to $external_generated_reason_filename"

  python ${WORK_DIR}/src/sft/generate_reason_external.py \
    --dataset_dir "${prompt_filename}" \
    --save_filename "${external_generated_reason_filename}" \
    --server_url "${SERVER_URLS[$i]}" \
    --model_name "${VLLM_MODEL_NAME}" \
    --max_workers 6 \
    --temperature 1.0 \
    --top_p 1.0 \
    --seed $(($BASE_SEED + $i)) \
    --N 2 &
    
  SCRIPT_PIDS[$i]=$!
done

# Wait for all generate scripts to complete
for pid in "${SCRIPT_PIDS[@]}"; do
  wait $pid
done

# Stop Servers
pkill -f "vllm.entrypoints.openai.api_server"
echo "vllm server has been stopped."
