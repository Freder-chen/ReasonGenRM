#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1

#####################################
# Paths and variables
#####################################
SCRIPT_DIR=$(dirname "$0")
WORK_DIR=$(realpath "$SCRIPT_DIR")

MODEL_NAME_OR_PATH="Qwen/Qwen2.5-14B-Instruct"

DATASET_DIR="Skywork/Skywork-Reward-Preference-80K-v0.2"

DATA_DIR="${WORK_DIR}/data/Skywork-Reward-Preference-80K-v0.2"
BASE_FILE="${DATA_DIR}/base.jsonl"
TAG_FILE="${DATA_DIR}/taged.jsonl"

SERVER_URL="http://0.0.0.0:8009/v1/chat/completions"
MODEL_NAME="qwen2_instruct"

#####################################
# Start the vllm server in background
#####################################
echo "Starting vllm server..."
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup \
python -m vllm.entrypoints.openai.api_server \
    --port 8009 \
    --served-model-name ${MODEL_NAME} \
    --model ${MODEL_PATH} \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 32768 \
    --max-num-batched-tokens 131072 \
    --max-num-seqs 16 \
    --disable-log-requests \
> vllm_server.log 2>&1 &

# Wait until the server is up
echo "Waiting for vllm server to start..."
until curl -s http://localhost:8009/health > /dev/null; do
  sleep 1
done
echo "vllm server is up and running."

#####################################
# Preprocess
#####################################

echo "Preparing reasoning dataset..."
python ${WORK_DIR}/src/prepare_reasoning_dataset.py \
    --dataset_dir "${DATASET_DIR}" \
    --save_filename "${BASE_FILENAME}" \
    --server_url "${SERVER_URL}" \
    --model_name "${MODEL_NAME}" \
    --temperature 1.0 \
    --top_p 0.9 \
    --repetition_penalty 1.05 \
    --max_workers 20

echo "Processing reasoning dataset..."
python ${WORK_DIR}/src/process_reasoning_dataset.py \
    --dataset_dir "${BASE_FILENAME}" \
    --save_filename "${TAG_FILENAME}" \
    --server_url "${SERVER_URL}" \
    --model_name "${MODEL_NAME}" \
    --temperature 1.0 \
    --top_p 0.9 \
    --repetition_penalty 1.05 \
    --max_workers 20

# Stop the vllm server
pkill -f "vllm.entrypoints.openai.api_server"
echo "vllm server has been stopped."