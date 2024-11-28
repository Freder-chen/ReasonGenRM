#!/bin/bash

SCRIPT_DIR="$(dirname "$0")"
WORK_DIR="$(realpath "$SCRIPT_DIR")"

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1

MODEL_NAME_OR_PATH="${WORK_DIR}/exp/Qwen2.5-7B-GenRM"

# Start vllm server in the background and redirect output to a log file
CUDA_VISIBLE_DEVICES=0,1,2,3 \
nohup \
python -m vllm.entrypoints.openai.api_server \
    --port 8009 \
    --served-model-name qwen2_instruct \
    --model ${MODEL_NAME_OR_PATH} \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 32768 \
    --max-num-batched-tokens 131072 \
    --max-num-seqs 8 \
    --disable-log-requests \
> "${WORK_DIR}/vllm_server_eval.log" 2>&1 &

# Wait for vllm server to be fully up
echo "Waiting for vllm server to start..."
while ! curl -s http://localhost:8009/health > /dev/null; do
  sleep 1  # Check every second
done

echo "vllm server has started, proceeding with the next steps..."

# Run the evaluation script
python ${WORK_DIR}/src/rewardbench_genrm.py \
    --model_name "${MODEL_NAME_OR_PATH}" \
    --server_url "http://0.0.0.0:8009/v1/chat/completions" \
    --record_filename "${WORK_DIR}/rewardbench_eval.txt" \
    --max_workers 8
echo "Reward benchmark evaluation has finished."

pkill -f "vllm.entrypoints.openai.api_server"
echo "vllm server has been stopped."
