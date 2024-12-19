#!/bin/bash

SCRIPT_DIR="$(dirname "$0")"
WORK_DIR="$(realpath "$SCRIPT_DIR/../..")"

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONPATH=$PYTHONPATH:$WORK_DIR

MODEL_NAME_OR_PATH="${WORK_DIR}/exp/Llama-3.1-8B-ReasonRM-Pro/sft"

# vLLM Config
VLLM_PORT=8009
LOG_DIR="$WORK_DIR/logs"
mkdir -p "$LOG_DIR"

# Start vllm server in the background and redirect output to a log file
CUDA_VISIBLE_DEVICES=0,1 \
nohup python -m vllm.entrypoints.api_server \
  --port ${VLLM_PORT} \
  --model ${MODEL_NAME_OR_PATH} \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 32768 \
  --max-num-batched-tokens 131072 \
  --max-num-seqs 10 \
  --disable-log-requests \
> "${LOG_DIR}/vllm_server_eval.log" 2>&1 &

# Wait for vllm server to be fully up
echo "Waiting for vllm server to start..."
while ! curl -s http://localhost:${VLLM_PORT}/health > /dev/null; do
  sleep 1  # Check every second
done

echo "vllm server has started, proceeding with the next steps..."

# Run the evaluation script
python ${WORK_DIR}/src/eval/rewardbench.py \
  --generate_url "http://localhost:${VLLM_PORT}/generate" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --record_filename "${WORK_DIR}/rewardbench_eval.txt" \
  --max_workers 10

echo "Reward benchmark evaluation has finished."

pkill -f "vllm.entrypoints.api_server"
echo "vllm server has been stopped."
