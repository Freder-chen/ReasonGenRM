# dataset name: Skywork/Skywork-Reward-Preference-80K-v0.2

SCRIPT_DIR="$(dirname "$0")"
WORK_DIR="$(realpath "$SCRIPT_DIR")"

MODEL_NAME_OR_PATH="Qwen/Qwen2.5-14B-Instruct"
DATASET_DIR="Skywork/Skywork-Reward-Preference-80K-v0.2"
SAVE_FILENAME="${WORK_DIR}/data/Skywork-Reward-Preference-80K-v0.2.jsonl"

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
    --max-num-seqs 6 \
    --disable-log-requests \
> vllm_server.log 2>&1 &

# Wait for vllm server to be fully up
echo "Waiting for vllm server to start..."
while ! curl -s http://localhost:8009/health > /dev/null; do
  sleep 1  # Check every second
done

echo "vllm server has started, proceeding with the next steps..."

# Prepare reasoning dataset
python ${WORK_DIR}/src/prepare_reasoning_dataset.py \
    --dataset_dir "${DATASET_DIR}" \
    --save_filename "${SAVE_FILENAME}" \
    --server_url "http://0.0.0.0:8009/v1/chat/completions" \
    --model_name "qwen2_instruct" \
    --temperature 1.0 \
    --top_p 0.9 \
    --repetition_penalty 1.05 \
    --max_workers 10

echo "Reasoning dataset has been prepared."

pkill -f "vllm.entrypoints.openai.api_server"
echo "vllm server has been stopped."