# requires install openrlhf, https://github.com/OpenRLHF/OpenRLHF

SCRIPT_DIR="$(dirname "$0")"
WORK_DIR="$(realpath "$SCRIPT_DIR/../..")"

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1
export NCCL_ALGO=Tree # A800

# MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
MODEL_PATH="Qwen/Qwen2.5-14B-Instruct"
DATASET_PATH="${WORK_DIR}/data/Skywork-Reward-Preference-80K-v0.2/sft.jsonl"
SAVE_PATH="${WORK_DIR}/exp/Qwen2.5-14B-GenRM"

set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --max_len 32768 \
   --dataset ${DATASET_PATH} \
   --input_key prompt \
   --output_key response \
   --train_batch_size 128 \
   --micro_train_batch_size 1 \
   --pretrain ${MODEL_PATH} \
   --save_path ${SAVE_PATH} \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 3 \
   --max_epochs 2 \
   --bf16 \
   --flash_attn \
   --learning_rate 5e-6 \
   --gradient_checkpointing \
   --packing_samples \
   --apply_chat_template \
   --use_tensorboard ${SAVE_PATH}/runs
EOF
# --use_wandb [WANDB_TOKENS]
# --packing_samples

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
