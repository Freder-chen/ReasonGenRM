#!/bin/bash

SCRIPT_DIR="$(dirname "$0")"
WORK_DIR="$(realpath "$SCRIPT_DIR/../..")"

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONPATH=$PYTHONPATH:$WORK_DIR # Add reason_openrlhf to the PYTHON path
export NCCL_ALGO=Tree # A800

MODEL_PATH="Qwen/QwQ-32B"
DATASET_PATH="${WORK_DIR}/data/Skywork-Reward-Preference-80K-v0.2/sft/QwQ-zero/reward.jsonl"
SAVE_PATH="${WORK_DIR}/exp/ReasonGenRM-QwQ-32B/sft"

# NOTE:
# pip install deepspeed==0.15.0
# pip install ring_flash_attn

set -x

read -r -d '' training_commands <<EOF
reason_openrlhf.cli.train_reason_sft \
   --max_len 131072 \
   --dataset ${DATASET_PATH} \
   --input_key prompt \
   --output_key assistant \
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
   --adam_offload \
   --flash_attn \
   --ring_attn_size 2 \
   --ring_head_stride 2 \
   --learning_rate 5e-6 \
   --gradient_checkpointing \
   --packing_samples \
   --apply_chat_template \
   --use_tensorboard ${SAVE_PATH}/runs
EOF
# --use_wandb [WANDB_TOKENS]

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi