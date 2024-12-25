#!/bin/bash

# NOTE: requires pip install ring_flash_attn

SCRIPT_DIR="$(dirname "$0")"
WORK_DIR="$(realpath "$SCRIPT_DIR/../..")"

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONPATH=$PYTHONPATH:$WORK_DIR # Add reason_openrlhf to the PYTHON path
export NCCL_ALGO=Tree # A800

MODEL_PATH="${WORK_DIR}/exp/Llama3.1-8B-ReasonRM/sft"
DATASET_PATH="${WORK_DIR}/data/Skywork-Reward-Preference-80K-v0.2/dpo/Llama3.1-8B-16sample/reward.jsonl"
SAVE_PATH="${WORK_DIR}/exp/Llama3.1-8B-ReasonRM/dpo"

set -x

read -r -d '' training_commands <<EOF
reason_openrlhf.cli.train_reason_dpo \
   --save_path ${SAVE_PATH} \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 128 \
   --micro_train_batch_size 1 \
   --pretrain ${MODEL_PATH} \
   --bf16 \
   --max_epochs 1 \
   --max_len 131072 \
   --zero_stage 3 \
   --learning_rate 5e-7 \
   --beta 0.1 \
   --dataset ${DATASET_PATH} \
   --apply_chat_template \
   --prompt_key prompt \
   --chosen_key chosen \
   --rejected_key rejected \
   --ring_attn_size 1 \
   --ring_head_stride 2 \
   --packing_samples \
   --flash_attn \
   --gradient_checkpointing \
   --use_tensorboard ${SAVE_PATH}/runs
EOF
# --use_wandb [WANDB_TOKENS] or True (use wandb login command)
# --ipo [for IPO]
# --label_smoothing 0.1 [for cDPO]
# --ref_offload
# --packing_samples
# --nll_loss_coef (Regularization with NLL loss)

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
