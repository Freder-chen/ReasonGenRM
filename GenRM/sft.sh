#!/bin/bash

SCRIPT_DIR="$(dirname "$0")"
WORK_DIR="$(realpath "$SCRIPT_DIR")"

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1
export NCCL_ALGO=Tree

# # Llama-3.1-8B-GenRM
# dataset_name_or_path="${WORK_DIR}/data/Skywork-Reward-Preference-80K-v0.2/base.jsonl"
# dataset_disk_path="${WORK_DIR}/data/Skywork-Reward-Preference-80K-v0.2/sft_disk_llama"
# model_name_or_path="meta-llama/Llama-3.1-8B-Instruct"
# exp_outdir="${WORK_DIR}/exp/Llama-3.1-8B-GenRM"
# deepspeed_config="${WORK_DIR}/deepspeed_config/ds_z1_config.json"

# Qwen2.5-7B-GenRM
dataset_name_or_path="${WORK_DIR}/data/Skywork-Reward-Preference-80K-v0.2/base.jsonl"
dataset_disk_path="${WORK_DIR}/data/Skywork-Reward-Preference-80K-v0.2/sft_disk_qwen"
model_name_or_path="Qwen/Qwen2.5-7B-Instruct"
exp_outdir="${WORK_DIR}/exp/Qwen2.5-7B-GenRM"
deepspeed_config="${WORK_DIR}/deepspeed_config/ds_z1_config.json"


if [ ! -d "${dataset_disk_path}" ]; then
    python ${WORK_DIR}/src/tokenize_sft_dataset.py \
        --max_tokens 16384 \
        --dataset_paths ${dataset_name_or_path} \
        --model_path ${model_name_or_path} \
        --output_dir ${dataset_disk_path}
fi

accelerate launch ${WORK_DIR}/src/train_sft.py \
    --deepspeed ${deepspeed_config} \
    --model_name_or_path ${model_name_or_path} \
    --dataset_name ${dataset_disk_path} \
    --learning_rate 5.0e-6 \
    --num_train_epochs 3 \
    --max_seq_length 32768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --warmup_steps 40 \
    --gradient_checkpointing \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --weight_decay 0.1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --trust_remote_code True \
    --save_strategy steps \
    --save_steps -1 \
    --save_total_limit 3 \
    --eval_strategy no \
    --report_to tensorboard \
    --output_dir ${exp_outdir} \
    --bf16
