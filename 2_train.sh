# model: Qwen/Qwen2.5-7B-Instruct
# general data: KingNish/reasoning-base-20k

SCRIPT_DIR="$(dirname "$0")"
WORK_DIR="$(realpath "$SCRIPT_DIR")"

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1

# Qwen2.5-7B-ReasonRM
dataset_name_or_path="${WORK_DIR}/data/Skywork-Reward-Preference-80K-v0.2-Qwen2.5-14B-Instruct.jsonl"
dataset_disk_path="${WORK_DIR}/data/Skywork-Reward-Preference-80K-v0.2-Reason"
model_name_or_path="Qwen/Qwen2.5-7B-Instruct"
exp_outdir="${WORK_DIR}/exp/Qwen2.5-7B-ReasonRM"
deepspeed_config="${WORK_DIR}/deepspeed_config/ds_z2_config.json"

# # Qwen2.5-7B-ReasonRM-wGen
# dataset1="${WORK_DIR}/data/Skywork-Reward-Preference-80K-v0.2-Qwen2.5-14B-Instruct.jsonl"
# dataset2="KingNish/reasoning-base-20k"
# dataset_name_or_path="${dataset1} ${dataset2}"
# dataset_disk_path="${WORK_DIR}/data/Skywork-Reward-Preference-80K-v0.2-Reason-wGeneralData"
# model_name_or_path="Qwen/Qwen2.5-7B-Instruct"
# exp_outdir="${WORK_DIR}/exp/Qwen2.5-7B-ReasonRM-wGen"
# deepspeed_config="${WORK_DIR}/deepspeed_config/ds_z2_config.json"

# # Qwen2.5-14B-ReasonRM
# dataset_name_or_path="${WORK_DIR}/data/Skywork-Reward-Preference-80K-v0.2-Qwen2.5-14B-Instruct.jsonl"
# dataset_disk_path="${WORK_DIR}/data/Skywork-Reward-Preference-80K-v0.2-Reason"
# model_name_or_path="Qwen/Qwen2.5-14B-Instruct"
# exp_outdir="${WORK_DIR}/exp/Qwen2.5-14B-ReasonRM"
# deepspeed_config="${WORK_DIR}/deepspeed_config/ds_z3_config.json"

# # Qwen2.5-14B-ReasonRM-wGen
# dataset1="${WORK_DIR}/data/Skywork-Reward-Preference-80K-v0.2-Qwen2.5-14B-Instruct.jsonl"
# dataset2="KingNish/reasoning-base-20k"
# dataset_name_or_path="${dataset1} ${dataset2}"
# dataset_disk_path="${WORK_DIR}/data/Skywork-Reward-Preference-80K-v0.2-Reason-wGeneralData"
# model_name_or_path="Qwen/Qwen2.5-14B-Instruct"
# exp_outdir="${WORK_DIR}/exp/Qwen2.5-14B-ReasonRM-wGen"
# deepspeed_config="${WORK_DIR}/deepspeed_config/ds_z3_config.json"


if [ ! -d "${dataset_disk_path}" ]; then
    # Note: Quality of generated results decreases significantly for inputs exceeding 6k tokens
    python ${WORK_DIR}/src/process_reasoning_dataset.py \
        --max_tokens 6144 \
        --dataset_paths ${dataset_name_or_path} \
        --model_path ${model_name_or_path} \
        --output_dir ${dataset_disk_path}
fi

accelerate launch ${WORK_DIR}/src/train_reasoning_sft.py \
    --deepspeed ${deepspeed_config} \
    --model_name_or_path ${model_name_or_path} \
    --dataset_name ${dataset_disk_path} \
    --learning_rate 5.0e-6 \
    --num_train_epochs 1.1 \
    --max_seq_length 32768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --warmup_ratio 0.03 \
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
    --save_steps 50 \
    --save_total_limit 3 \
    --eval_strategy no \
    --report_to tensorboard \
    --output_dir ${exp_outdir} \
    --bf16
