########################
# Script adapted for reward modeling using the Qwen model.
# Based on [trl](https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py).
########################

from typing import Optional
from dataclasses import dataclass
from datasets import load_from_disk
from transformers import AutoTokenizer

from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from utils.chat_templates import QWEN_RASON_CHAT_TEMPLATE


@dataclass
class ScriptArguments:
    """
    Arguments common to all scripts.

    dataset_name (`str`):
        Dataset name.
    dataset_train_split (`str`, *optional*, defaults to `"train"`):
        Dataset split to use for training.
    dataset_test_split (`str`, *optional*, defaults to `"test"`):
        Dataset split to use for evaluation.
    config (`str` or `None`, *optional*, defaults to `None`):
        Path to the optional config file.
    gradient_checkpointing_use_reentrant (`bool`, *optional*, defaults to `False`):
        Whether to apply `use_reentrant` for gradient_checkpointing.
    ignore_bias_buffers (`bool`, *optional*, defaults to `False`):
        Debug argument for distributed training. Fix for DDP issues with LM bias/mask buffers - invalid scalar type,
        inplace operation. See https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992.
    """

    dataset_name: str
    dataset_train_split: str = "train"
    dataset_test_split: str = "test"
    config: Optional[str] = None
    gradient_checkpointing_use_reentrant: bool = False
    ignore_bias_buffers: bool = False


# # Define function for formatting prompts
# def formatting_prompts_func(example):
#     messages = [
#         {"role": "user", "content": example['prompt']},
#         {"role": "reason", "content": example['reason']},
#         {"role": "assistant", "content": example['response']}
#     ]

#     # Tokenize user (prompt), reason, and assistant parts
#     prompt_continue_text = tokenizer.apply_chat_template(messages[:1], tokenize=False, add_reason_prompt=True)
#     reason_text = tokenizer.apply_chat_template(messages[:2], tokenize=False, add_reason_prompt=False)
#     reason_continue_text = tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)
#     assistant_text = tokenizer.apply_chat_template(messages, tokenize=False)

#     # Function to extract input_ids and attention_mask for a given text
#     def extract_ids_and_masks(text):
#         tokenized = tokenizer(text, return_tensors="pt", padding=False, truncation=True)
#         return tokenized["input_ids"].squeeze(0), tokenized["attention_mask"].squeeze(0)

#     # Extract input_ids and attention_mask for different parts
#     prompt_continue_input_ids, _ = extract_ids_and_masks(prompt_continue_text)
#     reason_input_ids, _ = extract_ids_and_masks(reason_text)
#     reason_continue_input_ids, _ = extract_ids_and_masks(reason_continue_text)
#     assistant_input_ids, assistant_attention_mask = extract_ids_and_masks(assistant_text)

#     # Generate labels with appropriate masking (-100 for non-label parts)
#     assistant_labels = assistant_input_ids.clone()
#     prompt_continue_len = len(prompt_continue_input_ids)
#     reason_len = len(reason_input_ids)
#     reason_continue_len = len(reason_continue_input_ids)

#     assistant_labels[:prompt_continue_len] = -100  # Mask prompt part
#     assistant_labels[reason_len:reason_continue_len] = -100  # Mask reason part

#     # Cut off the assistant labels to match the assistant input length
#     assistant_input_ids = assistant_input_ids[:tokenizer.model_max_length]
#     assistant_attention_mask = assistant_attention_mask[:tokenizer.model_max_length]
#     assistant_labels = assistant_labels[:tokenizer.model_max_length]

#     # Return final output with input_ids, attention_mask, and labels
#     return {
#         "input_ids": assistant_input_ids,
#         "attention_mask": assistant_attention_mask,
#         "labels": assistant_labels,
#     }


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()

    ################
    # Model init kwargs & Tokenizer
    ################
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, use_fast=True
    )
    tokenizer.model_max_length = training_args.max_seq_length
    # IMPORTANT: Set REASON_CHAT_TEMPLATE in tokenizer
    if 'qwen2' in model_config.model_name_or_path.lower():
        tokenizer.chat_template = QWEN_RASON_CHAT_TEMPLATE
    else:
        raise NotImplementedError("Reason Template for this model not implemented yet.")

    ################
    # Dataset
    ################
    dataset = load_from_disk(script_args.dataset_name)
    print(f'Dataset length: {len(dataset)}')

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model_config.model_name_or_path,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        peft_config=get_peft_config(model_config),
    )
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
