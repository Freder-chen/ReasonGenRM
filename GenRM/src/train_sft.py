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
    if 'qwen2' in model_config.model_name_or_path.lower():
        pass
    elif 'llama' in model_config.model_name_or_path.lower():
        tokenizer.pad_token = '<|finetune_right_pad_id|>'
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
