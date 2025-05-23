from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from openrlhf.datasets.utils import zero_pad_sequences


def preprocess_data(data, input_template=None, input_key="input", reason_key=None, output_key=None, apply_chat_template=None):
    assert apply_chat_template is not None and output_key is not None

    prompt_message = [{"role": "user", "content": data[input_key]}]
    reason_message = prompt_message + [{"role": "reason", "content": data[reason_key]}]
    full_message = reason_message + [{"role": "assistant", "content": data[output_key]}]

    prompt_continue_text = apply_chat_template(prompt_message, tokenize=False, add_reason_prompt=True)
    reason_text = apply_chat_template(reason_message, tokenize=False)
    reason_continue_text = apply_chat_template(reason_message, tokenize=False, add_generation_prompt=True)
    assistant_text = apply_chat_template(full_message, tokenize=False)

    # Remove trailing newline texts, e.g. Qwen
    reason_text = reason_text.strip()
    assistant_text = assistant_text.strip()

    return (
        prompt_continue_text,
        reason_text[len(prompt_continue_text) :],
        reason_continue_text[len(reason_text) :],
        assistant_text[len(reason_continue_text) :]
    )


class ReasonSFTDataset(Dataset):
    """
    Dataset for SFT model

    Args:
        dataset: dataset for SFT model
        tokenizer: tokenizer for SFT model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template=None,
        pretrain_mode=False,
        num_processors=8,  # Specify the number of processors you want to use
        multiple_of=1,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.pretrain_mode = pretrain_mode
        self.max_length = max_length
        self.multiple_of = multiple_of

        # chat template
        self.input_template = input_template
        self.input_key = getattr(self.strategy.args, "input_key", None)
        self.reason_key = 'reason'
        self.output_key = getattr(self.strategy.args, "output_key", None)
        self.apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if self.apply_chat_template:
            self.apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(self.strategy.args, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template

        # Parallel loading datasets
        processed_dataset = dataset.map(
            self.process_data, remove_columns=dataset.column_names, num_proc=num_processors
        )
        processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None)

        # Store the processed data in class attributes
        self.prompts = processed_dataset["prompt"]
        self.reasons = processed_dataset["reason"]
        self.reason_suffixs = processed_dataset["reason_suffix"]
        self.responses = processed_dataset["response"]
        self.prompt_ids_lens = processed_dataset["prompt_ids_len"]
        self.reason_suffix_ids_ranges = processed_dataset["reason_suffix_ids_range"]

    def process_data(self, data):
        prompt, reason, reason_suffix, response = preprocess_data(
            data,
            None if self.pretrain_mode else self.input_template,
            self.input_key,
            self.reason_key,
            self.output_key,
            apply_chat_template=None if self.pretrain_mode else self.apply_chat_template,
        )
        if not self.pretrain_mode:
            def get_token_length(text):
                token = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding=False,
                    truncation=True,
                    return_tensors="pt",
                    add_special_tokens=False,
                )
                return token["attention_mask"].int().sum().item()

            prompt_ids_len = get_token_length(prompt)
            reason_ids_len = prompt_ids_len + get_token_length(reason)
            reason_suffix_ids_len = reason_ids_len + get_token_length(reason_suffix)

            # filter the sample whose length is greater than max_length (64 for answer length)
            if not prompt or not reason or not reason_suffix or not response or prompt_ids_len >= self.max_length - 64:
                prompt = None
        else:
            prompt_ids_len = 0
            reason_suffix_ids_range = [0, 0]

        return {
            "prompt": prompt,
            "reason": reason,
            "reason_suffix": reason_suffix,
            "response": response,
            "prompt_ids_len": prompt_ids_len,
            "reason_suffix_ids_range": [reason_ids_len, reason_suffix_ids_len]
        }

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        reason = self.reasons[idx]
        reason_suffix = self.reason_suffixs[idx]
        response = self.responses[idx]

        prompt_ids_len = self.prompt_ids_lens[idx]
        reason_suffix_ids_range = self.reason_suffix_ids_ranges[idx]

        if not self.pretrain_mode:
            text = (prompt + reason + reason_suffix + response).rstrip("\n")
            if not text.endswith(self.tokenizer.eos_token):
                text += " " + self.tokenizer.eos_token
        else:
            text = prompt

        input_token = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        if not self.pretrain_mode:
            # to avoid EOS_token truncation
            input_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
            input_token["attention_mask"][0][-1] = True
        
        info = {
            "input": prompt,
            "output": response,
            "input_length": input_token["attention_mask"].int().sum().item(),
            "reason_suffix_ids_range": reason_suffix_ids_range
        }

        return prompt_ids_len, input_token["input_ids"], input_token["attention_mask"], info

    def collate_fn(self, item_list):
        prompt_ids_lens = []
        input_ids = []
        attention_masks = []
        infos = {"input": [], "output": []}

        for prompt_ids_len, input_id, attention_mask, info in item_list:
            prompt_ids_lens.append(prompt_ids_len)
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            infos["input"].append(info["input"])
            infos["output"].append(info["output"])

        input_ids = zero_pad_sequences(input_ids, "right", self.tokenizer.pad_token_id)
        attention_masks = zero_pad_sequences(attention_masks, "right")
        return prompt_ids_lens, input_ids, attention_masks, infos

    def packing_collate_fn(self, item_list):
        packed_input_ids = []
        packed_attention_masks = []
        prompt_ids_lens = []
        infos = {"input_length": [], "reason_suffix_ids_range": []}

        index = 1
        for prompt_ids_len, input_id, attention_mask, info in item_list:
            packed_input_ids.append(input_id.flatten())
            packed_attention_masks.append(torch.full_like(input_id.flatten(), index))
            prompt_ids_lens.append(prompt_ids_len)
            infos["input_length"].append(info["input_length"])
            infos["reason_suffix_ids_range"].append(info["reason_suffix_ids_range"])
            index += 1

        packed_input_ids = torch.cat(packed_input_ids, dim=0).unsqueeze(0)
        packed_attention_masks = torch.cat(packed_attention_masks, dim=0).unsqueeze(0)
        
        if self.multiple_of > 1 and packed_input_ids.numel() % self.multiple_of != 0: # not divisible by multiple_of; here we align for grouping
            padding_len = self.multiple_of - (packed_input_ids.numel() % self.multiple_of)
            packed_input_ids = F.pad(packed_input_ids, (0, padding_len), value=self.tokenizer.pad_token_id)
            packed_attention_masks = F.pad(packed_attention_masks, (0, padding_len), value=0)

        return prompt_ids_lens, packed_input_ids, packed_attention_masks, infos
