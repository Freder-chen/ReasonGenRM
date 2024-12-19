from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from openrlhf.datasets.utils import zero_pad_sequences


def preprocess_data(
    data,
    input_template=None,
    prompt_key=None,
    chosen_key="chosen",
    rejected_key="rejected",
    apply_chat_template=None,
    is_dpo=False,
) -> str:
    assert apply_chat_template is not None and prompt_key is not None
    assert len(data[chosen_key]) == 2 and len(data[rejected_key]) == 2

    prompt = apply_chat_template(data[prompt_key], tokenize=False, add_reason_prompt=True)
    
    chosen_reason = apply_chat_template(data[prompt_key] + data[chosen_key][:1], tokenize=False)
    chosen_reason_suffix = apply_chat_template(data[prompt_key] + data[chosen_key][:1], tokenize=False, add_generation_prompt=True)
    chosen_response = apply_chat_template(data[prompt_key] + data[chosen_key], tokenize=False)

    rejected_reason = apply_chat_template(data[prompt_key] + data[rejected_key][:1], tokenize=False)
    rejected_reason_suffix = apply_chat_template(data[prompt_key] + data[rejected_key][:1], tokenize=False, add_generation_prompt=True)
    rejected_response = apply_chat_template(data[prompt_key] + data[rejected_key], tokenize=False)

    chosen = {
        "reason": chosen_reason[len(prompt) :],
        "reason_suffix": chosen_reason_suffix[len(chosen_reason) :],
        "response": chosen_response[len(chosen_reason_suffix) :]
    }

    rejected = {
        "reason": rejected_reason[len(prompt) :],
        "reason_suffix": rejected_reason_suffix[len(rejected_reason) :],
        "response": rejected_response[len(rejected_reason_suffix) :]
    }

    return prompt, chosen, rejected


class ReasonDPODataset(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for reward model
        self.tokenizer: self.tokenizer for reward model
        self.max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template=None,
        is_dpo=False,
        num_processors=8,
        multiple_of=1,
    ) -> None:
        super().__init__()
        self.is_dpo = is_dpo
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        self.multiple_of = multiple_of

        # chat_template
        self.input_template = input_template
        self.prompt_key = getattr(self.strategy.args, "prompt_key", None)
        self.chosen_key = getattr(self.strategy.args, "chosen_key", None)
        self.rejected_key = getattr(self.strategy.args, "rejected_key", None)
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

        # Filter out None values if necessary
        processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None)

        # Store the processed data in class attributes
        self.prompts = processed_dataset["prompt"]
        self.chosens = processed_dataset["chosen"]
        self.rejects = processed_dataset["reject"]
        self.infos = processed_dataset["info"]

    def process_data(self, data):
        prompt, chosen, reject = preprocess_data(
            data,
            self.input_template,
            self.prompt_key,
            self.chosen_key,
            self.rejected_key,
            self.apply_chat_template,
            self.is_dpo,
        )
        assert self.is_dpo

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
        chosen_reason_ids_len = prompt_ids_len + get_token_length(chosen["reason"])
        chosen_reason_suffix_ids_len = chosen_reason_ids_len + get_token_length(chosen["reason_suffix"])
        rejected_reason_ids_len = prompt_ids_len + get_token_length(reject["reason"])
        rejected_reason_suffix_ids_len = rejected_reason_ids_len + get_token_length(reject["reason_suffix"])

        # Filter the sample whose length is greater than max_length (64 for answer length)
        if prompt_ids_len >= self.max_length - 64:
            prompt = None
        
        info = {
            "input_length": prompt_ids_len,
            "chosen_reason_suffix_ids_range": [chosen_reason_ids_len, chosen_reason_suffix_ids_len],
            "rejected_reason_suffix_ids_range": [rejected_reason_ids_len, rejected_reason_suffix_ids_len],
        }

        return {
            "prompt": prompt,
            "chosen": chosen["reason"] + chosen["reason_suffix"] + chosen["response"],
            "reject": reject["reason"] + reject["reason_suffix"] + reject["response"],
            "info": info,
        }

    def __len__(self):
        length = len(self.chosens)
        return length

    def __getitem__(self, idx):
        prompt, chosen, reject, info = self.prompts[idx], self.chosens[idx], self.rejects[idx], self.infos[idx]

        chosen = (prompt + chosen).rstrip("\n")
        if not chosen.endswith(self.tokenizer.eos_token):
            chosen += " " + self.tokenizer.eos_token
        chosen_token = self.tokenizer(
            chosen,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        reject = (prompt + reject).rstrip("\n")
        if not reject.endswith(self.tokenizer.eos_token):
            reject += " " + self.tokenizer.eos_token
        reject_token = self.tokenizer(
            reject,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        # to avoid EOS_token truncation
        chosen_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        reject_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        chosen_token["attention_mask"][0][-1] = True
        reject_token["attention_mask"][0][-1] = True

        return (
            chosen_token["input_ids"],
            chosen_token["attention_mask"],
            reject_token["input_ids"],
            reject_token["attention_mask"],
            info,
        )

    def collate_fn(self, item_list):
        chosen_ids = []
        chosen_masks = []
        reject_ids = []
        rejects_masks = []
        infos = {
            "input_length": [],
            "chosen_reason_suffix_ids_range": [],
            "rejected_reason_suffix_ids_range": [],
        }

        for chosen_id, chosen_mask, reject_id, rejects_mask, info in item_list:
            chosen_ids.append(chosen_id)
            chosen_masks.append(chosen_mask)
            reject_ids.append(reject_id)
            rejects_masks.append(rejects_mask)
            infos["input_length"].append(info["input_length"])
            infos["chosen_reason_suffix_ids_range"].append(info["chosen_reason_suffix_ids_range"])
            infos["rejected_reason_suffix_ids_range"].append(info["rejected_reason_suffix_ids_range"])

        padding_side = "right" if self.is_dpo else "left"
        chosen_ids = zero_pad_sequences(chosen_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        chosen_masks = zero_pad_sequences(chosen_masks, side=padding_side)
        reject_ids = zero_pad_sequences(reject_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        rejects_masks = zero_pad_sequences(rejects_masks, side=padding_side)

        return chosen_ids, chosen_masks, reject_ids, rejects_masks, infos

    def packing_collate_fn(self, item_list):
        chosen_ids = []
        chosen_att_masks = []
        chosen_seq_lens = []
        rejected_ids = []
        rejected_att_masks = []
        rejected_seq_lens = []
        infos = {
            "input_length": [],
            "chosen_reason_suffix_ids_range": [],
            "rejected_reason_suffix_ids_range": [],
        }
        
        index = 1
        for chosen_id, chosen_mask, reject_id, rejects_mask, info in item_list:
            chosen_ids.append(chosen_id.flatten())
            chosen_att_masks.append(torch.full_like(chosen_id.flatten(), index))
            chosen_seq_lens.append(len(chosen_id.flatten()))

            rejected_ids.append(reject_id.flatten())
            rejected_att_masks.append(torch.full_like(reject_id.flatten(), index + len(item_list)))
            rejected_seq_lens.append(len(reject_id.flatten()))

            infos["input_length"].append(info["input_length"])
            infos["chosen_reason_suffix_ids_range"].append(info["chosen_reason_suffix_ids_range"])
            infos["rejected_reason_suffix_ids_range"].append(info["rejected_reason_suffix_ids_range"])
            index += 1

        packed_input_ids = torch.cat(chosen_ids + rejected_ids, dim=0).unsqueeze(0)
        packed_attention_masks = torch.cat(chosen_att_masks + rejected_att_masks, dim=0).unsqueeze(0)
        packed_seq_lens = chosen_seq_lens + rejected_seq_lens

        if self.multiple_of > 1 and packed_input_ids.numel() % self.multiple_of != 0:
            padding_len = self.multiple_of - (packed_input_ids.numel() % self.multiple_of)
            packed_input_ids = F.pad(packed_input_ids, (0, padding_len), value=self.tokenizer.pad_token_id)
            packed_attention_masks = F.pad(packed_attention_masks, (0, padding_len), value=0)

        return packed_input_ids, packed_attention_masks, packed_seq_lens, infos
