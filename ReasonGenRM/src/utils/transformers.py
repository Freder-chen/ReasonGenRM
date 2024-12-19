import torch

from transformers import AutoConfig, AutoTokenizer


def load_custom_tokenizer(model_path, trust_remote_code=True, use_fast=True, use_reason_template=True):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=trust_remote_code, use_fast=use_fast
    )

    config = AutoConfig.from_pretrained(model_path)
    model_type = config.model_type

    if use_reason_template:
        if 'qwen2' in model_type.lower():
            from src.utils.chat_templates import QWEN_REASON_CHAT_TEMPLATE
            tokenizer.chat_template = QWEN_REASON_CHAT_TEMPLATE
        elif 'llama' in model_type.lower():
            from src.utils.chat_templates import LLAMA_REASON_CHAT_TEMPLATE
            tokenizer.chat_template = LLAMA_REASON_CHAT_TEMPLATE
        else:
            raise NotImplementedError(f"Reason Template for this model ({model_type}) not implemented yet.")

    if 'llama' in model_type.lower():
        tokenizer.pad_token = '<|finetune_right_pad_id|>'

    return tokenizer


def compute_conditional_probabilities(model, tokenizer, message):
    prompt_message = message[:-2]    # [{'role': 'user', 'content': prompt}]
    reason_message = [message[-2]]    # [{'role': 'reason', 'content': reason}]
    assistant_message = [message[-1]] # [{'role': 'assistant', 'content': target}]

    prompt_continue_text = tokenizer.apply_chat_template(prompt_message, tokenize=False, add_reason_prompt=True)
    prompt_reason_text = tokenizer.apply_chat_template(prompt_message + reason_message, tokenize=False)
    prompt_reason_continue_text = tokenizer.apply_chat_template(prompt_message + reason_message, tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(prompt_message + reason_message + assistant_message, tokenize=False)

    prompt_continue_ids = tokenizer.encode(prompt_continue_text, return_tensors='pt', add_special_tokens=False)
    prompt_reason_ids = tokenizer.encode(prompt_reason_text, return_tensors='pt', add_special_tokens=False)
    prompt_reason_continue_ids = tokenizer.encode(prompt_reason_continue_text, return_tensors='pt', add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, return_tensors='pt', add_special_tokens=False)

    prompt_continue_len = prompt_continue_ids.size(1)
    prompt_reason_len = prompt_reason_ids.size(1)
    prompt_reason_continue_len = prompt_reason_continue_ids.size(1)
    full_len = full_ids.size(1)

    with torch.no_grad():
        full_ids = full_ids.to('cuda')
        logits = model(full_ids).logits
        log_probs = torch.log_softmax(logits, dim=-1) # .cpu()

        # Compute P(reason | prompt)
        reason_token_ids = full_ids[0, prompt_continue_len:prompt_reason_len]
        reason_pred_probs = log_probs[0, prompt_continue_len - 1: prompt_reason_len - 1].gather(dim=-1, index=reason_token_ids.unsqueeze(-1)).squeeze(-1)
        p_p2r = torch.exp(reason_pred_probs.mean()).cpu().item()

        # Compute P(assistant | prompt, reason)
        assistant_token_ids = full_ids[0, prompt_reason_continue_len:full_len]
        assistant_pred_probs = log_probs[0, prompt_reason_continue_len - 1 : full_len - 1].gather(dim=-1, index=assistant_token_ids.unsqueeze(-1)).squeeze(-1)
        p_pr2a = torch.exp(assistant_pred_probs.mean()).cpu().item()

    return p_p2r, p_pr2a