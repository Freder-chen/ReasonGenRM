import torch


@torch.no_grad
def compute_conditional_probabilities(model, tokenizer, message, think_str="</think>"):
    prompt_message = message[:-1]    # [{'role': 'user', 'content': prompt}]
    assistant_message = [message[-1]] # [{'role': 'assistant', 'content': target}]

    prompt_continue_text = tokenizer.apply_chat_template(prompt_message, tokenize=False, add_generation_prompt=True)
    # NOTE: Directly using `apply_chat_template`` will result in the "think" section being removed.
    # full_text = tokenizer.apply_chat_template(prompt_message + assistant_message, tokenize=False)
    full_text = prompt_continue_text + assistant_message[0]["content"] + tokenizer.eos_token + "\n" # NOTE: QwQ format

    prompt_continue_ids = tokenizer.encode(prompt_continue_text, return_tensors="pt", add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, return_tensors="pt", add_special_tokens=False)

    prompt_continue_len = prompt_continue_ids.size(1)
    full_len = full_ids.size(1)
    
    _think_token_ids = tokenizer.encode(think_str, return_tensors='pt', add_special_tokens=False)
    assert _think_token_ids.size(1) == 1
    think_token_id = _think_token_ids[0][0]

    # Locate the final instance of think_token_id within full_ids[0]
    indices = torch.where(full_ids[0] == think_token_id)[0]
    if len(indices) == 0:
        raise ValueError(f"No '{think_str}' token found in the assistant's response.")
    last_think_pos = indices[-1].item()
    prompt_reason_len = last_think_pos + 1

    device = model.device
    full_ids = full_ids.to(device)
    log_probs = torch.log_softmax(model(full_ids).logits, dim=-1)

    # Compute P(reason | prompt)
    reason_token_ids = full_ids[0, prompt_continue_len:prompt_reason_len]
    reason_pred_probs = log_probs[0, prompt_continue_len - 1: prompt_reason_len - 1].gather(dim=-1, index=reason_token_ids.unsqueeze(-1)).squeeze(-1)
    p_p2r = torch.exp(reason_pred_probs.mean()).cpu().item()

    # Compute P(answer | prompt, reason)
    assistant_token_ids = full_ids[0, prompt_reason_len:full_len]
    assistant_pred_probs = log_probs[0, prompt_reason_len - 1 : full_len - 1].gather(dim=-1, index=assistant_token_ids.unsqueeze(-1)).squeeze(-1)
    p_pr2a = torch.exp(assistant_pred_probs.mean()).cpu().item()

    return p_p2r, p_pr2a

@torch.no_grad
def compute_assistant_probabilities(model, tokenizer, message):
    prompt_message = message[:-1]    # [{'role': 'user', 'content': prompt}]
    assistant_message = [message[-1]] # [{'role': 'assistant', 'content': target}]

    prompt_continue_text = tokenizer.apply_chat_template(prompt_message, tokenize=False, add_generation_prompt=True)
    # NOTE: Directly using `apply_chat_template`` will result in the "think" section being removed.
    # full_text = tokenizer.apply_chat_template(prompt_message + assistant_message, tokenize=False)
    full_text = prompt_continue_text + assistant_message[0]["content"] + tokenizer.eos_token + "\n" # NOTE: QwQ format

    prompt_continue_ids = tokenizer.encode(prompt_continue_text, return_tensors="pt", add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, return_tensors="pt", add_special_tokens=False)

    prompt_continue_len = prompt_continue_ids.size(1)
    full_len = full_ids.size(1)

    device = model.device
    full_ids = full_ids.to(device)
    log_probs = torch.log_softmax(model(full_ids).logits, dim=-1)

    # Compute P(answer | prompt)
    assistant_token_ids = full_ids[0, prompt_continue_len:full_len]
    assistant_pred_probs = log_probs[0, prompt_continue_len - 1 : full_len - 1].gather(dim=-1, index=assistant_token_ids.unsqueeze(-1)).squeeze(-1)
    p_p2a = torch.exp(assistant_pred_probs.mean()).cpu().item()

    return p_p2a
