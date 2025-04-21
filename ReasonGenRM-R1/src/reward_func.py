import re
import torch


def extract_verdict(text: list[str]) -> list[str]:
    found = re.findall(r'\[\[(A|B)\]\]', text)
    return f"[[{found[-1]}]]" if found else None


def compute_rewarad(
    response,
    target, # [[A]] or [[B]]
    correct_score=1.0,
    verdict_correct_score=0.1,
    format_score=-0.9,
    error_score=-1.0,
):
    if not response.endswith("<｜end▁of▁sentence｜>"):
        return error_score

    response = response[:-len("<｜end▁of▁sentence｜>")].strip()

    # think_matches = list(re.finditer(r"<think>(.*?)</think>", response, re.DOTALL))
    # The new version of chat template include the opening tag `<think>`
    think_matches = list(re.finditer(r"(.*?)</think>", response, re.DOTALL))
    if not think_matches:
        return error_score

    last_think_end = think_matches[-1].end()
    content_after = response[last_think_end:].strip()

    if content_after == target.strip():
        return correct_score
    elif extract_verdict(content_after) == target.strip():
        return verdict_correct_score
    else:
        return format_score
    

def reward_func(queries, prompts, labels):
    # queries is prompts + responses
    # labels is answers

    rewards = list()
    for query, prompt, label in zip(queries, prompts, labels):
        reward = compute_rewarad(query[len(prompt):], label)
        rewards.append(reward)
    return torch.tensor(rewards)
