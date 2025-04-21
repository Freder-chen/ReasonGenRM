import os
import re
import json
import torch
import argparse
from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import (
    compute_conditional_probabilities,
    compute_assistant_probabilities,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Process prompts and compute reward scores.')
    parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct', help='Name or path of the model to use.')
    parser.add_argument('--dataset_list', type=str, nargs='+', required=True, help='Paths to dataset files or directories.')
    parser.add_argument('--save_filename', required=True, type=str, help='Filename to save results.')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers.')
    parser.add_argument('--worker_index', type=int, default=0, help='Index of the current worker.')
    return parser.parse_args()


def normalize_assistant(response):
    think_matches = list(re.finditer(r"(.*?)</think>", response, re.DOTALL))
    if not think_matches:
        return None

    last_think_end = think_matches[-1].end()
    think_str = response[:last_think_end]
    answer_str = response[last_think_end:]

    found = re.findall(r"\[\[(A|B)\]\]", answer_str)
    if not found:
        return None

    norm_response = think_str + "\n\n" + f"[[{found[-1]}]]"
    return norm_response


def extract_verdict(response, answer):
    assert answer in ["A", "B"], f"Invalid answer provided {answer}."

    think_matches = list(re.finditer(r"(.*?)</think>", response, re.DOTALL))
    if not think_matches:
        return 0.0

    last_think_end = think_matches[-1].end()
    answer_str = response[last_think_end:]

    found = re.findall(r'\[\[(A|B)\]\]', answer_str)
    if not found:
        return 0.0

    return 1.0 if found[-1] == answer else -1.0


def process_one_sample(model, tokenizer, sample):
    scores = []
    for response in sample['responses']:
        assert sample['target'] in ["[[A]]", "[[B]]"], f"Invalid answer provided {sample['target']}."
        answer = {"[[A]]": "A", "[[B]]": "B"}[sample['target']]

        norm_response = normalize_assistant(response)
        correct = extract_verdict(norm_response or response, answer=answer)
        message = [
            {'role': 'user', 'content': sample['prompt']},
            {'role': 'assistant', 'content': norm_response or response},
        ]

        if norm_response is not None:
            p_p2r, p_pr2a = compute_conditional_probabilities(model, tokenizer, message)
            scores.append({'p2r': p_p2r, 'pr2a': p_pr2a, "correct": correct})
        else:
            p_p2a = compute_assistant_probabilities(model, tokenizer, message)
            scores.append({'p2r': 1, 'pr2a': p_p2a, "correct": -1})

    return scores


def merge_dataset_by_prompt(dataset):
    """
    Merge dataset items by 'prompt', combining responses and ensuring consistent targets.
    """
    grouped_data = defaultdict(lambda: {"target": [], "responses": []})
    for item in tqdm(dataset, desc='Group dataset'):
        prompt = item["prompt"]
        grouped_data[prompt]["target"].append(item["target"])
        grouped_data[prompt]["responses"].append(item["responses"])

    merged_dataset = []
    for prompt, group in tqdm(grouped_data.items(), desc="Merge dataset"):
        if len(set(group["target"])) == 1:
            target = group["target"][0]
            responses = [item for sublist in group["responses"] for item in sublist]
            merged_dataset.append({
                "prompt": prompt,
                "responses": responses,
                "target": target
            })
        else:
            print(
                "Multiple target values found when merging datasets:\n"
                f"{prompt}\n"
                f"{group}"
            )  # Error

    return Dataset.from_list(merged_dataset)


def split_dataset(dataset, num_workers, worker_index):
    """
    Split the dataset among workers based on the worker index.

    Args:
        dataset (Dataset): The dataset to be split.
        num_workers (int): The total number of workers.
        worker_index (int): The index of the current worker.

    Returns:
        Dataset: A subset of the dataset assigned to the current worker.
    """
    # Validate inputs
    if num_workers <= 0:
        raise ValueError("Number of workers must be greater than zero.")
    if worker_index < 0 or worker_index >= num_workers:
        raise ValueError(f"Worker index must be in range [0, {num_workers - 1}].")

    # Calculate split ranges
    total_samples = len(dataset)
    samples_per_worker = total_samples // num_workers
    remainder = total_samples % num_workers

    if worker_index < remainder:
        start_idx = worker_index * (samples_per_worker + 1)
        end_idx = start_idx + samples_per_worker + 1
    else:
        start_idx = worker_index * samples_per_worker + remainder
        end_idx = start_idx + samples_per_worker
    
    if start_idx >= total_samples:
        return dataset.select([])

    # Ensure indices are within range
    return dataset.select(range(start_idx, min(end_idx, total_samples)))


def main():
    args = parse_args()

    # Setup tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    ).eval()

    # Load the dataset
    all_datasets = []
    for dataset_path in args.dataset_list:
        if dataset_path.endswith(('.json', '.jsonl')):
            dataset = load_dataset('json', data_files=dataset_path, split='train')
        else:
            dataset = load_dataset(dataset_path, split='train')

        all_datasets.append(dataset)

    dataset = concatenate_datasets(all_datasets)
    dataset = merge_dataset_by_prompt(dataset)

    # Ensure necessary columns are present
    assert 'prompt' in dataset.column_names, "Dataset must contain a 'prompt' column"
    assert 'responses' in dataset.column_names, "Dataset must contain a 'responses' column"
    assert 'target' in dataset.column_names, "Dataset must contain a 'target' column"

    # Split the dataset among workers
    if args.num_workers > 1:
        dataset = split_dataset(dataset, args.num_workers, args.worker_index)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.save_filename)), exist_ok=True)

    # Checkpoint
    if os.path.exists(args.save_filename) and os.path.getsize(args.save_filename) > 0:
        checkpoint_dataset = load_dataset('json', data_files=args.save_filename, split='train')
        checkpoint_prompts = set(checkpoint_dataset['prompt'])
        dataset = dataset.filter(lambda x: x['prompt'] not in checkpoint_prompts)

    # Iterate over the dataset and process each sample
    for sample in tqdm(dataset, desc="Processing reward samples"):
        # Compute scores for the current sample
        scores = process_one_sample(model, tokenizer, sample)

        positive_data, negtive_data = [], []
        for response, score in zip(sample['responses'], scores):
            assert score is not None

            if score['correct'] > 0:
                positive_data.append({'response': response, **score})
            else:
                negtive_data.append({'response': response, **score})

        if len(positive_data) == 0 or len(negtive_data) == 0:
            continue
    
        chosen_sample = max(positive_data, key=lambda x: x['p2r'] * x['pr2a'] * x['correct'])
        rejected_sample = min(negtive_data, key=lambda x: x['p2r'] * x['pr2a'] * x['correct'])

        # Log the min-max score
        pos_score = {
            'p2r': chosen_sample['p2r'],
            'pr2a': chosen_sample['pr2a'],
            'correct': chosen_sample['correct']
        }
        neg_score = {
            'p2r': rejected_sample['p2r'],
            'pr2a': rejected_sample['pr2a'],
            'correct': rejected_sample['correct']
        }
        print(f"Score Count: {len(sample['responses'])}, Pos Score: {pos_score}, Neg Score: {neg_score}", flush=True)

        data_pair = {
            'prompt': [{"role": "user", "content": sample['prompt']}],
            'chosen': [
                {"role": "assistant", "content": chosen_sample['response']},
            ],
            'rejected': [
                {"role": "assistant", "content": rejected_sample['response']},
            ],
        }
        # Append the data pair to the specified file
        with open(args.save_filename, 'a', encoding='utf-8') as file:
            json.dump(data_pair, file, ensure_ascii=False)
            file.write('\n')


if __name__ == '__main__':
    main()
