import os
import json
import torch
import argparse
from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoModelForCausalLM

from src.utils import (
    load_custom_tokenizer,
    compute_conditional_probabilities
)


def parse_args():
    parser = argparse.ArgumentParser(description='Process prompts and compute reward scores.')
    parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct', help='Name or path of the model to use.')
    parser.add_argument('--dataset_dir', type=str, nargs='+', required=True, help='Paths to dataset files or directories.')
    parser.add_argument('--save_filename', required=True, type=str, help='Filename to save results.')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers.')
    parser.add_argument('--worker_index', type=int, default=0, help='Index of the current worker.')
    parser.add_argument('--thresh', type=float, default=0.5, help='Threshold for saving results.')
    return parser.parse_args()


def process_data_item(model, tokenizer, item):
    """
    Process a dataset item by calculating scores for each reason.
    """
    scores = []
    for reason in item['reasons']:
        message = [
            {'role': 'user', 'content': item['prompt']},
            {'role': 'reason', 'content': reason},
            {'role': 'assistant', 'content': item['target']},
        ]
        p_p2r, p_pr2a = compute_conditional_probabilities(model, tokenizer, message)
        scores.append({'p2r': p_p2r, 'pr2a': p_pr2a})
    return scores


def merge_dataset_by_prompt(dataset):
    """
    Merge dataset items by 'prompt', combining reasons and ensuring consistent targets.
    """
    grouped_data = defaultdict(lambda: {"target": [], "reasons": []})
    for item in tqdm(dataset, desc='Group dataset'):
        prompt = item["prompt"]
        grouped_data[prompt]["target"].append(item["target"])
        grouped_data[prompt]["reasons"].append(item["reasons"])

    merged_dataset = []
    for prompt, group in tqdm(grouped_data.items(), desc="Merge dataset"):
        if len(set(group["target"])) == 1:
            target = group["target"][0]
            reasons = [item for sublist in group["reasons"] for item in sublist]
            merged_dataset.append({
                "prompt": prompt,
                "reasons": reasons,
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
    tokenizer = load_custom_tokenizer(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
        use_reason_template=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to("cuda").eval()

    # Load the dataset
    all_datasets = []
    for dataset_path in args.dataset_dir:
        if dataset_path.endswith(('.json', '.jsonl')):
            dataset = load_dataset('json', data_files=dataset_path, split='train')
        else:
            dataset = load_dataset(dataset_path, split='train')

        # Rename columns if necessary
        if 'user' in dataset.column_names:
            dataset = dataset.rename_column("user", "prompt")
        if 'response' in dataset.column_names:
            dataset = dataset.rename_column("response", "target")
        
        all_datasets.append(dataset)
    
    dataset = concatenate_datasets(all_datasets)
    dataset = merge_dataset_by_prompt(dataset)

    # Ensure necessary columns are present
    assert 'prompt' in dataset.column_names, "Dataset must contain a 'prompt' column"
    assert 'reasons' in dataset.column_names, "Dataset must contain a 'reasons' column"
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

    # Iterate over the dataset and process each example
    for example in tqdm(dataset, desc="Processing reward examples"):
        # Compute scores for the current example
        scores = process_data_item(model, tokenizer, example)
        # NOTE: The condition `x['p2r'] > x['pr2a']` generally falls into two scenarios:
        # 1. The reasoning process results in incorrect answers.
        # 2. Overfitting occurs, often manifested as redundant or repetitive information.
        reset_incorrect_assistant_score = lambda x: x['pr2a'] * (x['p2r'] < x['pr2a']) * (x['pr2a'] > args.thresh)
        scores = [{'p2r': x['p2r'], 'pr2a': reset_incorrect_assistant_score(x)} for x in scores]

        # Find the indices of the maximum score based on custom scoring logic
        max_score_index = scores.index(max(scores, key=lambda x: x['p2r'] * x['pr2a']))

        # Log the maximum score
        max_score = scores[max_score_index]
        print(f"Score Count: {len(scores)}, Max Score: {max_score}", flush=True)

        # Evaluate if the scores meet the specified threshold criteria:
        # 1. Exclude responses likely to result in incorrect answers. - max_score['pr2a'] > 0
        # 2. Exclude responses that are inherently difficult to generate. - max_score['p2r'] > 0.2
        if args.thresh is None or (max_score['pr2a'] > 0 and max_score['p2r'] > 0.2):
            data = {
                'prompt': example['prompt'],
                'reason': example['reasons'][max_score_index],
                'assistant': example['target']
            }
            # Append the data pair to the specified file
            with open(args.save_filename, 'a', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False)
                file.write('\n')


if __name__ == '__main__':
    main()
