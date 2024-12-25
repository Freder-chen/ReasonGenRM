import os
import json
import asyncio
import aiohttp
import aiofiles
import argparse

from tqdm.asyncio import tqdm
from datasets import load_dataset

from src.utils import (
    load_custom_tokenizer,
    is_valid_item,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Process a dataset and generate reasoning responses.')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to the input dataset file or directory.')
    parser.add_argument('--save_filename', type=str, required=True, help='Path to the output JSONL file.')
    parser.add_argument('--server_url', type=str, default='http://0.0.0.0:8009/generate', help='Server URL for generating responses.')
    parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct', help='Name of the model to use.')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature for response generation.')
    parser.add_argument('--top_p', type=float, default=1.0, help='Top-p sampling parameter for response generation.')
    parser.add_argument('--max_tokens', type=int, default=8*1024, help='Maximum number of tokens in the generated response.')
    parser.add_argument('--max_workers', type=int, default=10, help='Maximum number of concurrent workers.')
    parser.add_argument('--samples', type=int, default=None, help='Number of samples to process from the dataset.')
    parser.add_argument('--iter', type=int, default=None, help='Index of the current iter.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--N', type=int, default=1, help='Number of responses to generate per prompt.')
    return parser.parse_args()


async def fetch_generate(prompt: str, max_tokens: int = 50, args=None, session=None):
    """
    Asynchronously send a request to the server and get the generation.
    """
    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "n": args.N,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
    }

    async with session.post(args.server_url, headers=headers, json=data) as response:
        response.raise_for_status()
        result = await response.json()
    return result["text"]


async def generate_response(tokenizer, message, args, session, max_new_tokens=1024, add_reason_prompt=False, add_generation_prompt=False):
    """
    Generate a response based on message and token limits.
    """
    input_text = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_reason_prompt=add_reason_prompt,
        add_generation_prompt=add_generation_prompt
    )
    response_texts = await fetch_generate(input_text, max_new_tokens, args, session)
    return [text[len(input_text):].strip() for text in response_texts]


async def process_data_item(tokenizer, item, args, session):
    """
    Process a single dataset item: compare responses and generate reasoning.
    """
    message = [{'role': 'user', 'content': item['prompt']}]
    reasons = await generate_response(
        tokenizer,
        message,
        args,
        session,
        add_reason_prompt=True,
        max_new_tokens=args.max_tokens,
    )
    return reasons


async def save_result(data, filename):
    """
    Asynchronously save data as a line in a JSONL file.
    """
    async with aiofiles.open(filename, 'a', encoding='utf-8') as f:
        await f.write(json.dumps(data, ensure_ascii=False) + '\n')


async def process_and_save(tokenizer, item, args, session, semaphore):
    """
    Process an item asynchronously and save the result.
    """
    async with semaphore:
        reasons = await process_data_item(tokenizer, item, args, session)
        if reasons:
            result = {
                'prompt': item['prompt'],
                'reasons': reasons,
                'target': item['target']
            }
            await save_result(result, args.save_filename)


async def main():
    args = parse_args()

    # Setup tokenizer
    tokenizer = load_custom_tokenizer(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
        use_reason_template=True # IMPORTANT: Set REASON_CHAT_TEMPLATE in tokenizer
    )

    # Load the dataset
    if args.dataset_dir.endswith(('.json', '.jsonl')):
        dataset = load_dataset('json', data_files=args.dataset_dir, split='train')
    else:
        dataset = load_dataset(args.dataset_dir, split='train')

    if 'user' in dataset.column_names:
        dataset = dataset.rename_column("user", "prompt")
    if 'response' in dataset.column_names:
        dataset = dataset.rename_column("response", "target")

    if args.samples:
        start_idx = args.iter * args.samples if args.iter is not None else 0
        end_idx = start_idx + args.samples
        dataset = dataset.select(range(start_idx, min(end_idx, len(dataset))))

    assert 'prompt' in dataset.column_names, "Dataset must contain a 'prompt' column"
    assert 'target' in dataset.column_names, "Dataset must contain a 'target' column"

    # Ensure the directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.save_filename)), exist_ok=True)

    # Checkpoint
    if os.path.exists(args.save_filename):
        checkpoint_dataset = load_dataset('json', data_files=args.save_filename, split='train')
        checkpoint_prompts = set(checkpoint_dataset['prompt'])
        dataset = dataset.filter(lambda x: x['prompt'] not in checkpoint_prompts)

    # Process the items
    semaphore = asyncio.Semaphore(args.max_workers)
    async with aiohttp.ClientSession() as session:
        tasks = [process_and_save(tokenizer, item, args, session, semaphore) for item in dataset]
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Internal Reason"):
            await task


if __name__ == '__main__':
    asyncio.run(main())
