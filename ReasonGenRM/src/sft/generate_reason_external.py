import os
import json
import asyncio
import aiohttp
import aiofiles
import argparse
from tqdm.asyncio import tqdm
from datasets import load_dataset

from src.utils.prompt_templates import REASON_PROMPT_TEMPLATE


def parse_args():
    parser = argparse.ArgumentParser(description='Process a dataset and generate reasoning responses.')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to the input dataset file or directory.')
    parser.add_argument('--save_filename', type=str, required=True, help='Path to the output JSONL file.')
    parser.add_argument('--server_url', type=str, default='http://0.0.0.0:8009/v1/chat/completions', help='Server URL for generating responses.')
    parser.add_argument('--model_name', type=str, default='qwen2_instruct', help='Name of the model to use for generating responses.')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature for response generation.')
    parser.add_argument('--top_p', type=float, default=1.0, help='Top-p sampling parameter for response generation.')
    parser.add_argument('--max_tokens', type=int, default=8*1024, help='Maximum number of tokens in the generated response.')
    parser.add_argument('--max_workers', type=int, default=10, help='Maximum number of concurrent workers.')
    parser.add_argument('--samples', type=int, default=None, help='Number of samples to process from the dataset.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--N', type=int, default=1, help='Number of responses to generate per prompt.')
    return parser.parse_args()


async def fetch_responses(session, prompt, args):
    """Send a prompt to the server and get generated responses."""
    data = {
        "model": args.model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "n": args.N,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
    }
    try:
        async with session.post(args.server_url, json=data) as response:
            response.raise_for_status()
            result = await response.json()
            return [choice['message']['content'] for choice in result.get('choices', [])]
    except Exception as e:
        import traceback; traceback.print_exc()
        return None


async def generate_reasons(item, args, session):
    """Generate reasoning responses for a given dataset item."""
    prompt = REASON_PROMPT_TEMPLATE.format(
        question=item['prompt'],
        response=item['target']
    )
    return await fetch_responses(session, prompt, args)


async def save_result(data, output_file):
    """Asynchronously save data as a line in a JSONL file."""
    async with aiofiles.open(output_file, 'a', encoding='utf-8') as f:
        await f.write(json.dumps(data, ensure_ascii=False) + '\n')


async def process_and_save(item, args, session, semaphore):
    """Process a single dataset item and save the result."""
    async with semaphore:
        reasons = await generate_reasons(item, args, session)
        if reasons:
            result = {
                'prompt': item['prompt'],
                'reasons': [r.strip() for r in reasons],
                'target': item['target']
            }
            await save_result(result, args.save_filename)


async def main():
    args = parse_args()

    # Load the dataset
    if args.dataset_dir.endswith(('.json', '.jsonl')):
        dataset = load_dataset('json', data_files=args.dataset_dir, split='train')
    else:
        dataset = load_dataset(args.dataset_dir, split='train')

    # Rename columns if necessary
    if 'user' in dataset.column_names:
        dataset = dataset.rename_column('user', 'prompt')
    if 'response' in dataset.column_names:
        dataset = dataset.rename_column('response', 'target')

    # Select a subset of the data if specified
    if args.samples:
        dataset = dataset.select(range(args.samples))

    # Ensure required columns are present
    assert 'prompt' in dataset.column_names, "Dataset must contain a 'prompt' column"
    assert 'target' in dataset.column_names, "Dataset must contain a 'target' column"

    # Ensure the directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.save_filename)), exist_ok=True)

    # Checkpoint
    if os.path.exists(args.save_filename):
        checkpoint_dataset = load_dataset('json', data_files=args.save_filename, split='train')
        checkpoint_prompts = set(checkpoint_dataset['prompt'])
        dataset = dataset.filter(lambda x: x['prompt'] not in checkpoint_prompts)

    # Process the dataset items
    semaphore = asyncio.Semaphore(args.max_workers)
    async with aiohttp.ClientSession() as session:
        tasks = [process_and_save(item, args, session, semaphore) for item in dataset]
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="External Reason"):
            await task


if __name__ == '__main__':
    asyncio.run(main())
