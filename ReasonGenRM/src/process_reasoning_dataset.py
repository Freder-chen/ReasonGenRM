import os
import re
import json
import argparse
import asyncio
import aiohttp
import aiofiles
from tqdm.asyncio import tqdm
from datasets import load_dataset
from utils.prompt_templates import REASON_QUALITY_EVALUATION_PROMPT_TEMPLATE


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate the quality of reasons in a dataset.')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to the dataset directory or file.')
    parser.add_argument('--save_filename', type=str, required=True, help='Filename to save the evaluation results.')
    parser.add_argument('--server_url', type=str, default='http://0.0.0.0:8009/v1/chat/completions', help='URL of the evaluation server.')
    parser.add_argument('--model_name', type=str, default='qwen2_instruct', help='Name of the model to use.')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature for the model.')
    parser.add_argument('--top_p', type=float, default=0.8, help='Top-p sampling parameter.')
    parser.add_argument('--repetition_penalty', type=float, default=1.05, help='Repetition penalty parameter.')
    parser.add_argument('--max_workers', type=int, default=10, help='Maximum number of concurrent workers.')
    return parser.parse_args()


async def fetch_response(session, prompt, args):
    """
    Asynchronously send a request to the server and get the response.
    """
    data = {
        "model": args.model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty
    }
    try:
        async with session.post(args.server_url, json=data) as response:
            response.raise_for_status()
            result = await response.json()
            return result['choices'][0]['message']['content']
    except Exception as e:
        print(f"Request error: {e}")
        return None


async def evaluate_reason(session, reason, args):
    """
    Evaluate the quality of a single reason and extract the score.
    """
    prompt = REASON_QUALITY_EVALUATION_PROMPT_TEMPLATE.format(reason=reason)
    content = await fetch_response(session, prompt, args)
    if content:
        match = re.search(r'\[(\d)\]', content)
        if match:
            return int(match.group(1))
        else:
            print(f"Score not found: {content}")
    return None


async def save_result(data, filename):
    """
    Asynchronously save data as a line in a JSONL file.
    """
    async with aiofiles.open(filename, 'a', encoding='utf-8') as f:
        await f.write(json.dumps(data, ensure_ascii=False) + '\n')


async def process_item(item, args, session, semaphore):
    """
    Process a single data item: evaluate the reason and save the result.
    """
    async with semaphore:
        score = await evaluate_reason(session, item['reason'], args)
        if score is not None:
            item['reason_score'] = score
            await save_result(item, args.save_filename)


async def main():
    args = parse_args()

    # Load the dataset
    if args.dataset_dir.endswith(('.json', '.jsonl')):
        dataset = load_dataset('json', data_files=args.dataset_dir, split='train')
    else:
        dataset = load_dataset(args.dataset_dir, split='train')

    # Filter items that contain the 'reason' field
    items = [item for item in dataset if 'reason' in item]
    print(f"Total items: {len(dataset)}, Items with 'reason': {len(items)}")

    # Create the directory to save results
    os.makedirs(os.path.dirname(os.path.abspath(args.save_filename)), exist_ok=True)

    semaphore = asyncio.Semaphore(args.max_workers)
    async with aiohttp.ClientSession() as session:
        tasks = [process_item(item, args, session, semaphore) for item in items]
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            await task


if __name__ == '__main__':
    asyncio.run(main())
