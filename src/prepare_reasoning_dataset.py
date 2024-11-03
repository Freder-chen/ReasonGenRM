import os
import json
import random
import argparse
import asyncio
import aiohttp
import aiofiles
from tqdm.asyncio import tqdm
from datasets import load_dataset
from utils.prompt_templates import CRITIC_PROMPT_TEMPLATE, REASON_PROMPT_TEMPLATE


def parse_args():
    parser = argparse.ArgumentParser(description='Process dataset and generate reasoning.')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to the dataset directory.')
    parser.add_argument('--save_filename', type=str, required=True, help='Filename to save the processed data.')
    parser.add_argument('--server_url', type=str, default='http://0.0.0.0:8009/v1/chat/completions', help='URL of the server to fetch chat completions.')
    parser.add_argument('--model_name', type=str, default='qwen2_instruct', help='Name of the model to use.')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature.')
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


def is_valid_item(item):
    """
    Check if the dataset item is valid based on the structure of the chosen and rejected messages.
    Return True if valid, False otherwise.
    """
    chosen = item['chosen']
    rejected = item['rejected']

    if len(chosen) != 2 or len(rejected) != 2:
        return False
    if chosen[0]['role'] != 'user' or rejected[0]['role'] != 'user':
        return False
    if chosen[1]['role'] != 'assistant' or rejected[1]['role'] != 'assistant':
        return False
    if chosen[0]['content'] != rejected[0]['content']:
        return False
    if chosen[1]['content'] == rejected[1]['content']:
        return False

    return True


async def process_data_item(item, session, args):
    """
    Process a single dataset item: compare responses and generate reasoning.
    """
    question = item['chosen'][0]['content']
    chosen_response = item['chosen'][1]['content']
    rejected_response = item['rejected'][1]['content']

    # Randomly decide the order of responses
    if random.choice([True, False]):
        prompt = CRITIC_PROMPT_TEMPLATE.format(
            question=question,
            response_a=chosen_response,
            response_b=rejected_response
        )
        selected_response = '[[A]]'
    else:
        prompt = CRITIC_PROMPT_TEMPLATE.format(
            question=question,
            response_a=rejected_response,
            response_b=chosen_response
        )
        selected_response = '[[B]]'

    # Generate reasoning
    reason_prompt = REASON_PROMPT_TEMPLATE.format(question=prompt, response=selected_response)
    reason = await fetch_response(session, reason_prompt, args)
    if reason:
        return {'prompt': prompt, 'reason': reason, 'response': selected_response}
    return None


async def save_result(data, filename):
    """
    Asynchronously save data as a line in a JSONL file.
    """
    async with aiofiles.open(filename, 'a', encoding='utf-8') as f:
        await f.write(json.dumps(data, ensure_ascii=False) + '\n')


async def process_and_save(item, args, session, semaphore):
    """
    Process an item asynchronously and save the result.
    """
    async with semaphore:
        result = await process_data_item(item, session, args)
        if result:
            await save_result(result, args.save_filename)


async def main():
    args = parse_args()

    # Load the dataset
    dataset = load_dataset(args.dataset_dir, split='train')

    # Filter valid items
    valid_items = [item for item in dataset if is_valid_item(item)]
    print(f"Total items: {len(dataset)}, Valid items: {len(valid_items)}")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.save_filename)), exist_ok=True)

    semaphore = asyncio.Semaphore(args.max_workers)
    async with aiohttp.ClientSession() as session:
        tasks = [process_and_save(item, args, session, semaphore) for item in valid_items]
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            await task


if __name__ == '__main__':
    asyncio.run(main())
