import os
import json
import random
import argparse
import datasets
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def fetch_chat(content, args):
    """
    Send a chat request to the server and return the response content.
    """
    data = {
        "model": args.model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content},
        ],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty
    }
    try:
        response = requests.post(args.server_url, headers={"Content-Type": "application/json"}, data=json.dumps(data))
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None


def save_jsonl_item(data, filename):
    """
    Append a single JSON object as a new line to a JSONL file.
    """
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')


def check_item_validity(item):
    """
    Check if the dataset item is valid based on the structure of the chosen and rejected messages.
    Return True if valid, False otherwise.
    """
    chosen_messages = item['chosen']
    rejected_messages = item['rejected']

    # Ensure valid structure with exactly two parts: user message and assistant response
    if len(chosen_messages) != 2 or len(rejected_messages) != 2:
        return False

    # Ensure user message is consistent and identical between chosen and rejected
    if chosen_messages[0]['role'] != 'user' or rejected_messages[0]['role'] != 'user':
        return False

    # Ensure assistant response structure is valid
    if chosen_messages[1]['role'] != 'assistant' or rejected_messages[1]['role'] != 'assistant':
        return False

    chosen_content = chosen_messages[0]['content']
    rejected_content = rejected_messages[0]['content']

    # Skip different user messages
    if chosen_content != rejected_content:
        return False

    chosen_response = chosen_messages[1]['content']
    rejected_response = rejected_messages[1]['content']

    # Skip identical responses
    if chosen_response == rejected_response:
        return False

    return True


def process_item(item, args):
    """
    Process a single dataset item, comparing chosen and rejected messages,
    generating reasoning, and saving the result.
    """
    chosen_messages = item['chosen']
    rejected_messages = item['rejected']

    assert len(chosen_messages) == len(rejected_messages) == 2

    chosen_content = chosen_messages[0]['content']
    chosen_response = chosen_messages[1]['content']
    rejected_response = rejected_messages[1]['content']

    # Prepare the prompts for comparison
    results = []
    random_choice = random.choice([True, False])
    if random_choice:
        prompt = CRITIC_PROMPT_TEMPLATE.format(
            question=chosen_content,
            response_a=chosen_response,
            response_b=rejected_response
        )
        response = '[[A]]'
    else:
        prompt = CRITIC_PROMPT_TEMPLATE.format(
            question=chosen_content,
            response_a=rejected_response,
            response_b=chosen_response
        )
        response = '[[B]]'

    reason = fetch_chat(REASON_PROMPT_TEMPLATE.format(question=prompt, response=response), args)
    if reason:
        results.append({'prompt': prompt, 'reason': reason, 'response': response})

    return results


def process_and_save(item, args):
    """
    Wrapper function to process an item and save the results.
    """
    results = process_item(item, args)
    if results:
        for result in results:
            save_jsonl_item(result, args.save_filename)


def main():
    args = parse_args()

    # Load the dataset
    dataset = datasets.load_dataset(args.dataset_dir, split='train')

    # Filter out invalid items
    valid_items = [item for item in dataset if check_item_validity(item)]
    print(f"Number of skipped items: {len(dataset) - len(valid_items)}, Number of valid items: {len(valid_items)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.save_filename)), exist_ok=True)

    # Use a thread pool to process items concurrently
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_and_save, item, args) for item in valid_items]
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass  # Ensures that exceptions are raised if any


if __name__ == '__main__':
    main()
