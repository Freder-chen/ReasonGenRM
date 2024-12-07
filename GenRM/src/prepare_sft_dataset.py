import os
import json
import random
import argparse
import asyncio
import aiohttp
import aiofiles
from jinja2 import Template
from tqdm.asyncio import tqdm
from datasets import load_dataset

# Copy form https://huggingface.co/Skywork/Skywork-Critic-Llama-3.1-70B
CRITIC_PROMPT_TEMPLATE = """\
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user\'s instructions and answers the user\'s question better. 
Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. 
Please directly output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better.

[User Question]
{question}

[The Start of Assistant A's Answer]
{response_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{response_b}
[The End of Assistant B's Answer]\
"""

HISTORY_DIALOG_CHAT_TEMPLATE = '''
{%- if messages|length == 1 %}
    {{- messages[0]['content'] }}
{%- else %}
{%- for message in messages -%}
    {{- '<turn>' + message['role'] + '\n' }}
    {{- message['content'] + '\n' }}
{%- endfor %}
{%- endif %}
'''
history_template = Template(HISTORY_DIALOG_CHAT_TEMPLATE)


def parse_args():
    parser = argparse.ArgumentParser(description='Process dataset and generate reasoning.')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to the dataset directory.')
    parser.add_argument('--save_filename', type=str, required=True, help='Filename to save the processed data.')
    parser.add_argument('--max_workers', type=int, default=10, help='Maximum number of concurrent workers.')
    return parser.parse_args()


def is_valid_item(item):
    """
    Check if the dataset item is valid based on the structure of the chosen and rejected messages.
    Return True if valid, False otherwise.
    """
    chosen = item['chosen']
    rejected = item['rejected']
    if len(chosen) != len(rejected):
        return False

    if chosen[-1]['role'] != 'assistant' or rejected[-1]['role'] != 'assistant':
        return False
    if chosen[-1]['content'] == rejected[-1]['content']:
        return False
    
    for c, r in zip(item['chosen'][:-1], item['rejected'][:-1]):
        if c['role'] != r['role']:
            return False
        elif c['content'] != r['content']:
            return False

    return True


async def process_data_item(item):
    """
    Process a single dataset item: compare responses and generate reasoning.
    """
    question = history_template.render(messages=item['chosen'][:-1])
    chosen_response = item['chosen'][-1]['content']
    rejected_response = item['rejected'][-1]['content']

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

    return {'prompt': prompt, 'response': selected_response} 


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
        result = await process_data_item(item)
        if result:
            data = {
                'prompt': [{'role': 'user', 'content': result['prompt']}],
                'response': [{ 'role': 'assistant', 'content': result['response']}]
            }
            await save_result(data, args.save_filename)


async def main():
    args = parse_args()

    # Load the dataset
    dataset = load_dataset(args.dataset_dir, split='train')

    # Filter valid items
    valid_dataset = dataset.filter(is_valid_item)
    print(f"Total items: {len(dataset)}, Valid items: {len(valid_dataset)}")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.save_filename)), exist_ok=True)

    semaphore = asyncio.Semaphore(args.max_workers)
    async with aiohttp.ClientSession() as session:
        tasks = [process_and_save(item, args, session, semaphore) for item in valid_dataset]
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            await task


if __name__ == '__main__':
    asyncio.run(main())
