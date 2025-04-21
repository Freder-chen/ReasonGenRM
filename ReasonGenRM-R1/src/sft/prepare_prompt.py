import os
import json
import random
import argparse
from tqdm import tqdm
from jinja2 import Template
from datasets import load_dataset

from src.utils.prompt_templates import CRITIC_PROMPT_TEMPLATE
from src.utils.chat_templates import HISTORY_DIALOG_CHAT_TEMPLATE
from src.utils import is_valid_item

random.seed(42)

history_template = Template(HISTORY_DIALOG_CHAT_TEMPLATE)


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare prompt dataset.')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to the dataset directory.')
    parser.add_argument('--save_filename', type=str, required=True, help='Filename to save the processed data.')
    parser.add_argument('--dataset_split', type=str, default="train", help='Filename to save the processed data.')
    return parser.parse_args()


def process_data_item(item):
    question = history_template.render(messages=item['chosen'][:-1]).strip()
    chosen_response = item['chosen'][-1]['content'].strip()
    rejected_response = item['rejected'][-1]['content'].strip()

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

    return {'prompt': prompt, 'target': selected_response}


def save_result(data, filename):
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')


def main():
    args = parse_args()

    # Load the dataset
    dataset = load_dataset(args.dataset_dir, split=args.dataset_split)

    # Filter valid items
    valid_dataset = dataset.filter(is_valid_item)
    print(f"Total items: {len(dataset)}, Valid items: {len(valid_dataset)}")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.save_filename)), exist_ok=True)
    for item in tqdm(valid_dataset, desc="Prepare prompt"):
        result = process_data_item(item, args)
        if result:
            save_result(result, args.save_filename)


if __name__ == '__main__':
    main()
