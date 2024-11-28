import argparse
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from utils.chat_templates import QWEN_RASON_CHAT_TEMPLATE


def parse_args():
    parser = argparse.ArgumentParser(description='Process and tokenize dataset.')
    parser.add_argument('--max_tokens', type=int, default=1024 * 6, help='Maximum number of tokens.')
    parser.add_argument('--dataset_paths', type=str, nargs='+', required=True, help='Paths to dataset files or directories.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pretrained model.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the processed dataset.')
    return parser.parse_args()


def load_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    if 'qwen2' in model_path.lower():
        tokenizer.chat_template = QWEN_RASON_CHAT_TEMPLATE
    else:
        raise NotImplementedError("Reason Template for this model not implemented yet.")
    return tokenizer


def preprocess_function(examples, tokenizer, max_tokens):
    # Helper function to extract the first existing key's values
    def get_field(data, keys):
        for key in keys:
            if key in data:
                return data[key]
        raise ValueError(f"Missing required keys: {keys}")

    # Extract prompts, reasons, and responses
    prompts = get_field(examples, ['prompt', 'user'])
    reasons = get_field(examples, ['reason', 'reasoning'])
    responses = get_field(examples, ['response', 'assistant'])

    # Filter out invalid entries
    def is_valid(p, r, a):
        if ('[[A]]' in r and '[[B]]' in a) or ('[[B]]' in r and '[[A]]' in a):
            return False
        if r.startswith('[[A]]') or r.startswith('[[B]]'):
            return False
        return True

    filtered_data = [
        (p, r, a) for p, r, a in zip(prompts, reasons, responses) if is_valid(p, r, a)
    ]

    if not filtered_data:
        return {'input_ids': [], 'labels': [], 'attention_mask': []}

    prompts, reasons, responses = zip(*filtered_data)

    # Build message sequences
    prompt_messages = [[{"role": "user", "content": p}] for p in prompts]
    reason_messages = [
        [{"role": "user", "content": p}, {"role": "reason", "content": r}]
        for p, r in zip(prompts, reasons)
    ]
    full_messages = [
        [{"role": "user", "content": p}, {"role": "reason", "content": r}, {"role": "assistant", "content": a}]
        for p, r, a in zip(prompts, reasons, responses)
    ]

    # Tokenize messages
    prompt_continue_ids = tokenizer.apply_chat_template(prompt_messages, tokenize=True, add_reason_prompt=True)
    reason_ids = tokenizer.apply_chat_template(reason_messages, tokenize=True)
    reason_continue_ids = tokenizer.apply_chat_template(reason_messages, tokenize=True, add_generation_prompt=True)
    assistant_ids = tokenizer.apply_chat_template(full_messages, tokenize=True)

    # Remove trailing newline tokens
    newline_token_id = tokenizer.encode("\n", add_special_tokens=False)[0]
    reason_ids = [ids[:-1] if ids and ids[-1] == newline_token_id else ids for ids in reason_ids]
    assistant_ids = [ids[:-1] if ids and ids[-1] == newline_token_id else ids for ids in assistant_ids]

    # Calculate lengths
    prompt_continue_lens = [len(ids) for ids in prompt_continue_ids]
    reason_lens = [len(ids) for ids in reason_ids]
    reason_continue_lens = [len(ids) for ids in reason_continue_ids]
    assistant_lens = [len(ids) for ids in assistant_ids]

    input_ids_list = []
    labels_list = []
    attention_masks = []

    for i in range(len(prompts)):
        if assistant_lens[i] >= max_tokens:
            continue

        input_ids = assistant_ids[i]
        labels = [-100] * assistant_lens[i]

        # Label the reasoning and assistant's response
        labels[prompt_continue_lens[i]:reason_lens[i]] = input_ids[prompt_continue_lens[i]:reason_lens[i]]
        labels[reason_continue_lens[i]:] = input_ids[reason_continue_lens[i]:]

        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_masks.append([1] * assistant_lens[i])

    return {
        'input_ids': input_ids_list,
        'labels': labels_list,
        'attention_mask': attention_masks
    }


def main():
    args = parse_args()
    tokenizer = load_tokenizer(args.model_path)

    all_datasets = []

    for dataset_path in args.dataset_paths:
        # Load dataset
        if dataset_path.endswith(('.json', '.jsonl')):
            dataset = load_dataset('json', data_files=dataset_path, split='train')
        else:
            dataset = load_dataset(dataset_path, split='train')

        # Filter datasets by reason score if available
        if 'reason_score' in dataset.column_names:
            dataset = dataset.filter(lambda x: x['reason_score'] in [4, 5])

        print(f"Original dataset length: {len(dataset)}")

        # Preprocess dataset
        tokenized_dataset = dataset.map(
            lambda examples: preprocess_function(examples, tokenizer, args.max_tokens),
            batched=True,
            remove_columns=dataset.column_names
        )

        print(f"Processed dataset length: {len(tokenized_dataset)}")
        all_datasets.append(tokenized_dataset)

    # Concatenate all tokenized datasets
    if all_datasets:
        final_dataset = concatenate_datasets(all_datasets)
        print(f"Total dataset length: {len(final_dataset)}")

        # Save the tokenized dataset
        final_dataset.save_to_disk(args.output_dir)
    else:
        print("No data to save.")


if __name__ == '__main__':
    main()
