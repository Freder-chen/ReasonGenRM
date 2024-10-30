import argparse
from datasets import load_dataset
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
    # Helper function to get the first existing key's value
    def get_value(data, keys):
        for key in keys:
            if key in data:
                return data[key]
        return None

    # Extract prompts, reasons, and responses
    prompts = get_value(examples, ['prompt', 'user'])
    reasons = get_value(examples, ['reason', 'reasoning'])
    responses = get_value(examples, ['response', 'assistant'])
    if not (prompts and reasons and responses):
        raise ValueError("Missing required keys in dataset.")

    # TODO: Find a better way to handle the mismatch between reason and response
    # Filter out examples using list comprehension
    prompts, reasons, responses = zip(
        *[
            (p, r, a) 
            for p, r, a in zip(prompts, reasons, responses) 
            if not (('[[A]]' in r and '[[B]]' in a) or ('[[B]]' in r and '[[A]]' in a))
        ]
    )

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
    prompt_continue_ids_batch = tokenizer.apply_chat_template(prompt_messages, tokenize=True, add_reason_prompt=True)
    reason_ids_batch = tokenizer.apply_chat_template(reason_messages, tokenize=True)
    reason_continue_ids_batch = tokenizer.apply_chat_template(reason_messages, tokenize=True, add_generation_prompt=True)
    assistant_ids_batch = tokenizer.apply_chat_template(full_messages, tokenize=True)

    # Remove trailing newline tokens
    newline_token_id = tokenizer.encode("\n", add_special_tokens=False)[0]
    reason_ids_batch = [ids[:-1] if ids and ids[-1] == newline_token_id else ids for ids in reason_ids_batch]
    assistant_ids_batch = [ids[:-1] if ids and ids[-1] == newline_token_id else ids for ids in assistant_ids_batch]

    # Calculate lengths
    prompt_continue_lens = [len(ids) for ids in prompt_continue_ids_batch]
    reason_lens = [len(ids) for ids in reason_ids_batch]
    reason_continue_lens = [len(ids) for ids in reason_continue_ids_batch]
    assistant_lens = [len(ids) for ids in assistant_ids_batch]

    input_ids_list = []
    labels_list = []
    attention_masks = []

    for i in range(len(prompts)):
        if assistant_lens[i] >= max_tokens:
            continue

        input_ids = assistant_ids_batch[i]
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
    
    # Load datasets
    datasets = []
    for dataset_path in args.dataset_paths:
        if any(dataset_path.endswith(end) for end in ['.json', '.jsonl']):
            dataset = load_dataset('json', data_files=dataset_path, split="train")
        else:
            dataset = load_dataset(dataset_path, split="train")
        datasets.append(dataset)
    
    # Preprocess datasets
    tokenized_datasets = []
    for dataset in datasets:
        print(f"Origin dataset length: {len(dataset)}")
        tokenized_dataset = dataset.map(
            lambda examples: preprocess_function(examples, tokenizer, args.max_tokens),
            batched=True,
            remove_columns=dataset.column_names
        )
        print(f"Filtered dataset length: {len(tokenized_dataset)}")
        tokenized_datasets.append(tokenized_dataset)

    # Concatenate datasets
    from datasets import concatenate_datasets
    concated_dataset = concatenate_datasets(tokenized_datasets)
    print(f"Total dataset length: {len(concated_dataset)}")
    
    # Save tokenized dataset
    concated_dataset.save_to_disk(args.output_dir)


def test():
    args = parse_args()
    tokenizer = load_tokenizer(args.model_path)
    examples = {
        "prompt": ["I am Qwen."],
        "reason": ["I am Qwen."],
        "response": ["[[A]][[B]]"]
    }
    output = preprocess_function(examples, tokenizer, args.max_tokens)
    print(output['input_ids'])
    print(output['labels'])


if __name__ == '__main__':
    # test()
    main()
