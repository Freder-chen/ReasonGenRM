import os
import json
import argparse
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description='Process a dataset and generate reasoning responses.')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to the input dataset file or directory.')
    parser.add_argument('--save_filename', type=str, required=True, help='Path to the output JSONL file.')

    parser.add_argument('--model_name_or_path', type=str, default='qwen2_instruct', help='Name of the model to use for generating responses.')
    parser.add_argument('--tensor_parallel_size', type=int, default=10, help='Number of GPUs to use for tensor parallelism.')

    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature for response generation.')
    parser.add_argument('--top_p', type=float, default=1.0, help='Top-p sampling parameter for response generation.')
    parser.add_argument('--max_tokens', type=int, default=32*1024, help='Maximum number of tokens in the generated response.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--N', type=int, default=1, help='Number of responses to generate per prompt.')

    parser.add_argument('--samples', type=int, default=None, help='Number of samples to process from the dataset.')

    return parser.parse_args()


def save_json_line(data, output_file):
    with open(output_file, 'a', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
        f.write('\n')


# def process_and_save(args, engine, dataset):
#     sampling_params = SamplingParams(
#         temperature=args.temperature,
#         top_p=args.top_p,
#         max_tokens=args.max_tokens,
#     )
#     for item in tqdm(dataset):
#         messages = [[{'role': 'user', 'content': item['prompt']}] * args.N]
#         outputs = engine.chat(messages, sampling_params, use_tqdm=False)
#         responses = [output.outputs[0].text.strip() for output in outputs]
#         save_json_line({
#             'prompt': item['prompt'],
#             'responses': responses,
#             'target': item['target']
#         }, args.save_filename)

def process_and_save(args, engine, dataset):
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    dataset = list(dataset)
    total_samples = len(dataset)
    batch_size = 128 // args.N # NOTE: 128 is the mini-batch size
    for batch_start in tqdm(range(0, total_samples, batch_size)):
        batch = dataset[batch_start:batch_start + batch_size]
        
        batch_messages = []
        for item in batch:
            item_messages = [[{'role': 'user', 'content': item['prompt']}] for _ in range(args.N)]
            batch_messages.extend(item_messages)

        outputs = engine.chat(batch_messages, sampling_params, use_tqdm=False)

        output_idx = 0
        for item in batch:
            end_idx = output_idx + args.N
            item_outputs = outputs[output_idx:end_idx]
            output_idx = end_idx
            
            responses = [output.outputs[0].text.strip() for output in item_outputs]
            save_json_line({
                'prompt': item['prompt'],
                'responses': responses,
                'target': item['target']
            }, args.save_filename)


def main():
    args = parse_args()

    # Load the dataset
    if args.dataset_dir.endswith(('.json', '.jsonl')):
        dataset = load_dataset('json', data_files=args.dataset_dir, split='train')
    else:
        dataset = load_dataset(args.dataset_dir, split='train')
    
    # Ensure required columns are present
    assert 'prompt' in dataset.column_names, "Dataset must contain a 'prompt' column"
    assert 'target' in dataset.column_names, "Dataset must contain a 'target' column"

    # Select a subset of the data if specified
    if args.samples:
        dataset = dataset.select(range(args.samples))

    # Ensure the directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.save_filename)), exist_ok=True)

    # Checkpoint
    if os.path.exists(args.save_filename):
        checkpoint_dataset = load_dataset('json', data_files=args.save_filename, split='train')
        checkpoint_prompts = set(checkpoint_dataset['prompt'])
        dataset = dataset.filter(lambda x: x['prompt'] not in checkpoint_prompts)

    # Init engine
    engine = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        seed=args.seed,
        enable_prefix_caching=True,
        enforce_eager=True,
        trust_remote_code=True,
    )

    # Process the dataset
    process_and_save(args, engine, dataset)


if __name__ == '__main__':
    main()
