import re
import asyncio
import aiohttp
import argparse
import pandas as pd
from tqdm import tqdm
from typing import Dict, List
from tabulate import tabulate
from datasets import load_dataset

from src.utils import load_custom_tokenizer
from src.utils.prompt_templates import CRITIC_PROMPT_TEMPLATE

# Define categories and subset mappings
CATEGORIES = {
    "Chat": ["alpacaeval-easy", 'alpacaeval-length', 'alpacaeval-hard', 'mt-bench-easy', 'mt-bench-med'],
    "Chat Hard": ['mt-bench-hard', 'llmbar-natural', 'llmbar-adver-neighbor', 'llmbar-adver-GPTInst', 'llmbar-adver-GPTOut', 'llmbar-adver-manual'],
    "Safety": ['refusals-dangerous', 'refusals-offensive', 'xstest-should-refuse', 'xstest-should-respond', 'donotanswer'],
    "Reasoning": ['math-prm', 'hep-cpp', 'hep-go', 'hep-java', 'hep-js', 'hep-python', 'hep-rust'],
}

# Example counts and subset mapping for weighted scores
EXAMPLE_COUNTS = {
    "alpacaeval-easy": 100,
    "alpacaeval-length": 95,
    "alpacaeval-hard": 95,
    "mt-bench-easy": 28,
    "mt-bench-med": 40,
    "mt-bench-hard": 37,
    "llmbar-natural": 100,
    "llmbar-adver-neighbor": 134,
    "llmbar-adver-GPTInst": 92,
    "llmbar-adver-GPTOut": 47,
    "llmbar-adver-manual": 46,
    "refusals-dangerous": 100,
    "refusals-offensive": 100,
    "xstest-should-refuse": 154,
    "xstest-should-respond": 250,
    "donotanswer": 136,
    "math-prm": 984,  # actual length 447, upweighting to be equal to code
    "hep-cpp": 164,
    "hep-go": 164,
    "hep-java": 164,
    "hep-js": 164,
    "hep-python": 164,
    "hep-rust": 164,
}

def parse_arguments():
    parser = argparse.ArgumentParser(description="Arguments for the evaluation script.")
    parser.add_argument('--data_set_name', type=str, default='allenai/reward-bench', help="Dataset path or name")
    parser.add_argument('--record_filename', type=str, default='./rewardbench_eval.txt', help="Output file path")
    parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct', help="Model path or name")
    parser.add_argument('--generate_url', type=str, default='http://localhost:8009/generate', help="Generate API URL")
    parser.add_argument('--max_workers', type=int, default=20, help="Maximum concurrent requests")
    return parser.parse_args()


async def fetch_generate(session: aiohttp.ClientSession, prompt: str, max_tokens: int = 8, generate_url: str = "http://localhost:8009/generate"):
    """
    Send a POST request to the generation API and return the generated text.
    """
    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1.0,
        "stream": False
    }
    
    async with session.post(generate_url, headers=headers, json=data) as response:
        response.raise_for_status()
        result = await response.json()
        return result["text"][0]


async def generate_response(session: aiohttp.ClientSession, tokenizer, messages, max_new_tokens=1024, add_reason_prompt=False, add_generation_prompt=False):
    """
    Generate a response based on messages and token limits.
    """
    input_text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_reason_prompt=add_reason_prompt, 
        add_generation_prompt=add_generation_prompt
    )
    response_text = await fetch_generate(session, input_text, max_new_tokens, tokenizer.generate_url)
    # Extract the generated portion
    return response_text[len(input_text):]


async def generate_reason_and_assistant(session: aiohttp.ClientSession, tokenizer, messages, max_reason_tokens=4096, max_generation_tokens=64):
    """
    Generate both the reason and assistant responses.
    """
    reason = await generate_response(
        session,
        tokenizer, 
        messages, 
        add_reason_prompt=True, 
        max_new_tokens=max_reason_tokens
    )
    
    assistant = await generate_response(
        session,
        tokenizer, 
        messages + [{'role': 'reason', 'content': reason}], 
        add_generation_prompt=True, 
        max_new_tokens=max_generation_tokens
    )
    
    return reason, assistant


async def process_item(session: aiohttp.ClientSession, tokenizer, prompt: str):
    """
    Process a single example and extract the decision (A, B, or tie).
    """
    messages = [{'role': 'user', 'content': prompt}]
    reason_response, assistant_response = await generate_reason_and_assistant(
        session, tokenizer, messages, max_reason_tokens=8192, max_generation_tokens=4096
    )
    # Logging the reason and assistant responses for debugging
    if assistant_response.strip() not in ['[[A]]', '[[B]]']:
        print(
            '='*50, reason_response,
            '-'*50, assistant_response,
            sep='\n'
        )

    # Extract decision using regex
    matches = re.findall(r'\[\[(A|B)\]\]', assistant_response.strip())
    if matches:
        return 1.0 if matches[-1] == 'A' else 0.0, reason_response
    return 0.5, reason_response  # Default to tie if no valid match


async def process_example(session: aiohttp.ClientSession, tokenizer, example: Dict[str, str]) -> Dict[str, any]:
    """
    Process an individual example and return its accuracy score.
    """
    prompt1 = CRITIC_PROMPT_TEMPLATE.format(
        question=example['prompt'],
        response_a=example['chosen'],
        response_b=example['rejected']
    )
    prompt2 = CRITIC_PROMPT_TEMPLATE.format(
        question=example['prompt'],
        response_a=example['rejected'],
        response_b=example['chosen']
    )
    # Process both prompts
    correct1, reason1 = await process_item(session, tokenizer, prompt1)
    correct2, reason2 = await process_item(session, tokenizer, prompt2)
    # Average accuracy
    final_correct = (correct1 + (1 - correct2)) / 2.0
    if final_correct < 0.5:
        print(
            '='*50, example['subset'],
            '='*50, example['prompt'],
            '-'*50, example['chosen'],
            '-'*50, example['rejected'],
            '+'*50, reason1,
            '*'*50, reason2,
            sep='\n'
        )
    return {'id': example['id'], 'subset': example['subset'], 'correct': final_correct}


async def process_dataset(session: aiohttp.ClientSession, tokenizer, dataset, max_concurrency: int) -> pd.DataFrame:
    """
    Process the dataset using asynchronous tasks, limiting concurrency.
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def sem_task(example):
        async with semaphore:
            return await process_example(session, tokenizer, example)

    tasks = [asyncio.create_task(sem_task(example)) for example in dataset]
    results = []
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing examples"):
        try:
            result = await f
            results.append(result)
        except Exception as e:
            print(f"Processing failed with error: {e}")
    return pd.DataFrame(results)


def calculate_accuracy(df: pd.DataFrame, categories: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Calculate accuracy per category and return a DataFrame with the results.
    """
    df_acc = []
    for category, subsets in categories.items():
        for subset in subsets:
            df_subset = df[df['subset'] == subset]
            if not df_subset.empty:
                accuracy = df_subset['correct'].mean()
                df_acc.append({
                    'category': category,
                    'subset': subset,
                    'n': len(df_subset),
                    'accuracy': accuracy
                })
    return pd.DataFrame(df_acc)


def calculate_scores_per_section(example_counts: Dict[str, int], subset_mapping: Dict[str, List[str]], metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate weighted accuracy scores for each section.
    """
    section_scores = {}
    for section, tests in subset_mapping.items():
        total_weighted_score = sum(metrics.get(test, 0) * example_counts.get(test, 0) for test in tests)
        total_examples = sum(example_counts.get(test, 0) for test in tests)
        section_scores[section] = round(100 * total_weighted_score / total_examples, 2) if total_examples > 0 else 0.0
    return section_scores


async def main(args):
    # Setup tokenizer
    tokenizer = load_custom_tokenizer(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
        use_reason_template=True # IMPORTANT: Set REASON_CHAT_TEMPLATE in tokenizer
    )
    tokenizer.generate_url = args.generate_url  # Attach generate URL to tokenizer for easy access
    
    # Load dataset
    print(f"Loading dataset: {args.data_set_name}")
    dataset = load_dataset(args.data_set_name, split='filtered', keep_in_memory=True)

    # NOTE: debug
    # dataset = dataset.select(range(32))
    # dataset = dataset.filter(lambda x: x['subset'] in ['donotanswer'])

    # Create an aiohttp session
    async with aiohttp.ClientSession() as session:
        # Process dataset with concurrency limit
        print("Processing dataset...")
        df_results = await process_dataset(session, tokenizer, dataset, max_concurrency=args.max_workers)

        # Calculate accuracy per category
        df_accuracy = calculate_accuracy(df_results, CATEGORIES)

        # Calculate scores per section
        metrics = df_accuracy.set_index('subset')['accuracy'].to_dict()
        section_scores = calculate_scores_per_section(EXAMPLE_COUNTS, CATEGORIES, metrics)

        # Prepare final results
        df_final = pd.DataFrame([{'attribute': 'correct', **section_scores}])

        # Output results
        output = f"{'=' * 50}\n{args.model_name_or_path}\n{'-' * 50}\n"
        output += tabulate(df_accuracy, headers='keys', tablefmt='grid') + '\n'

        scores = []
        for section in ['Chat', 'Chat Hard', 'Safety', 'Reasoning']:
            score = df_final[section].values[0]
            output += f"{section}: {score}\n"
            scores.append(score)
        avg_score = sum(scores) / len(scores)
        output += f"Avg Score: {avg_score:.2f}\n"
        print(output)

        # Save results
        with open(args.record_filename, 'a') as f:
            f.write(output)


if __name__ == "__main__":
    args = parse_arguments()
    asyncio.run(main(args))
