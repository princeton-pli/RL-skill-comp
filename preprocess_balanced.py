## adapted from https://github.com/Jiayi-Pan/TinyZero/blob/main/examples/data_preprocess/countdown.py

"""
Preprocess dataset for countdown task - given a target number and N numbers, generate equations to reach target
"""

import json
import os
from datasets import Dataset
import argparse

def make_prefix(dp, template_type):
    target = dp['target']
    numbers = dp['nums']
    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.
Assistant: Let me solve this step by step.
<think>"""
    elif template_type == 'qwen-instruct':
        """This works for Qwen Instruct Models"""
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\n Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data')
    parser.add_argument('--template_type', type=str, default='base')

    args = parser.parse_args()

    data_source = 'countdown'

    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example, template_type=args.template_type)
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": example['target']
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    "numbers": example['nums'],
                    "puzzle_size": example['puzzle_size'],
                    "canonical_pattern_index": example['canonical_pattern_index'],
                    "canonical_pattern": example['canonical_pattern']
                }
            }
            return data
        return process_fn

    # Generate train/test data for n = 3, 4
    train_dataset = []
    test_dataset = []
    for i in range(18):
        with open(f"data/countdown_3_pattern_{i}.json") as f:
            data = json.load(f)
        train_dataset += data[10:]
        test_dataset += data[:10]

    for i in range(96):
        with open(f"data/countdown_4_pattern_{i}.json") as f:
            data = json.load(f)
        train_dataset += data[10:]
        test_dataset += data[:10]

    train_dataset = Dataset.from_list(train_dataset)
    test_dataset = Dataset.from_list(test_dataset)
    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = os.path.join(args.local_dir, "balanced")

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    # Generate test data for n = 5
    test_dataset = []
    for i in range(558):
        with open(f"data/countdown_5_pattern_{i}.json") as f:
            data = json.load(f)
        test_dataset += data[:10]

    test_dataset = Dataset.from_list(test_dataset)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    local_dir = os.path.join(args.local_dir, "balanced5")
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    # Generate test data for n = 6
    test_dataset = []
    for i in range(4328):
        with open(f"data/countdown_6_pattern_{i}.json") as f:
            data = json.load(f)
        test_dataset += data[:1]

    test_dataset = Dataset.from_list(test_dataset)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    local_dir = os.path.join(args.local_dir, "balanced6")
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))