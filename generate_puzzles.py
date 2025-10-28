import argparse
import json
import numpy as np
import os
from collections import Counter

def plug_in_nums(pattern, nums):
    for i in range(len(nums)):
        pattern = pattern.replace(chr(ord('A') + i), str(nums[i]))
    return pattern

def is_integer(result):
    if isinstance(result, int):
        return True
    if isinstance(result, float) and result.is_integer():
        return True
    return False


def generate(puzzle_size, 
             pattern_index, 
             seed, 
             num_data):

    np.random.seed(seed)

    with open("annotated_expressions.json") as f:
        data = json.load(f)

    data = data[str(puzzle_size)]
    patterns = []
    unique_patterns = set()
    for d in data:
        pattern = d["canonical"]
        if pattern not in unique_patterns:
            patterns.append(pattern)
            unique_patterns.add(pattern)

    if pattern_index >= len(patterns):
        raise ValueError("Pattern index out of range")
    
    pattern = list(patterns)[pattern_index]
    print(f"Pattern: {pattern}")

    results = []
    counter = 0

    already_seen = set()

    while len(results) < num_data and counter < 10000000:
        counter += 1
        nums = np.random.randint(1, 100, size=puzzle_size)

        if tuple(sorted(nums)) in already_seen:
            continue

        expression = plug_in_nums(pattern, nums)
        try:
            result = eval(expression)
            if is_integer(result) and 1 <= result < 100:
                results.append({
                    "nums": nums.tolist(),
                    "target": int(result),
                    "puzzle_size": puzzle_size,
                    "canonical_pattern_index": pattern_index,
                    "canonical_pattern": pattern
                })
                already_seen.add(tuple(sorted(nums)))
        except:
            continue

    print(len(results))
    print(counter)

    os.makedirs("data", exist_ok=True)

    with open(f"data/countdown_{puzzle_size}_pattern_{pattern_index}.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--puzzle_size", type=int, default=3)
    parser.add_argument("--pattern_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_data", type=int, default=10)
    args = parser.parse_args()

    results = generate(args.puzzle_size, 
                       args.pattern_index, 
                       args.seed, 
                       args.num_data)