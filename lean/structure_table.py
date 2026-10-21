"""Structure distribution of the RL model's proofs.

Every sampled response counts, whether or not it passes verification. Each
response is parsed and classified; responses without a binary-split proof
(`other`, including unparseable ones) are excluded from the percentages.

Usage:
    python lean/structure_table.py lean/outputs/id.jsonl lean/outputs/ood.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from classify import THRESHOLDS, distribution, format_distribution


def main():
    p = argparse.ArgumentParser()
    p.add_argument('generations', type=Path, nargs='+',
                   help='output files of lean/generate.py')
    args = p.parse_args()

    for path in args.generations:
        records = [json.loads(line) for line in path.open()]
        texts = [resp for r in records for resp in r['responses']]
        print(f'{path}: {len(records)} questions, {len(texts)} responses')
        for name, (left_thr, right_thr) in THRESHOLDS.items():
            print(f'  {name}: {format_distribution(distribution(texts, left_thr, right_thr))}')


if __name__ == '__main__':
    main()
