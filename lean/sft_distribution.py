"""Structure distribution of the Goedel-Prover-V2 SFT dataset.

We scan the first 10 of the 54 parquet shards of Goedel-LM/SFT_dataset_v2
(323K of 1.74M samples) and classify the final assistant message of every
sample: the proof for whole-proof samples, the corrected proof for revision
samples.

Usage:
    python lean/sft_distribution.py [--n-shards 10]
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from classify import THRESHOLDS, distribution, format_distribution

REPO = 'Goedel-LM/SFT_dataset_v2'
N_TOTAL_SHARDS = 54


def responses(n_shards):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    for i in range(n_shards):
        path = hf_hub_download(REPO, f'data/train-{i:05d}-of-{N_TOTAL_SHARDS:05d}.parquet',
                               repo_type='dataset')
        for messages in pq.read_table(path, columns=['messages']).column('messages').to_pylist():
            yield messages[-1]['content']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n-shards', type=int, default=10)
    args = p.parse_args()
    os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '180')

    texts = list(responses(args.n_shards))
    print(f'{REPO}: {len(texts)} samples from {args.n_shards} shards')
    for name, (left_thr, right_thr) in THRESHOLDS.items():
        print(f'  {name}: {format_distribution(distribution(texts, left_thr, right_thr))}')


if __name__ == '__main__':
    main()
