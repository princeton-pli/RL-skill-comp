"""Sample proofs from Goedel-Prover-V2-32B on the in-distribution (ID) or
out-of-distribution (OOD) question set.

Usage:
    python lean/generate.py --model models/Goedel-Prover-V2-32B \
        --input lean/data/id.jsonl --output lean/outputs/id.jsonl --tp 4
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('-k', type=int, default=32)
    p.add_argument('--temperature', type=float, default=0.6)
    p.add_argument('--top-p', type=float, default=0.95)
    p.add_argument('--max-tokens', type=int, default=8192)
    p.add_argument('--tp', type=int, default=4, help='tensor-parallel size')
    p.add_argument('--gpu-mem-util', type=float, default=0.85)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    from vllm import LLM, SamplingParams

    records = [json.loads(line) for line in args.input.open()]

    llm = LLM(model=args.model, tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_mem_util, dtype='bfloat16',
              trust_remote_code=True, max_model_len=args.max_tokens + 4096)
    tokenizer = llm.get_tokenizer()
    prompts = [tokenizer.apply_chat_template([{'role': 'user', 'content': r['prompt']}],
                                             tokenize=False, add_generation_prompt=True)
               for r in records]
    sampling = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p,
                              max_tokens=args.max_tokens, seed=args.seed)
    outputs = llm.generate(prompts, sampling)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w') as f:
        for r, out in zip(records, outputs):
            r['responses'] = [c.text for c in out.outputs]
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'Wrote {len(records)} questions x {args.k} responses to {args.output}')


if __name__ == '__main__':
    main()
