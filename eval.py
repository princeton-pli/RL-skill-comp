from transformers import AutoTokenizer
from datasets import load_dataset
from vllm import LLM, SamplingParams
import os
import json
import torch
from utils import *
import argparse

def main(args):
    dataset_id = f"data/{args.eval_dataset}/test.parquet"

    checkpoint_path = args.checkpoint_path
    if args.result_path is None:
        result_dir = checkpoint_path.replace("checkpoints", "results")
    else:
        result_dir = args.result_path
    result_path = os.path.join(result_dir, "{}_temp{:.1f}_n{}_max{}.json".format(args.eval_dataset, args.temperature, args.n, args.max_tokens))

    print("Saving results to {}".format(result_path), flush=True)

    os.makedirs(result_dir, exist_ok=True)

    dataset = load_dataset("parquet", data_files=dataset_id, split="train")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    # gemerate r1 prompt with a prefix for the model to already start with the thinking process
    def generate_r1_prompt(example):
        prompt = example["prompt"]
        return {"prompt": tokenizer.apply_chat_template(prompt, tokenize=False, continue_final_message=True)}

    # convert our dataset to the r1 prompt
    dataset = dataset.map(lambda x: generate_r1_prompt(x))
    prompts = dataset["prompt"]

    llm = LLM(checkpoint_path, tensor_parallel_size=torch.cuda.device_count(), dtype="bfloat16", trust_remote_code=True)
    sampling_params = SamplingParams(
        temperature=args.temperature, 
        top_p=1, 
        max_tokens=args.max_tokens, 
        n=args.n,
    )

    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)
    outputs = [{"outputs": [output.outputs[i].text for i in range(len(output.outputs))], 
                "target": dataset[idx]["target"], 
                "nums": dataset[idx]["nums"],
                "canonical_pattern_index": dataset[idx]["canonical_pattern_index"],
                "canonical_pattern": dataset[idx]["canonical_pattern"]} for idx, output in enumerate(outputs)]

    with open(result_path, "w") as f:
        json.dump(outputs, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/Qwen2.5-1.5B/balanced-grpo-seed1/global_step_50")
    parser.add_argument("--result_path", default=None)
    parser.add_argument("--eval_dataset", type=str, default="balanced")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--max_tokens", type=int, default=1024)
    args = parser.parse_args()

    main(args)