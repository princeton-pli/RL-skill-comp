import argparse
import json
import os
from tqdm import tqdm

from utils import extract_patterns, get_size
from grader_utils import compute_score


def load_data(idx, annotated=True):
    filename = ANNOTATED_RESULT_FILENAME if annotated else RESULT_FILENAME
    result_file = f"{RESULT_DIR}/global_step_{idx}/{filename}"
    with open(result_file) as f:
        data = json.load(f)
    return data


def extract_unique_patterns_by_n(all_patterns, active_ns: set[int]):
    out = {n: set() for n in active_ns}
    for curr_patterns in all_patterns:      # per-question
        for patterns in curr_patterns:      # per-output
            for p in patterns:              # per-pattern
                n = get_size(p)
                if n in out:
                    out[n].add(p)
    return out


def get_scores_and_patterns(data):
    all_scores = []      # scores across all outputs across all questions
    all_expressions = [] # list of expressions (before canonicalization) across all outputs across all questions
    all_patterns = []    # list of patterns across all outputs across all questions

    for entry in tqdm(data):
        target = entry["target"]
        nums = entry["nums"]
        outputs = entry["outputs"]
        curr_scores = []      # scores across all outputs
        curr_expressions = [] # list of expressions across all outputs
        curr_patterns = []    # list of patterns across all outputs

        for output in outputs:
            try:
                score = compute_score(None, output, target, {"numbers": nums}, verbose=False)
                curr_scores.append(score == 1.0)
            except Exception as e:
                curr_scores.append(False)
                if args.debug:
                    print(f"Skipping malformed output: {output} - {e}", flush=True)

            try:
                expressions, patterns, _ = extract_patterns(output, nums)
                curr_expressions.append(expressions)
                curr_patterns.append(patterns)
            except Exception as e:
                curr_expressions.append([])
                curr_patterns.append([])
                if args.debug:
                    print(f"Skipping malformed output: {output} - {e}", flush=True)

        all_scores.append(curr_scores)
        all_expressions.append(curr_expressions)
        all_patterns.append(curr_patterns)

    return all_scores, all_expressions, all_patterns

def get_annotated_scores_and_patterns(data):    
    all_scores = [entry["scores"] for entry in data]
    all_expressions = [entry["expressions"] for entry in data]
    all_patterns = [entry["patterns"] for entry in data]

    return all_scores, all_expressions, all_patterns

def save_annotated_data(data, all_scores, all_expressions, all_patterns, idx):
    annotated_data = []
    for entry, scores, expressions, patterns in zip(data, all_scores, all_expressions, all_patterns):
        annotated_entry = {
            "target": entry["target"],
            "nums": entry["nums"],
            "outputs": entry["outputs"],
            "scores": scores,
            "expressions": expressions,
            "patterns": patterns
        }
        annotated_data.append(annotated_entry)

    filename = f"{RESULT_DIR}/global_step_{idx}/{ANNOTATED_RESULT_FILENAME}"
    with open(filename, 'w') as f:
        json.dump(annotated_data, f, indent=2)

    return

def get_all_scores_and_patterns():
    all_scores = []      # scores across all outputs across all questions across all checkpoints
    all_expressions = [] # list of expressions (before canonicalization) across all outputs across all questions across all checkpoints
    all_patterns = []    # list of patterns across all outputs across all questions across all checkpoints

    for index in tqdm(CHECKPOINTS):
        filename = f"{RESULT_DIR}/global_step_{index}/{ANNOTATED_RESULT_FILENAME}"
        if os.path.exists(filename) and not args.overwrite:
            data = load_data(index, annotated=True)
            curr_scores, curr_expressions, curr_patterns, ns = get_annotated_scores_and_patterns(data)
        else:
            data = load_data(index, annotated=False)
            curr_scores, curr_expressions, curr_patterns, ns = get_scores_and_patterns(data)
            save_annotated_data(data, curr_scores, curr_expressions, curr_patterns, index)

        all_scores.append(curr_scores)
        all_expressions.append(curr_expressions)
        all_patterns.append(curr_patterns)

    return all_scores, all_expressions, all_patterns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", default=None)
    parser.add_argument("--base_model", type=str, default="Qwen2.5-1.5B")
    parser.add_argument("--exp_name", type=str, default="balanced-grpo-seed1")
    parser.add_argument("--eval_dataset", type=str, default="balanced")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--start_index", type=int, default=50)
    parser.add_argument("--end_index", type=int, default=1600)
    parser.add_argument("--index_increment", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    RESULT_BASE_DIR = "results" if args.result_dir is None else os.path.join(args.result_dir, "results")
    RESULT_DIR=f'{RESULT_BASE_DIR}/{args.base_model}/{args.exp_name}'

    RESULT_FILENAME = f"{args.eval_dataset}_temp{args.temperature}_n{args.n}_max{args.max_tokens}.json"
    ANNOTATED_RESULT_FILENAME = f"{args.eval_dataset}_temp{args.temperature}_n{args.n}_max{args.max_tokens}_annotated.json"

    CHECKPOINTS = []
    for IDX in range(args.start_index, args.end_index + args.index_increment - 1, args.index_increment):
        result_file = f"{RESULT_DIR}/global_step_{IDX}/{RESULT_FILENAME}"
        if os.path.exists(result_file):
            CHECKPOINTS.append(IDX)

    all_scores, all_expressions, all_patterns = get_all_scores_and_patterns()
