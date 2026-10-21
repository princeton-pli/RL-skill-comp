## How Does RL Post-training Induce Skill Composition? A Case Study Using Countdown

This repository contains the code for our paper: How Does RL Post-training Induce Skill Composition? A Case Study Using Countdown.

**************************** **Updates** ****************************
* 12/01/2025: Our paper is on [arXiv](https://arxiv.org/abs/2512.01775).
* 12/01/2025: Our paper is accepted as a Spotlight poster at the [Efficient Reasoning Workshop](https://openreview.net/forum?id=qqR4hUUfbe) and as a poster at the [MATH-AI Workshop](https://openreview.net/forum?id=5KqwASPclR) at NeurIPS 2025. See you in San Diego!
* 10/21/2026: Our paper is accepted at [NeurIPS 2026](https://openreview.net/forum?id=zb8jA9cOxq). See you in Atlanta!

## Quick Links

- [How Does RL Post-training Induce Skill Composition? A Case Study Using Countdown](#RL-skill-comp)
- [Quick Links](#quick-links)
- [Experiments](#experiments)
  - [Prepare Conda Environment](#prepare-conda-environment)
  - [Prepare Base Models](#prepare-base-models)
  - [Generate Data](#generate-data)
  - [Example Scripts](#example-scripts)
- [Formal Theorem Proving (Lean)](#formal-theorem-proving-lean)
- [Bugs or Questions?](#bugs-or-questions)
- [Citation](#citation)

## Experiments

In the following section, we provide instructions on reproducing the experiments in our paper.

### Prepare Conda Environment
```Shell
git clone https://github.com/volcengine/verl.git
cd verl
git reset --hard 083da9ab130efa2dc284eeb821a3edd6ce570fe3

conda create -n verl python==3.10
conda activate verl
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.51.1 vllm==0.8.5.post1
pip3 install flash-attn==2.7.4.post1 --no-build-isolation
pip3 install -e .
pip install transformers==4.51.1 vllm==0.8.5.post1
pip install flashinfer-python==0.2.2 -i https://flashinfer.ai/whl/cu124/torch2.6/
```

### Prepare Base Models
```Shell
mkdir models
cd models
git clone https://huggingface.co/Qwen/Qwen2.5-1.5B
git clone https://huggingface.co/Qwen/Qwen2.5-3B
git clone https://huggingface.co/Qwen/Qwen2.5-7B
```

For Llama models, use the provided chat template.
```Shell
git clone https://huggingface.co/meta-llama/Llama-3.2-3B
cp -rf tokenizer_config.json models/Llama-3.2-3B/tokenizer_config.json
```

### Generate Data
```Shell
for i in {0..17}
do
python generate_puzzles.py --puzzle_size 3 --pattern_index ${i} --num_data 4000
done

for i in {0..95}
do
python generate_puzzles.py --puzzle_size 4 --pattern_index ${i} --num_data 4000
done

for i in {0..557}
do
python generate_puzzles.py --puzzle_size 5 --pattern_index ${i} --num_data 10
done

for i in {0..3493}
do
python generate_puzzles.py --puzzle_size 6 --pattern_index ${i} --num_data 1
done

python preprocess_balanced.py
```


### Example Scripts
Always set the following three environment variables
```Shell
export PROJECT_DIR
export CHECKPOINT_DIR
export RESULT_DIR
```
If not set, all three base directories will point to the current directory.


```Shell
## for 1.5B model
export MODEL_NAME=Qwen2.5-1.5B
sbatch --array=1-3 -t 10:00:00 scripts/train_grpo.sh ## train with 3 seeds

## for 3B model
export MODEL_NAME=Qwen2.5-3B
sbatch --array=1-3 -t 20:00:00 scripts/train_grpo.sh ## train with 3 seeds

## for 7B model
export MODEL_NAME=Qwen2.5-7B
sbatch ---gres=gpu:h100:8 --array=1-3 -t 30:00:00 scripts/train_grpo.sh ## train with 8 GPUs instead
```

```Shell
export MODEL_NAME=Qwen2.5-1.5B
export EXP_NAME=balanced-grpo-seed1
sbatch --array=1-32 --dependency=afterok:${jobid} scripts/eval.sh ## after training is complete
```

```Shell
export MODEL_NAME=Qwen2.5-1.5B
export EXP_NAME=balanced-grpo-seed1
sbatch --dependency=afterok:${jobid} scripts/analyze.sh ## after evaluation is complete
```

## Formal Theorem Proving (Lean)

The `lean/` folder reproduces Section 4.5 and Appendix B.5.

| File | Description |
|---|---|
| `lean/classify.py` | Extracts the proof body from a response and classifies its top-level binary split as balanced / left-heavy / right-heavy |
| `lean/data/id.jsonl` | 100 in-distribution questions from [Goedel-LM/SFT_dataset_v2](https://huggingface.co/datasets/Goedel-LM/SFT_dataset_v2) (Apache-2.0): 14 left-heavy, 51 balanced, 35 right-heavy |
| `lean/data/ood.jsonl` | 135 out-of-distribution questions from the [LeanDojo Benchmark 4](https://github.com/lean-dojo/LeanDojo) (CC BY 2.0), with statements from [mathlib4](https://github.com/leanprover-community/mathlib4) (Apache-2.0): 46 left-heavy, 43 balanced, 46 right-heavy |
| `lean/generate.py` | Samples proofs from Goedel-Prover-V2-32B with vLLM (k = 32, temperature 0.6, top-p 0.95, 8192 new tokens) |
| `lean/sft_distribution.py` | Distribution in the SFT dataset (first row of Tables 3 and 12) |
| `lean/structure_table.py` | Distribution of the generated proofs (second and third rows of Tables 3 and 12) |

Each question records the structure of its reference proof (`ref_bucket`) and the exact prompt given to the model (`prompt`).

```Shell
## Distribution in the SFT dataset (downloads 10 of the 54 shards)
python lean/sft_distribution.py

## Generate proofs (Goedel-Prover-V2-32B on 4 GPUs)
git clone https://huggingface.co/Goedel-LM/Goedel-Prover-V2-32B models/Goedel-Prover-V2-32B
python lean/generate.py --model models/Goedel-Prover-V2-32B --input lean/data/id.jsonl --output lean/outputs/id.jsonl --tp 4
python lean/generate.py --model models/Goedel-Prover-V2-32B --input lean/data/ood.jsonl --output lean/outputs/ood.jsonl --tp 4

## Distribution in the generated proofs
python lean/structure_table.py lean/outputs/id.jsonl lean/outputs/ood.jsonl
```

## Bugs or Questions?

If you have any questions related to the code or the paper, feel free to email Simon (juhyunp 'at' princeton 'dot' edu) and Simran (skaur 'at' princeton 'dot' edu). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can give more effective help!

## Citation

```
@inproceedings{parkkaur2025RLskillcomposition,
      title={How Does RL Post-training Induce Skill Composition? A Case Study on Countdown}, 
      author={Simon Park and Simran Kaur and Sanjeev Arora},
      year={2026},
      booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
      url={https://openreview.net/forum?id=zb8jA9cOxq}, 
}
```