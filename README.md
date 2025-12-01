## How Does RL Induce Skill Composition? A Case Study Using Countdown

This repository contains the code for our paper: How Does RL Induce Skill Composition? A Case Study Using Countdown.

**************************** **Updates** ****************************
* 12/01/2025: Our paper is accepted as a Spotlight poster at the [Efficient Reasoning Workshop](https://openreview.net/forum?id=qqR4hUUfbe) and as a poster at the [MATH-AI Workshop](https://openreview.net/forum?id=5KqwASPclR) at NeurIPS 2025. See you in San Diego!

## Quick Links

- [How Does RL Induce Skill Composition? A Case Study Using Countdown](#RL-skill-comp)
- [Quick Links](#quick-links)
- [Experiments](#experiments)
  - [Prepare Conda Environment](#prepare-conda-environment)
  - [Prepare Base Models](#prepare-base-models)
  - [Generate Data](#generate-data)
  - [Example Scripts](#example-scripts)
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

for i in {0..4327}
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

## Bugs or Questions?

If you have any questions related to the code or the paper, feel free to email Simon (juhyunp 'at' princeton 'dot' edu) and Simran (skaur 'at' princeton 'dot' edu). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can give more effective help!

## Citation

TODO