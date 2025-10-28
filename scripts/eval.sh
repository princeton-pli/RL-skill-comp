#!/bin/bash
#SBATCH --gres=gpu:h100:1
#SBATCH -N 1 -n 1
#SBATCH --mem-per-gpu=96G 
#SBATCH --cpus-per-gpu 8
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH -t 01:30:00
#SBATCH --array 1-32

conda activate verl

project_dir=${PROJECT_DIR:-$PWD}
checkpoint_dir=${CHECKPOINT_DIR:-${project_dir}}
result_dir=${RESULT_DIR:-${project_dir}}

model_name=${MODEL_NAME:-Qwen2.5-1.5B}
exp_name=${EXP_NAME:-balanced-grpo-seed1}
eval_dataset=${EVAL_DATASET:-balanced}
extra_args=${EXTRA_ARGS:-""}

checkpoint_path=${checkpoint_dir}/checkpoints/${model_name}/${exp_name}/global_step_$((50 * SLURM_ARRAY_TASK_ID))
result_path=${result_dir}/results/${model_name}/${exp_name}/global_step_$((50 * SLURM_ARRAY_TASK_ID))

if python -m verl.model_merger merge --backend fsdp --local_dir ${checkpoint_path}/actor/ --target_dir ${checkpoint_path}; then
    rm -rf ${checkpoint_path}/actor/
    rm -rf ${checkpoint_path}/critic/
    rm -rf ${checkpoint_path}/data.pt
elif python -m verl.model_merger merge --backend fsdp --local_dir ${checkpoint_path} --target_dir ${checkpoint_path}; then
    rm -rf ${checkpoint_path}/*.pt
fi

python eval.py --checkpoint_path ${checkpoint_path} --result_path ${result_path} --eval_dataset ${eval_dataset} ${extra_args}

chmod -R 770 ${checkpoint_dir}/checkpoints/${model_name}/${exp_name}
chmod -R 770 ${result_dir}/results/${model_name}/${exp_name}