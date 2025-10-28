#!/bin/bash
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH -t 03:00:00

conda activate verl

project_dir=${PROJECT_DIR:-$PWD}
result_dir=${RESULT_DIR:-${project_dir}}

model_name=${MODEL_NAME:-Qwen2.5-1.5B}
exp_name=${EXP_NAME:-balanced-grpo-seed1}
eval_dataset=${EVAL_DATASET:-balanced}
extra_args=${EXTRA_ARGS:-""}

for CURRENT_INDEX in $(seq 50 50 1600)
do
    echo "--> Starting sub-task: Model=${model_name}, Exp_Name=${exp_name}, Index=${CURRENT_INDEX}"

    python analyze.py \
        --result_dir ${result_dir} \
        --base_model ${model_name} \
        --exp_name ${exp_name} \
        --eval_dataset ${eval_dataset} \
        --start_index ${CURRENT_INDEX} \
        --end_index ${CURRENT_INDEX} \
        --overwrite ${extra_args} &
done
wait

chmod -R 770 results/${model_name}/${exp_name}
