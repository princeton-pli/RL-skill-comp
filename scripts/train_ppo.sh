#!/bin/bash
#SBATCH --gres=gpu:h100:4
#SBATCH -N 1 -n 1
#SBATCH --mem-per-gpu=96G 
#SBATCH --cpus-per-gpu 8
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH -t 24:00:00
#SBATCH --array 1-1

conda activate verl

export WANDB_MODE="offline"

project_dir=${PROJECT_DIR:-$PWD}
cd ${project_dir}/verl
checkpoint_dir=${CHECKPOINT_DIR:-${project_dir}}

model_name=${MODEL_NAME:-Qwen2.5-3B}
data_source=${DATA_SOURCE:-balanced}
exp_name=${EXP_NAME:-${data_source}-ppo-seed${SLURM_ARRAY_TASK_ID}}

max_length=${MAX_LENGTH:-1024}
train_path=../data/${data_source}/train.parquet
test_path=../data/${data_source}/test.parquet

output_dir=${CHECKPOINT_PATH:-${checkpoint_dir}/checkpoints/${model_name}}

N_GPUS="$(( $(echo $SLURM_JOB_GPUS| grep -o , | wc -l) + 1 ))"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files="$train_path" \
    data.val_files="$test_path" \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=${max_length} \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    +data.seed=${SLURM_ARRAY_TASK_ID} \
    actor_rollout_ref.model.path=${project_dir}/models/${model_name} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    critic.optim.lr=1e-5 \
    critic.model.path=${project_dir}/models/${model_name} \
    critic.ppo_micro_batch_size_per_gpu=8 \
    trainer.val_before_train=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=1 \
    trainer.default_local_dir=${output_dir}/${exp_name} \
    trainer.rollout_data_dir=${output_dir}/${exp_name} \
    trainer.project_name=countdown \
    trainer.experiment_name=${model_name}-${exp_name} \
    trainer.balance_batch=False \
    custom_reward_function.path=../grader_utils.py \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    critic.strategy=fsdp2 \
    reward_model.strategy=fsdp2 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True
    
chmod -R 770 ${output_dir}/${exp_name}