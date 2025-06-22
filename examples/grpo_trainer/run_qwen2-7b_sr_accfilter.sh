# SR: Scalable RL
set -x

DATA_HOME=/mnt/data/user/tc_agi/jibo/datasets
MODEL_HOME=/mnt/data/user/tc_agi/user/tc_agi/yutianyu
export VLLM_ATTENTION_BACKEND=XFORMERS

export http_proxy=10.97.64.39:18000 && export https_proxy=10.97.64.39:18000

WANDB_PRJ_NAME=scalable_rl
export WANDB_MODE=online # online
export TENSORBOARD_DIR=/data/logs/tensorboard

export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

nnodes=${VERL_N_TRAIN_NODE:-1}
EXP_NAME=qwen2_7b_function_rm_kl0-jeeves_debug_notebook_origin_py
# KL_COEF=0

# export http_proxy= && export https_proxy=

# export PYTHONPATH=/home/jeeves/.local/lib/python3.10/site-packages/:$PYTHONPATH
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_HOME/PRIME-RL-Eurus-2-RL-Data/train/math_train_r1answerformat_space.parquet \
    data.val_files=$DATA_HOME/PRIME-RL-Eurus-2-RL-Data/test/math_validation_r1answerformat_space.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=$MODEL_HOME/Qwen2-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=$KL_COEF \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard','wandb'] \
    trainer.project_name=$WANDB_PRJ_NAME \
    trainer.experiment_name=$EXP_NAME \
    +trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$nnodes \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 \
    +data.filter_accuracy=True \
    +data.filter_truncated=False \
    +data.accuracy_lower_bound=0.1 \
    +data.accuracy_upper_bound=0.9 \
    +data.filter_cache_regenerate=True \
    +wandb_dir=/data/checkpoints/wandb $@


# mkdir /data/checkpoints/wandb
# cp -r wandb/* /data/checkpoints/wandb

    # trainer.logger=['console','wandb'] \
    # trainer.logger=['console'] \
