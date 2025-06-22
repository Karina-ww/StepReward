# SR: Scalable RL
set -x

script_basename=$(basename "$0")
EXP_NAME="${script_basename%.sh}"

TRAIN_FILES=/user/jibo/datasets/PRIME-RL-Eurus-2-RL-Data/train/math_train-03-19-systemr1-n600.parquet
VAL_FILES=['/user/jibo/datasets/math/test-OC-03-20.parquet','/user/jibo/datasets/math/test-UserR1-03-20.parquet','/user/jibo/datasets/gpqa_diamond/test-OC-03-20.parquet','/user/jibo/datasets/gpqa_diamond/test-UserR1-03-20.parquet','/user/jibo/datasets/mmlu_pro/test-1000-OC-03-20.parquet','/user/jibo/datasets/mmlu_pro/test-1000-UserR1-03-20.parquet','/user/jibo/datasets/hellaswag/val-1000-OC-03-20.parquet','/user/jibo/datasets/hellaswag/val-1000-UserR1-03-20.parquet','/user/jibo/datasets/PRIME-RL-Eurus-2-RL-Data/math_validation-Original-04-03.parquet','/user/jibo/datasets/PRIME-RL-Eurus-2-RL-Data/math_validation-UserR1-04-03.parquet']

MODEL=/user/jibo/models/Qwen2.5-7B

MAX_RESPONSE_LENGTH=3072
MAX_PROMPT_LENGTH=1024

export TENSORBOARD_DIR=/user/jibo/gitrepo/verl/tensorboard
mkdir ${TENSORBOARD_DIR} -p
VAL_SAVE_RESULTS_DIR=/data/logs/test_generations
mkdir ${VAL_SAVE_RESULTS_DIR} -p
export WANDB_DIR=/user/jibo/gitrepo/verl # need to create /user/jibo/gitrepo/verl/wandb in advance
mkdir ${WANDB_DIR} -p

export VLLM_ATTENTION_BACKEND=XFORMERS


WANDB_PRJ_NAME=scalable_rl
export WANDB_MODE=online # online

export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

nnodes=${VERL_N_TRAIN_NODE:-1}
KL_COEF=0

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILES \
    data.val_files=$VAL_FILES \
    data.train_batch_size=256 \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_COEF \
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
    +trainer.val_before_train=True \
    +trainer.val_only=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$nnodes \
    trainer.save_freq=100 \
    trainer.test_freq=20 \
    trainer.total_epochs=1 \
    +trainer.val_save_results_dir=${VAL_SAVE_RESULTS_DIR} \
    trainer.val_generations_to_log_to_wandb=10 \
    +trainer.train_generations_to_log_to_wandb=1 \
    +trainer.train_generations_to_log_to_wandb_2=50 \
    reward_model.reward_manager=naive \
    +reward_model.val_reward_manager=naive \
    +wandb_dir=${WANDB_DIR} $@

