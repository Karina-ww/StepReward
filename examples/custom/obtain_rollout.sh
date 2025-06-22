# bash examples/grpo_trainer/run_qwen2-7b_sr.sh 
# SR: Scalable RL
set -x


# Extract just the filename (e.g., "test.sh")
script_basename=$(basename "$0")
# Remove the ".sh" extension (e.g., "test")
EXP_NAME="${script_basename%.sh}"

# TRAIN_FILES=/user/jibo/datasets/BytedTsinghua-SIA-DAPO-Math-17k/dapo-math-17k-SysR1-04-14.parquet
# TRAIN_FILES=/user/jibo/datasets/scalable_rl/tuluonly-nonverifiable-03-24-SysR1-v2.parquet
TRAIN_FILES=/user/jibo/datasets/TIGER-Lab-WebInstruct-verified/train-SysR1-04-20-v2.parquet
VAL_FILES=['/user/jibo/datasets/scalable_rl/test-Math500+GPQA+MMLUPro+HellaSwag+Eurus-OC+SysR1-04-14.parquet','/user/jibo/datasets/AIME2024/AIME2024-OC-03-28.parquet','/user/jibo/datasets/AIME2024/AIME2024-SysR1-03-28.parquet','/user/jibo/datasets/math-ai-amc23/AMC2023-OC-04-13.parquet','/user/jibo/datasets/math-ai-amc23/AMC2023-SysR1-04-13.parquet']

MODEL=/user/jibo/models/DeepSeek-R1-Distill-Qwen-7B

MAX_RESPONSE_LENGTH=3072

# export TENSORBOARD_DIR=/user/jibo/gitrepo/verl/tensorboard
export TENSORBOARD_DIR=/data/logs/tensorboard
mkdir ${TENSORBOARD_DIR} -p
# VAL_SAVE_RESULTS_DIR=/user/jibo/gitrepo/verl/val_results
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
    data.max_prompt_length=512 \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
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
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=$KL_COEF \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard','wandb'] \
    trainer.project_name=$WANDB_PRJ_NAME \
    trainer.experiment_name=$EXP_NAME \
    +trainer.val_before_train=False \
    +trainer.obtain_rollout_only=True \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=$nnodes \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=1 \
    +trainer.val_save_results_dir=${VAL_SAVE_RESULTS_DIR} \
    trainer.val_generations_to_log_to_wandb=10 \
    +trainer.train_generations_to_log_to_wandb=1 \
    +trainer.train_generations_to_log_to_wandb_2=50 \
    reward_model.reward_manager=naive \
    +reward_model.repetition_penalty=True \
    +reward_model.val_reward_manager=naive \
    +wandb_dir=/data/checkpoints/wandb $@

