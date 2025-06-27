set -x

# --- Control WandB Usage ---
# Set USE_WANDB to "false" to disable WandB logging.
USE_WANDB=${USE_WANDB:-"false"}

# Basic Project Settings
WANDB_PRJ_NAME=rlpr
EXP_NAME=reproduce_gemma
MODEL=path_to_base_model
N_GPUS_PER_NODE=8

# Judge Settings
# You can choose one of the following options:

# Rule as Judge (Default)
unset OPENAI_API_KEY
unset OPENAI_API_BASE
export USED_MODEL=${USED_MODEL:-"no_api"}

# # OpenSouce Model as Judge
# export USED_MODEL=Qwen/Qwen2.5-72B-Instruct
# export CLIENT_IP=http://127.0.0.1:8001

# # API-Based Model as Judge
# export OPENAI_API_KEY=your_openai_api_key
# export OPENAI_API_BASE=your_openai_api_base
# export USED_MODEL=gpt-4.1 # or gpt-4o


# Train and Validation Files
TRAIN_FILES=./datasets/train/rlpr_train.parquet
VAL_DIR=${VAL_DIR:-"./datasets/test"}
VAL_FILES=[${VAL_DIR}'/MMLUPro-1000_Avg2.parquet',${VAL_DIR}'/Math-500_Avg2.parquet',${VAL_DIR}'/gpqa_diamond_Avg4.parquet',${VAL_DIR}'/AIME2024_Avg16.parquet',${VAL_DIR}'/WebInstruct-verified-val_Avg2.parquet',${VAL_DIR}'/Minerva_Avg4.parquet',${VAL_DIR}'/TheoremQA_Avg2.parquet']

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 # For Gemma only

# Logging and Checkpointing
export LOGS_PATH=data/logs
export TENSORBOARD_DIR=./tensorboard
mkdir -p "${TENSORBOARD_DIR}"
VAL_SAVE_RESULTS_DIR=data/logs/test_generations_${EXP_NAME}
mkdir -p "${VAL_SAVE_RESULTS_DIR}"
LOCAL_DIR=data/checkpoints/${EXP_NAME}
mkdir -p "${LOCAL_DIR}"

# --- Conditional WandB Setup ---
TRAINER_LOGGER_CONFIG="['console']" # Default logger
declare -a WANDB_PARAMETERS # Array to hold WandB specific parameters

if [ "$USE_WANDB" = "true" ]; then
    echo "WandB logging ENABLED. Make sure you have logged in."
    export WANDB_MODE=online
    export WANDB_DIR_PATH=./wandb # Define path for WandB data
    mkdir -p "${WANDB_DIR_PATH}"

    export WANDB_DIR=${WANDB_DIR_PATH}

    TRAINER_LOGGER_CONFIG="['console','wandb']"
    WANDB_PARAMETERS=(
        "trainer.project_name=$WANDB_PRJ_NAME"
        "trainer.val_generations_to_log_to_wandb=10"
        "+trainer.train_generations_to_log_to_wandb=1"
        "+trainer.train_generations_to_log_to_wandb_2=50"
        "+wandb_dir=${WANDB_DIR}" # Use the exported WANDB_DIR which is ./wandb
    )
else
    echo "WandB logging DISABLED."
    # WANDB_PARAMETERS array remains empty
    # TRAINER_LOGGER_CONFIG remains ['console']
    # Unset WANDB_DIR if you don't want it in the environment when WandB is disabled
    unset WANDB_DIR
fi


export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

nnodes=${VERL_N_TRAIN_NODE:-1}
KL_COEF=0


# Main Training Command and Configuration
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILES \
    data.val_files=$VAL_FILES \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    +data.filter_accuracy=True \
    +data.filter_truncated=False \
    +data.filter_ema_ratio=0.99 \
    +data.filter_ema_start_step=6 \
    +data.filter_start_step=11 \
    +data.filter_mode=EMA \
    +data.filter_target=final_reward_std \
    +data.accuracy_lower_bound=0 \
    +data.std_filter_beta=1.0 \
    +data.accuracy_upper_bound=1000000 \
    +data.filter_cache_regenerate=True \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16000 \
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
    +actor_rollout_ref.rollout.val_top_p=0.95 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +actor_rollout_ref.actor.clip_ratio_low=0.2 \
    +actor_rollout_ref.actor.clip_ratio_high=0.27 \
    algorithm.kl_ctrl.kl_coef=$KL_COEF \
    trainer.critic_warmup=0 \
    trainer.logger=${TRAINER_LOGGER_CONFIG} \
    trainer.experiment_name=$EXP_NAME \
    "${WANDB_PARAMETERS[@]}" \
    +trainer.val_before_train=True \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=$nnodes \
    trainer.save_freq=40 \
    trainer.test_freq=40 \
    +trainer.test_decoding_strategy=sampling \
    trainer.total_epochs=100 \
    +trainer.val_save_results_dir=${VAL_SAVE_RESULTS_DIR} \
    trainer.default_local_dir=${LOCAL_DIR} \
    reward_model.reward_manager=prob \
    +reward_model.reward_manager_shaping_function_name=threshold_0 \
    +reward_model.compute_score_name=mean_exp_log_softmax \
    +reward_model.repetition_penalty=True \
    +reward_model.val_reward_manager=naive \
    +reward_model.format_mode=R1_nothink \
    "$@"