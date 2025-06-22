set -x

DATA_HOME=/mnt/data/user/tc_agi/jibo/datasets
MODEL_HOME=/mnt/data/user/tc_agi/jibo/models
export VLLM_ATTENTION_BACKEND=XFORMERS

export http_proxy=10.97.64.39:18000 && export https_proxy=10.97.64.39:18000

WANDB_PRJ_NAME=scalable_rl
export WANDB_MODE=online # online
export TENSORBOARD_DIR=/data/logs/tensorboard

export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

nnodes=${VERL_N_TRAIN_NODE:-1}
EXP_NAME=qwen2_7b_function_rm_kl0-jeeves_debug_notebook_origin_py
KL_COEF=0


torchrun -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.prompt_key=question \
    data.response_key=answer \
    data.micro_batch_size_per_gpu=8 \
    model.partial_pretrain=$MODEL_HOME/Qwen2.5-7B-Instruct \
    trainer.default_hdfs_dir=hdfs://user/verl/experiments/gsm8k/deepseek-coder-6.7b-instruct/ \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-deepseek-coder-6.7b-instruct \
    trainer.total_epochs=4 \
    trainer.logger=['console','wandb']