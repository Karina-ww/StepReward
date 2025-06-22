set -x

env

CPUS_PER_TASK=80


WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-12345}
export VERL_N_TRAIN_NODE=${1:-1}

export RAY_RUNTIME_ENV_JSON=$(cat ./ray_runtime_env.json)
echo RAY_RUNTIME_ENV_JSON is $RAY_RUNTIME_ENV_JSON

# Start Ray head node
if [ "$RANK" == "0" ]; then
    echo "Starting Ray head node on $MASTER_ADDR"
    RAY_RUNTIME_ENV_JSON=$(cat ./ray_runtime_env.json) ray start --head \
        --port=$MASTER_PORT \
        --num-cpus="${CPUS_PER_TASK}" \
        --num-gpus=8 \
        --block &
    
    # Get the Ray address
    ray_address="ray://${MASTER_ADDR}:${MASTER_PORT}"
    
    # Wait for Ray to start
    sleep 10

    # Start your training script
    bash ./examples/grpo_trainer/run_qwen2-7b_seq_balance.sh
else
    # Start Ray worker nodes
    echo "Starting Ray worker node on $CURRENT_NODE"
    RAY_RUNTIME_ENV_JSON=$(cat ./ray_runtime_env.json) ray start --address="$MASTER_ADDR:$MASTER_PORT" \
        --num-cpus="${CPUS_PER_TASK}" \
        --block
fi

# Wait for all processes to finish
wait

