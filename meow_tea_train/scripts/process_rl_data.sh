#!/bin/bash
set -euo pipefail

# Configuration variables
env_name="textworld"
task_prefix="w4-o6-q8"
instance_id_start=50001
instance_id_end=55000
hf_data_repo="PEARLS-Lab/meow-tea-taro-dataset"
hf_instances_dir="textworld/w4-o6-q8/instances"
hf_train_data_dir="multiturn_ppo_data/5000_data"
local_instances_dir="local/$hf_instances_dir"
local_train_data_dir="local/multiturn_ppo_data/5000_data"
local_parquet_dir="local/train_parquet"
reward_method="single"

echo "Processing multiturn RL data for tasks ${env_name}-${task_prefix} ${instance_id_start}-${instance_id_end}"

# Process everything with all configuration variables
python3 -m meow_tea_train.agentic_utils.data_process.rl_data_processor \
    --instance_id_range "$instance_id_start" "$instance_id_end" \
    --task_prefix "$task_prefix" \
    --env_name "$env_name" \
    --hf_data_repo "$hf_data_repo" \
    --hf_instances_dir "$hf_instances_dir" \
    --hf_train_data_dir "$hf_train_data_dir" \
    --local_instances_dir "$local_instances_dir" \
    --local_train_data_dir "$local_train_data_dir" \
    --local_parquet_dir "$local_parquet_dir" \
    --reward_method "$reward_method"

echo "Processing complete!"