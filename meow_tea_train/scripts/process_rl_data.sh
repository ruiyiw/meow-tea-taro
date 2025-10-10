#!/bin/bash
set -euo pipefail

# Configuration variables
env_name="textworld"
task_prefix="w2-o3-q4"
task_id_start=50001
task_id_end=55000
hf_data_repo="Pamela153/textworld_w2-o3-q4"
hf_instances_dir="games"
hf_train_data_dir="multiturn_ppo_data/5000_data"
local_instances_dir="local/games"
local_train_data_dir="local/multiturn_ppo_data/5000_data"
local_parquet_dir="local/train_parquet"
reward_method="single"

echo "Processing multiturn RL data for tasks ${env_name}-${task_prefix} ${task_id_start}-${task_id_end}"

# Process everything with all configuration variables
python3 -m scripts_tmp.process_utils.rl_data_processor \
    --env_name "$env_name" \
    --task_prefix "$task_prefix" \
    --task_id_range "$task_id_start" "$task_id_end" \
    --hf_data_repo "$hf_data_repo" \
    --hf_instances_dir "$hf_instances_dir" \
    --hf_train_data_dir "$hf_train_data_dir" \
    --local_instances_dir "$local_instances_dir" \
    --local_train_data_dir "$local_train_data_dir" \
    --local_parquet_dir "$local_parquet_dir" \
    --reward_method "$reward_method"

echo "Processing complete!"