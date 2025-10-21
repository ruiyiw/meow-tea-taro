set -x
export HYDRA_FULL_ERROR=1

# DATA/TASK CONFIG
env_name="alfworld"
task_prefix="text_based"
hf_data_repo="PEARLS-Lab/meow-tea-taro-dataset"
hf_train_data_dir="$env_name/$task_prefix/multiturn_sft_data/100_data"
local_train_data_dir="local/${hf_train_data_dir}"
local_parquet_dir="local/train_parquet"

# MODEL AND TRAINING CONFIG
base_model=Qwen/Qwen2.5-7B-Instruct
train_batch_size=8
micro_batch_size_per_gpu=1
max_length=2048
nproc_per_node=8
save_freq=40 # per steps
test_freq=5 # per steps
total_epochs=1

# PROJECT CONFIG
project_name="" # TODO (optional). WandB project name.
experiment_name="" # TODO (optional). WandB experiment name.
save_path=checkpoints # The local path to save checkpoints.
save_hf_repo_id="your-hf-repo-id" # TODO (optional). HF repo id to save the trained model. If empty, do not save.


# Step 1: Process RL data
echo "Processing multiturn SFT data for tasks ${env_name}-${task_prefix}"

python3 -m meow_tea_train.agentic_utils.data_process.sft_data_processor \
    --hf_data_repo $hf_data_repo \
    --hf_train_data_dir $hf_train_data_dir \
    --local_train_data_dir $local_train_data_dir \
    --local_parquet_dir $local_parquet_dir \


torchrun --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$local_parquet_dir/train.parquet \
    data.val_files=$local_parquet_dir/validation.parquet \
    data.train_batch_size=$train_batch_size \
    data.micro_batch_size_per_gpu=$micro_batch_size_per_gpu \
    +data.chat_based.enable=true \
    +data.chat_based.messages_key=messages \
    data.max_length=$max_length \
    model.partial_pretrain=$base_model \
    trainer.default_local_dir=$save_path \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    +trainer.save_hf_repo_id=$save_hf_repo_id \
    trainer.total_epochs=$total_epochs $@ \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=false