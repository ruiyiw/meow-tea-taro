#!/bin/bash
tw_basic_world_size=$TW_BASIC_WORLD_SIZE
tw_basic_nb_objects=$TW_BASIC_NB_OBJECTS
tw_basic_quest_length=$TW_BASIC_QUEST_LENGTH
tw_base_seed=$TW_BASE_SEED
tw_tasks_count=$TW_TASKS_COUNT
tasks_dir=$TASKS_DIR

for i in $(seq 1 $tw_tasks_count)
do
    tw-make custom \
        --world-size $tw_basic_world_size \
        --nb-objects $tw_basic_nb_objects \
        --quest-length $tw_basic_quest_length \
        --seed $((tw_base_seed + i)) \
        --output ${tasks_dir}/w${tw_basic_world_size}-o${tw_basic_nb_objects}-q${tw_basic_quest_length}_$((tw_base_seed+i)).z8
done