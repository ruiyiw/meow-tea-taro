# Copyright 2025 Ruiyi Wang, PEARLS Lab, UC San Diego
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def textworld_make_map_fn(split, instances_dir, dataset_id, reward_method):
    """
    Create a mapping function for processing TextWorld dataset examples.
    """
    def process_fn(example, idx):
        data = {
            "data_source": f"textworld_{dataset_id}",
            "prompt": [
                {
                    "role": "user",
                    "content": example["prompt"],
                }
            ],
            "reward_model": {
                "style": "rule", 
                "ground_truth": f"{example['task_prefix']}_{example['instance_id']}"
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "response": example["response"],
                "prompt": example["prompt"],
                "instance_path": instances_dir,
                "instance_file": f"{example['task_prefix']}_{example['instance_id']}",
                "reward_method": reward_method
            },
        }
        return data

    return process_fn
