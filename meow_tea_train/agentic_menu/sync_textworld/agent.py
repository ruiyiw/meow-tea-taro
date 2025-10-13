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


from typing import List, Tuple, Dict
import os
import torch
import numpy as np
from tensordict import TensorDict

from vllm.inputs.data import TokensPrompt
from verl import DataProto
from .env import TextWorldEnvBase
from ..base.agent import BaseAgent


class TextWorldAgent(BaseAgent) :
    """
    An instance of textworld agent that interacts with the textworld env.
    """
    def __init__(
        self,
        env,
        prompts: DataProto,
        inference_engine,
        sampling_params,
        tokenizer,
        max_iter,
        n_traj,
        max_prompt_len,
        max_response_len
    ):  
        self.env = env
        self.device = prompts.batch["input_ids"].device
        self.inference_engine = inference_engine
        self.sampling_params = sampling_params
        self.tokenizer = tokenizer
        self.max_iter = max_iter
        self.n_traj = n_traj
        self.max_prompt_len = max_prompt_len
        self.max_response_len = max_response_len
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.sep_token_id = tokenizer.eos_token_id
        self.sep_token = tokenizer.convert_ids_to_tokens(self.sep_token_id)

        # Repeat input batch according to n_traj, the actual batch size = bs * n_traj
        self.input_batch = prompts.repeat(repeat_times=self.n_traj, interleave=True)
        self.batch_size = self.input_batch.batch["input_ids"].size(0)

        # Extract instance env info from data ground truth
        self.instance_dir = self.input_batch[0].non_tensor_batch["extra_info"]["instance_path"]
        self.instance_id_batch = [self.input_batch[i].non_tensor_batch["extra_info"]["instance_file"] for i in range(self.batch_size)]


    def load_env(self, instance_dir: str, instance_id: str) -> TextWorldEnvBase:
        if self.env == "textworld":
            from .env import TextWorldEnv
            instance_env = TextWorldEnv(instance_file=os.path.join(instance_dir, f"{instance_id}.z8"))
        elif self.env == "alfworld":
            from .env import AlfWorldEnv
            instance_env = AlfWorldEnv(instance_file=os.path.join(instance_dir, f"{instance_id}.tw-pddl"))
        else:
            raise NotImplementedError(f"Environment {self.env} not supported in TextWorldAgent.")
        return instance_env
        

    def interact(self, instance_env: TextWorldEnvBase, action: str) -> Tuple[str, bool, float, TextWorldEnvBase]:
        """
        A one-step interaction with the current instance environment.

        Args:
            instance_env (TextWorldEnvBase): the textworld environment instance
            action (str): the action to take at this step
        Returns:
            next_obs (str): the next observation after taking the action
            has_won (bool): whether the game has been won
            reward (float): the reward obtained from taking the action
            instance_env (TextWorldEnvBase): the updated instance environment
        """
        next_obs, has_won, reward = instance_env.one_step(command=action)
        return next_obs, has_won, reward, instance_env


    def run(self) -> DataProto:
        """
        The main function of generating multiturn rollouts by interacting with the textworld environments.

        Returns:
            A DataProto with the following fields:
            Tensor batch:
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids including both env and agent responses
            - response_mask: [bsz, response_length], 1 for agent tokens, 0 for env tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            Non-tensor batch:
            - sep_token_positions: 1D object ndarray, int indices of sep token ids in responses
            - final_rewards: 1D float ndarray, float values of 1.0/0 indicating game success status
            - interm_rewards: 1D object ndarray, list of float values of intermediate rewards per action
            - max_total_rewards: 1D float ndarray, float values of maximum total rewards for each instance
            - raw_response_text: 1D object ndarray, action only sequence, for eval purpose

        """
        messages_batch = [messages for messages in self.input_batch.non_tensor_batch["raw_prompt"]] 
        assert all(len(messages) == 1 for messages in messages_batch)
        # the initial message should be the following format: [{"role": "user", "content": ""}]
        # save the first user prompt
        prompt_str_batch = [messages[0]["content"] for messages in messages_batch] 
        # save the list of actions taken so far
        all_actions_batch = [[] for _ in range(self.batch_size)] 
        # save the list of env states observed so far
        all_states_batch = [[prompt_str_batch[i]] for i in range(self.batch_size)]
        # save the active status of each instance in the batch
        # active status: True if the instance is still active, False if the instance has stopped
        active_batch_idx = [True for _ in range(self.batch_size)]
        # save the final reward (1.0/0.0) for each instance
        final_reward_batch = [0.0 for _ in range(self.batch_size)] 
        # save the intermediate reward for each action taken so far
        interm_reward_batch = [[] for _ in range(self.batch_size)]
        # save the accumulated reward for each instance
        # accumulated reward: used in dense reward mode, sum of previous intermediate rewards
        accumulated_reward_batch = [0.0 for _ in range(self.batch_size)]
        # save the environment pddl instance for each batch
        instance_env_batch = [self.load_env(self.instance_dir, self.instance_id_batch[i]) for i in range(self.batch_size)] 

        # Start multi-turn rollouts in batches
        for k in range(self.max_iter):
            # Select active instances (instances that stop early) from batch at each turn
            selected_idx = np.where(active_batch_idx)[0].tolist()
            active_messages_batch = [messages_batch[idx] for idx in selected_idx]

            # Act
            output_str_batch, valid_output_batch_idx = self.batch_generate(active_messages_batch)

            # Env feedback
            for i in range(len(selected_idx)):
                # We only proceed to provide env observation if the output action is valid (contains sep token)
                if valid_output_batch_idx[i]:
                    all_actions_batch[selected_idx[i]].append(output_str_batch[i])
                    next_obs, has_won, reward, next_instance_env = self.interact(instance_env=instance_env_batch[selected_idx[i]], action=all_actions_batch[selected_idx[i]][-1])
                    # Update the next instance environment
                    instance_env_batch[selected_idx[i]] = next_instance_env
                    all_states_batch[selected_idx[i]].append(f"current state: {next_obs}" + "\n\nyour action: ")
                    # Construct next message
                    messages_batch[selected_idx[i]].extend([
                        {
                            "role": "assistant",
                            "content": all_actions_batch[selected_idx[i]][-1]
                        },
                        {
                            "role": "user",
                            "content": all_states_batch[selected_idx[i]][-1]
                        }
                    ])
                    # Provide intermediate reward per action (used in dense reward mode)
                    interm_reward_batch[selected_idx[i]].append(reward - accumulated_reward_batch[selected_idx[i]])
                    accumulated_reward_batch[selected_idx[i]] = reward
                    # Terminate rollout early if agent passes the game or exceeds max prompt length  
                    if has_won:
                        # If has won the game, assign final reward 1.0
                        active_batch_idx[selected_idx[i]] = False
                        final_reward_batch[selected_idx[i]] = 1.0
                    else:
                        # If current prompt length exceeds max length, terminate rollout early
                        if self._exceeds_prompt_length(messages_batch[selected_idx[i]]):
                            active_batch_idx[selected_idx[i]] = False
                else:
                    # If the output action is invalid (does not contain sep token), terminate rollout early
                    active_batch_idx[selected_idx[i]] = False
            
        # Get total rewards for each instance
        max_total_rewards_batch = [
            instance_env_batch[i].get_total_rewards()
            for i in range(self.batch_size)
        ]
        
        result_tensor_batch, result_non_tensor_batch = self.convert_result_to_dataproto(
            messages_batch=messages_batch,
            final_reward_batch=final_reward_batch,
            interm_reward_batch=interm_reward_batch,
            max_total_rewards_batch=max_total_rewards_batch,
        )
        
        non_tensor_batch = self.input_batch.non_tensor_batch
        for k, v in result_non_tensor_batch.items():
            non_tensor_batch[k] = v
        
        tensor_batch = TensorDict(result_tensor_batch, batch_size=self.batch_size)
        return DataProto(batch=tensor_batch, non_tensor_batch=non_tensor_batch)


    def batch_generate(self, messages_batch: List[List[Dict]]) -> Tuple[List[str], List[bool]]:
        """
        Call vLLM generate in batches. 
        Given a batch of chat messages so far, returns a batch of the latest assistant message string, and the batch indices of invalid output (action missing sep token)
        Note: sep_token is removed from output.

        Args:
            messages_batch (List[List[Dict]]): a batch of chat messages so far
        Returns:
            output_str_batch (List[str]): a batch of generated assistant message strings
            valid_output_batch_idx (List[bool]): a batch of bool indicating if the output contains sep_token
        """
        # Apply chat template to batch of messages
        tokens_prompt_batch = []
        for messages in messages_batch:
            input_tokens = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensor="pt"
            )
            tokens_prompt_batch.append(TokensPrompt(prompt_token_ids=input_tokens))

        outputs = self.inference_engine.generate(
            prompts=tokens_prompt_batch,
            sampling_params=self.sampling_params
        )

        output_ids_batch = [output.outputs[0].token_ids for output in outputs]
        # Find action separation token for each action. If not found, end interaction early.
        valid_action_batch_idx = self._has_sep_token(output_ids_batch)

        output_str_batch = []
        for i, (output_ids, has_sep) in enumerate(zip(output_ids_batch, valid_action_batch_idx)):
            if has_sep:
                output_str = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                output_str_batch.append(output_str)
            else:
                output_str_batch.append("")

        return output_str_batch, valid_action_batch_idx


    def convert_result_to_dataproto(self, messages_batch: List[List[Dict]], final_reward_batch: List[float], interm_reward_batch: List[List[float]], max_total_rewards_batch: List[float]) -> Tuple[Dict, Dict]:
        """
        Convert the rollout result to DataProto format.

        Args:
            messages_batch (List[List[Dict]]): a batch of chat messages after rollout
            final_reward_batch (List[float]): final score (0.0/1.0) for each instance
            interm_reward_batch (List[List[float]]): intermediate reward per action such as [0, 0.2, 0.1, 0, 0.5, ...] for each instance
            max_total_rewards_batch (List[float]): maximum total rewards for each instance

        Returns:
            Tensor batch:
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids including both env and agent responses
            - response_mask: [bsz, response_length], 1 for agent tokens, 0 for env tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            Non-tensor batch:
            - sep_token_positions: 1D object ndarray, int indices of sep token ids in responses
            - final_rewards: 1D float ndarray, float values of 1.0/0.0 indicating game winning status
            - interm_rewards: 1D object ndarray, list of float values of intermediate rewards per action
            - max_total_rewards: 1D float ndarray, float values of maximum total rewards for each instance
            - raw_response_text: 1D object ndarray, action only sequence, for eval purpose
        """
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        prompt_ids = torch.full((self.batch_size, self.max_prompt_len), self.pad_token_id, dtype=torch.long, device=self.device)
        response_ids = torch.full((self.batch_size, self.max_response_len), self.pad_token_id, dtype=torch.long, device=self.device)
        total_len = self.max_prompt_len + self.max_response_len
        input_ids = torch.full((self.batch_size, total_len), self.pad_token_id, dtype=torch.long, device=self.device)
        loss_mask = torch.zeros((self.batch_size, self.max_response_len), dtype=torch.long, device=self.device)
        attention_mask = torch.zeros((self.batch_size, total_len), dtype=torch.long, device=self.device)
        position_ids = torch.zeros((self.batch_size, total_len), dtype=torch.long, device=self.device)

        sep_token_positions = [[] for _ in range(self.batch_size)]
        final_rewards = final_reward_batch
        interm_rewards = interm_reward_batch
        max_total_rewards = max_total_rewards_batch
        raw_response_text = ["" for _ in range(self.batch_size)]

        for i in range(self.batch_size):
            assert len(messages_batch[i]) % 2 == 1 # Should be "user...assistant...user......user"
            # Construct prompt_ids from the first user message
            prompt_tokens = self.tokenizer.apply_chat_template(
                messages_batch[i][:1], tokenize=True, add_generation_prompt=True
            )
            # Truncate prompt ids
            if len(prompt_tokens) > self.max_prompt_len:
                prompt_tokens = prompt_tokens[:self.max_prompt_len]
            # Add left padding to prompt ids
            prompt_ids[i, (self.max_prompt_len - len(prompt_tokens)):] = torch.tensor(prompt_tokens) 

            # Construct response_ids from subsequent user messages throughout the interaction
            response_tokens = []
            # index positions of sep tokens in response_ids  
            sep_tokens: List[int] = []  
            response_loss_mask = []
            response_text = ""
            for k in range(len(messages_batch[i]) // 2):
                # Append action tokens
                turn_action_tokens = self.tokenizer.encode(messages_batch[i][2*k+1]["content"], add_special_tokens=False)
                response_tokens.extend(turn_action_tokens)
                # Append sep token index and sep token id
                sep_tokens.append(len(response_tokens))
                response_tokens.append(self.sep_token_id)
                # Create response loss mask, 1 for agent tokens, 0 for env tokens
                # Response loss mask should include sep token
                response_loss_mask.extend([1] * (len(turn_action_tokens) + 1)) 
                response_text += messages_batch[i][2*k+1]["content"] + self.sep_token
                # Append env responses (user messages), excluded from loss
                turn_env_text = self._truncate_system_template(
                    self.tokenizer.apply_chat_template(
                        [messages_batch[i][2*k+2]], tokenize=False, add_generation_prompt=True
                    )
                )
                # Append env tokens
                turn_env_tokens = self.tokenizer.encode(turn_env_text, add_special_tokens=True)
                response_tokens.extend(turn_env_tokens)
                # Create response loss mask, 1 for agent tokens, 0 for env tokens
                response_loss_mask.extend([0] * len(turn_env_tokens))

            # Truncate response ids
            if len(response_tokens) > self.max_response_len:
                response_tokens = response_tokens[:self.max_response_len]
                # Truncate response_loss_mask in parallel
                response_loss_mask = response_loss_mask[:self.max_response_len] 
            # Add right padding to response ids
            response_ids[i, :len(response_tokens)] = torch.tensor(response_tokens)

            # Construct input_ids from prompt_ids and response_ids
            input_ids[i, :self.max_prompt_len] = prompt_ids[i]
            input_ids[i, self.max_prompt_len:total_len] = response_ids[i]

            # Truncate loss mask
            assert len(response_loss_mask) == len(response_tokens)
            if len(response_loss_mask) > self.max_response_len:
                response_loss_mask = response_loss_mask[:self.max_response_len]
            loss_mask[i, :len(response_loss_mask)] = torch.tensor(response_loss_mask)

            # Construct attention mask and position ids
            attention_mask[i, (self.max_prompt_len - len(prompt_ids)):self.max_prompt_len] = 1  # Prompt actual tokens
            attention_mask[i, self.max_prompt_len:(self.max_prompt_len + len(response_tokens))] = 1  # Response actual tokens
            
            position_ids[i, (self.max_prompt_len - len(prompt_ids)):self.max_prompt_len] = torch.arange(len(prompt_ids))
            response_positions = torch.arange(len(prompt_ids), len(prompt_ids) + self.max_response_len)
            position_ids[i, self.max_prompt_len:] = response_positions

            # Add to non tensor batch info list
            sep_token_positions[i] = sep_tokens
            raw_response_text[i] = response_text

        # Formatting result
        tensor_batch = {
            "prompts": prompt_ids,
            "responses": response_ids,
            "response_mask": loss_mask, # this is the loss mask applied to any policy-related loss and updates
            "input_ids": input_ids,  # here input_ids become the whole sentences
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

        non_tensor_batch = {
            "sep_token_positions": self._create_consistent_object_array(sep_token_positions),
            "final_rewards": np.array(final_rewards),
            "interm_rewards": self._create_consistent_object_array(interm_rewards),
            "max_total_rewards": np.array(max_total_rewards),
            "raw_response_text": self._create_consistent_object_array(raw_response_text)
        }
        
        return tensor_batch, non_tensor_batch
                
