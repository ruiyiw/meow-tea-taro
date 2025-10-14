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

import os
import logging
from omegaconf import DictConfig
from vllm import SamplingParams
from omegaconf import OmegaConf

from verl import DataProto
from verl.single_controller.ray.base import RayWorkerGroup
import logging
import os
import pickle
import socket
import threading
from contextlib import contextmanager
from copy import deepcopy
from types import MethodType
from typing import Any, Dict, List, Union

import numpy as np
from torch.distributed.device_mesh import DeviceMesh
import zmq
from filelock import FileLock
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps

from verl import DataProto
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.utils.profiler import GPUMemoryLogger
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class vLLMSyncAgenticRollout(vLLMRollout):
    """
    A synchronous rollout class that wraps vLLMRollout but replaces the generation logic with a multi-turn agentic interaction.

    It inherits vLLMRollout in verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py
    Only the `generate_sequences` function is overridden.
    """
    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig, 
        device_mesh: DeviceMesh,
    ):
        # Initialize the parent vLLMRollout
        super().__init__(config, model_config, device_mesh)
        self.config = config

        # Convert agentic dict to OmegaConf for dot notation access
        if hasattr(config, "agentic") and config.agentic:
            self.agentic_config = OmegaConf.create(config.agentic)
        else:
            raise ValueError("config.agentic required for vLLMSyncAgenticRollout.")


    
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """
        Generate sequences using multi-turn agentic interaction.
        Override the parent's generate_sequences method.
        """
        env_name = self.agentic_config.environment.name.lower()
        if env_name in ["textworld", "alfworld"]:
            from meow_tea_train.agentic_menu.sync_textworld.agent import TextWorldAgent
            agent = TextWorldAgent(
                env=env_name,
                prompts=prompts,
                inference_engine=self.inference_engine,
                sampling_params=self.sampling_params,
                tokenizer=self.tokenizer,
                max_iter=self.agentic_config.environment.max_iter,
                n_traj=self.agentic_config.environment.n_traj,
                max_prompt_len=self.config.prompt_length,
                max_response_len=self.config.response_length
            )
            return agent.run()
        else:
            raise NotImplementedError(f"Environment {env_name} not supported in vLLMSyncAgenticRollout.")