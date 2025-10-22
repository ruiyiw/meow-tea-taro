from typing import Any, Optional
from uuid import uuid4
import ray
from pathlib import Path
from shutil import rmtree
import contextlib
import os
import json

from verl.utils.rollout_trace import rollout_trace_op
from sweagent.environment.swe_env import SWEEnv
from sweagent.agent.agents import DefaultAgent, DefaultAgentConfig
from sweagent.run.common import save_predictions
from sweagent.run.evaluate import evaluate_instance
from ..base.agent_loop import AgentLoopBase, AgentLoopOutput, AgentLoopMetrics, register
from .env_wrapper import batch_instance_from_dict, remove_runtime_root


@contextlib.contextmanager
def silence_stdout_hard():
    """
    Mute ALL stdout (Python prints and C-level fd=1) inside the block.
    Leaves stderr untouched, so logger output still appears.
    """
    devnull = open(os.devnull, "w")
    saved_fd = os.dup(1)            # save current stdout fd
    try:
        os.dup2(devnull.fileno(), 1)  # point fd=1 to /dev/null
        yield
    finally:
        os.dup2(saved_fd, 1)          # restore stdout
        os.close(saved_fd)
        devnull.close()


@ray.remote(num_cpus=0.01)
def run_sweagent_remote(
    instance_dict: dict,
    sweagent_config: dict,
    output_base_dir: str,
    request_id: str,
    **kwargs
):
    """Run SWE-agent in a Ray remote function.
    
    The DefaultAgent inside will make HTTP calls to the OpenAI-compatible server
    configured via environment variables (OPENAI_BASE_URL, etc.)
    """
    from loguru import logger
      
    # Turn instance dict into a BatchInstance
    batch_instance = batch_instance_from_dict(d=instance_dict)
    instance_id = str(instance_dict.get("instance_id"))

    # Create output directory for each task
    global_path = Path(output_base_dir)
    global_path.mkdir(parents=True, exist_ok=True)

    # SWE-agent will create: <runtime_root>/<instance_id>/workspace/<repo>/.git
    runtime_root = global_path / f"{instance_id}_{request_id}"
    if runtime_root.exists():
        rmtree(runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=True)

    # Per-trajectory output dir for logs/artifacts
    output_dir = global_path / f"{instance_id}_{request_id}_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Point SWE-agent at the UNIQUE runtime root
    batch_instance.env.deployment.instance_root = str(runtime_root)
    batch_instance.env.deployment.conda_root = str(runtime_root / ".conda")

    agent = None
    env = None
    result = None
    reward = 0.0
    error = None
    
    try:
        # Create SWEEnv
        env = SWEEnv.from_config(batch_instance.env)
        
        # Create DefaultAgent - it will use environment variables for LLM connection
        agent = DefaultAgent.from_config(DefaultAgentConfig.model_validate(sweagent_config.get("agent", {})))
        
        # Start environment
        with silence_stdout_hard():
            env.start()

        with silence_stdout_hard():
            result = agent.run(
                problem_statement=batch_instance.problem_statement,
                env=env,
                output_dir=output_dir,
            )
    except Exception as e:
        logger.error(f"Error processing instance {instance_id}: {e}", exc_info=True)
        error = str(e)
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            remove_runtime_root(runtime_root=runtime_root, traj_root=global_path)
    
    # Persist outputs or error
    if result is not None and getattr(result, "info", None) is not None:
        save_predictions(output_dir, instance_id, result)
    else:
        (output_dir / "error.txt").write_text(error or "agent returned None")

    # Evaluate if agent completed successfully
    if agent is not None:
        try:
            with silence_stdout_hard():
                eval_summary = evaluate_instance(
                    instance=batch_instance,
                    output_dir=output_dir,
                    timeout=600,
                )
                (output_dir / "eval_summary.json").write_text(json.dumps(eval_summary, indent=2))
            if result:
                report = (eval_summary or {}).get("report") or {}
                node = report.get(instance_id) or {}
                pass_ratio = node.get("pass_ratio")
                if pass_ratio is not None:
                    reward = float(pass_ratio)
        except Exception as e:
            logger.debug(f"Error during evaluation {e}")
            error = str(e)

    messages = agent.messages if agent is not None else []
    return messages, reward, error


@register("software_agent")
class SoftwareAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(
        cls, 
        config,
        trajectory,
        tokenizer, 
        processor, 
        **kwargs
    ):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level SoftwareAgentLoop initialization")

        # Initialize tools from config file
        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.trajectory = trajectory

        cls.sweagent_config_path = config.agentic.agent_loop.kwargs.sweagent_config_path
        cls.sweagent_traj_dir = config.agentic.agent_loop.kwargs.sweagent_trajs_dir
        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.response_length = config.actor_rollout_ref.rollout.response_length

    
    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run SWE-agent loop by dispatching to Ray remote function.
        
        This avoids AsyncLLMServerManager and instead relies on:
        1. Ray remote execution for parallelism
        2. DefaultAgent making direct HTTP calls to OpenAI-compatible server
        3. Environment variables (OPENAI_BASE_URL) to configure the LLM endpoint
        """
        request_id = uuid4().hex

        # Get instance data from kwargs
        instance_dict = kwargs.get("instance", {})
        
        # Load SWE-agent config
        import yaml
        sweagent_config = yaml.safe_load(
            Path(self.sweagent_config_path).expanduser().read_text(encoding="utf-8")
        )
        
        # Determine output directory
        global_step = self.trajectory["step"]
        training_phase = "eval" if self.trajectory["validate"] else "train"
        output_base_dir = Path(self.sweagent_traj_dir) / f"step_{global_step}" / training_phase
        
        # Run SWE-agent remotely
        # The DefaultAgent inside will make HTTP calls to the server
        # configured via OPENAI_BASE_URL environment variable
        messages, reward, error = await run_sweagent_remote.remote(
            instance_dict=instance_dict,
            sweagent_config=sweagent_config,
            output_base_dir=str(output_base_dir),
            request_id=request_id,
        )
        
        if not messages or error:
            # Return empty trajectory on failure
            return self._create_empty_trajectory(kwargs.get("raw_prompt", []), error)
        
        # Process messages to extract prompt_ids and response_ids
        # Assume first 2 messages are system + user (initial prompt)
        initial_messages = messages[:2]
        response_messages = messages[2:]
        
        # Tokenize initial prompt
        initial_input_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                initial_messages,
                add_generation_prompt=False,
                tokenize=True,
                **self.apply_chat_template_kwargs,
            ),
        )
        
        # Process response messages
        response_ids = []
        response_mask = []
        
        # Remove trailing user messages (final git diff)
        last_idx = len(response_messages) - 1
        while last_idx >= 0 and response_messages[last_idx]["role"] == "user":
            last_idx -= 1
        
        if last_idx >= 0:
            response_messages = response_messages[:last_idx + 1]
        
        # Tokenize each response message
        for message in response_messages:
            msg_encoding = await self.loop.run_in_executor(
                None,
                lambda m=message: self.tokenizer.apply_chat_template(
                    [m],
                    add_generation_prompt=False,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
            
            response_ids.extend(msg_encoding)
            
            # Mask: 0 for user, 1 for assistant
            if message["role"] == "user":
                response_mask.extend([0] * len(msg_encoding))
            else:  # assistant
                response_mask.extend([1] * len(msg_encoding))
        
        # Truncate to response_length
        response_ids = response_ids[:self.response_length]
        response_mask = response_mask[:self.response_length]
        
        # Create output
        output = AgentLoopOutput(
            prompt_ids=initial_input_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=None,  # SWE-agent doesn't provide logprobs
            reward_score=reward,
            num_turns=len(messages) // 2,  # Approximate turns
            metrics=AgentLoopMetrics(
                generate_sequences=len(response_messages),
                tool_calls=0,  # SWE-agent uses tools but we don't track separately here
            ),
            extra_fields={"error": error} if error else {},
        )
        
        return output
    
    def _create_empty_trajectory(self, raw_prompt: list, error: str) -> AgentLoopOutput:
        """Create an empty/dummy trajectory for failed cases."""
        failure_message = [{"role": "assistant", "content": f"Failed: {error or 'Unknown error'}"}]
        
        response_ids = self.tokenizer.apply_chat_template(
            failure_message,
            add_generation_prompt=False,
            tokenize=True,
            **self.apply_chat_template_kwargs,
        )
        
        prompt_ids = self.tokenizer.apply_chat_template(
            raw_prompt,
            add_generation_prompt=False,
            tokenize=True,
            **self.apply_chat_template_kwargs,
        )
        
        response_mask = [1] * len(response_ids)
        
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=None,
            reward_score=0.0,
            num_turns=1,
            metrics=AgentLoopMetrics(generate_sequences=0, tool_calls=0),
            extra_fields={"error": error},
        )
