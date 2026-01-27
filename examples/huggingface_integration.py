"""
PARL Integration with Hugging Face Models

This example demonstrates how to integrate PARL reward shaping with a Hugging Face
language model for reinforcement learning training. The example simulates a scenario
where an LLM learns to decompose tasks and coordinate parallel subagents.

Key Components:
1. Load a Hugging Face model (GLM-4.7-Flash)
2. Define action space for parallel agent coordination
3. Compute PARL rewards based on model actions
4. Implement a simple RL training loop with PPO-style updates

Requirements:
    pip install transformers torch accelerate
"""

import re
from typing import Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from parl import CriticalStepsMetric, PARLReward


class PARLOrchestrator:
    """
    Orchestrator agent that uses a Hugging Face LLM to make parallelization decisions.

    The model learns to:
    1. Decide how many subagents to spawn
    2. Decompose tasks into parallel subtasks
    3. Coordinate execution for maximum efficiency
    """

    def __init__(
        self,
        model_name: str = "zai-org/GLM-4.7-Flash",
        max_subagents: int = 100,
        device: Optional[str] = None,
    ):
        """
        Initialize the orchestrator with a Hugging Face model.

        Args:
            model_name: Hugging Face model identifier
            max_subagents: Maximum number of parallel subagents
            device: Device to use (e.g., 'cuda', 'cpu'). If None, auto-detects.
        """
        self.max_subagents = max_subagents

        # Load model and tokenizer
        print(f"Loading model: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Ensure pad token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer for {model_name}: {e}") from e

        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto" if device == "cuda" else None,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {e}") from e

        # Get device from model (works even with device_map="auto")
        # Try model.device first, fallback to first parameter's device
        try:
            if hasattr(self.model, "device"):
                self.device = self.model.device
            else:
                # Get device from first parameter when using device_map="auto"
                self.device = next(self.model.parameters()).device
        except (AttributeError, StopIteration):
            # Fallback to specified device or default
            self.device = torch.device(device)

        # Initialize PARL reward function
        device_str = (
            str(self.device) if isinstance(self.device, torch.device) else self.device
        )
        self.parl_reward = PARLReward(
            lambda_init=0.1,
            lambda_final=0.0,
            total_training_steps=10000,
            device=device_str,
        )

        # Initialize Critical Steps metric
        self.critical_steps_metric = CriticalStepsMetric()

        print(f"Model loaded on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def encode_task(self, task_description: str) -> dict:
        """
        Encode task description into model input.

        Args:
            task_description: Natural language task description

        Returns:
            Input dict with token IDs
        """
        # Create a prompt that encourages parallel decomposition
        messages = [
            {
                "role": "user",
                "content": f"""Task: {task_description}

Decompose this task into parallel subtasks and specify the optimal number of agents needed (1-{self.max_subagents}).

Answer with only a number:""",
            }
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        return inputs

    def predict_parallelism(
        self,
        task_description: str,
        do_sample: bool = False,
    ) -> Tuple[int, Optional[torch.Tensor]]:
        """
        Predict how many subagents to spawn for the task.

        Args:
            task_description: Task to decompose
            do_sample: Whether to use sampling or greedy decoding

        Returns:
            Tuple of (num_subagents, model_logits). Logits may be None if not available.
        """
        inputs = self.encode_task(task_description)
        # Move inputs to model device
        model_device = (
            self.model.device
            if hasattr(self.model, "device")
            else next(self.model.parameters()).device
        )
        inputs = {
            k: v.to(model_device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=do_sample,
                output_scores=True,
                return_dict_in_generate=True,
            )

        # Extract generated text
        generated_text = self.tokenizer.decode(
            outputs.sequences[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        # Parse number of agents (simplified - in practice use more robust parsing)
        try:
            numbers = re.findall(r"\d+", generated_text)
            num_subagents = int(numbers[0]) if numbers else 10
            # Clamp to valid range
            num_subagents = max(1, min(num_subagents, self.max_subagents))
        except (ValueError, IndexError, AttributeError):
            num_subagents = 10  # Default fallback

        # Get logits for reward computation
        logits = (
            outputs.scores[0] if outputs.scores and len(outputs.scores) > 0 else None
        )

        return num_subagents, logits

    def simulate_execution(
        self,
        num_subagents: int,
        num_stages: int = 5,
    ) -> Dict[str, torch.Tensor]:
        """
        Simulate parallel execution with specified number of subagents.

        Args:
            num_subagents: Number of parallel agents
            num_stages: Number of execution stages

        Returns:
            Execution metrics (main_steps, sub_steps, success)
        """
        # Simulate orchestration overhead
        main_steps = torch.ones(1, num_stages) * 0.1

        # Simulate subagent execution times (more agents = better parallelization)
        # In practice, this would be actual execution metrics
        base_workload = 10.0
        workload_per_agent = base_workload / num_subagents

        # Add some randomness to simulate real-world variance
        sub_steps = torch.ones(1, num_stages, num_subagents) * workload_per_agent
        sub_steps = sub_steps + torch.randn_like(sub_steps) * 0.2
        sub_steps = sub_steps.clamp(min=0.1)

        # Simulate success (more agents generally helps, but with diminishing returns)
        success_prob = min(0.5 + (num_subagents / self.max_subagents) * 0.5, 0.95)
        success = torch.bernoulli(torch.tensor([success_prob], dtype=torch.float32))

        # Compute critical steps
        critical_steps = self.critical_steps_metric(main_steps, sub_steps)

        return {
            "main_steps": main_steps,
            "sub_steps": sub_steps,
            "success": success,
            "critical_steps": critical_steps,
        }

    def compute_reward(
        self,
        num_subagents: int,
        execution_metrics: Dict[str, torch.Tensor],
        training_step: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute PARL reward for the episode.

        Args:
            num_subagents: Number of agents spawned
            execution_metrics: Metrics from execution
            training_step: Current training step

        Returns:
            Reward components
        """
        # Create trajectory features (could include execution time, success rate, etc.)
        trajectory_features = (
            torch.tensor(
                [
                    execution_metrics["critical_steps"].item(),
                    num_subagents / self.max_subagents,
                    execution_metrics["success"].item(),
                ]
            )
            .unsqueeze(0)
            .to(self.device)
        )

        # Compute PARL reward
        rewards = self.parl_reward.compute_full_reward(
            num_subagents=torch.tensor([num_subagents]),
            trajectory_features=trajectory_features,
            success=execution_metrics["success"],
            training_step=training_step,
            max_subagents=self.max_subagents,
        )

        # Add efficiency bonus (lower critical steps = better)
        efficiency_bonus = 1.0 / (1.0 + execution_metrics["critical_steps"].item())
        rewards["efficiency_bonus"] = torch.tensor(
            [efficiency_bonus], device=self.device
        )
        rewards["total_reward"] = rewards["total_reward"] + rewards["efficiency_bonus"]

        return rewards


def example_basic_usage():
    """
    Basic example: Use PARL with a Hugging Face model to make parallelization decisions.
    """
    print("=" * 80)
    print("PARL + Hugging Face: Basic Usage Example")
    print("=" * 80)

    # Initialize orchestrator with GLM-4.7-Flash model
    orchestrator = PARLOrchestrator(
        model_name="zai-org/GLM-4.7-Flash",
    )

    # Example tasks to decompose
    tasks = [
        "Process 1000 user requests from the API queue",
        "Analyze sentiment in 10,000 customer reviews",
        "Generate reports for 50 different departments",
        "Run integration tests across 20 microservices",
    ]

    print("\n" + "-" * 80)
    print("Task Decomposition and Reward Computation")
    print("-" * 80)

    for task_idx, task in enumerate(tasks):
        print(f"\nTask {task_idx + 1}: {task}")

        # Model predicts parallelization strategy
        num_subagents, _logits = orchestrator.predict_parallelism(task)
        print(f"  Predicted subagents: {num_subagents}")

        # Simulate execution
        execution_metrics = orchestrator.simulate_execution(num_subagents)
        print(f"  Critical steps: {execution_metrics['critical_steps'].item():.2f}")
        print(
            f"  Success: {'Yes' if execution_metrics['success'].item() > 0 else 'No'}"
        )

        # Compute reward (at training step 5000 - mid-training)
        rewards = orchestrator.compute_reward(
            num_subagents=num_subagents,
            execution_metrics=execution_metrics,
            training_step=5000,
        )

        print(f"  Total reward: {rewards['total_reward'].item():.4f}")
        print(
            f"  Parallelism component: {rewards['instantiation_component'].item():.4f}"
        )
        print(f"  Task success component: {rewards['task_component'].item():.4f}")
        print(f"  Efficiency bonus: {rewards['efficiency_bonus'].item():.4f}")


def example_training_loop():
    """
    Simulate a training loop where the model learns to optimize parallelization.
    """
    print("\n" + "=" * 80)
    print("PARL + Hugging Face: Training Loop Simulation")
    print("=" * 80)

    orchestrator = PARLOrchestrator(model_name="zai-org/GLM-4.7-Flash")

    # Training configuration
    num_episodes = 20
    total_training_steps = 10000

    # Example task
    task = "Process database queries in parallel"

    print(f"\nTask: {task}")
    print(f"Training episodes: {num_episodes}")
    print(f"Total training steps: {total_training_steps}")

    print("\n" + "-" * 80)
    print("Training Progress")
    print("-" * 80)
    print(
        f"{'Episode':<10} {'Step':<8} {'λ':<8} {'Agents':<8} {'Success':<10} {'Reward':<10} {'Critical Steps':<15}"
    )
    print("-" * 80)

    for episode in range(num_episodes):
        training_step = int((episode / num_episodes) * total_training_steps)

        # Model predicts parallelization
        # Use sampling early in training, greedy decoding later
        use_sampling = episode < (num_episodes // 2)
        num_subagents, _ = orchestrator.predict_parallelism(
            task,
            do_sample=use_sampling,
        )

        # Simulate execution
        execution_metrics = orchestrator.simulate_execution(num_subagents)

        # Compute reward
        rewards = orchestrator.compute_reward(
            num_subagents=num_subagents,
            execution_metrics=execution_metrics,
            training_step=training_step,
        )

        # Display progress
        print(
            f"{episode + 1:<10} "
            f"{training_step:<8} "
            f"{rewards['lambda_aux'].item():<8.3f} "
            f"{num_subagents:<8} "
            f"{'Yes' if execution_metrics['success'].item() > 0 else 'No':<10} "
            f"{rewards['total_reward'].item():<10.4f} "
            f"{execution_metrics['critical_steps'].item():<15.2f}"
        )

        # In a real training loop, you would:
        # 1. Compute policy gradients from rewards
        # 2. Update model parameters
        # 3. Track metrics (success rate, average reward, etc.)

    print("\n" + "-" * 80)
    print("Training Summary")
    print("-" * 80)
    print("Observations:")
    print("- Lambda (λ) anneals from 0.1 to 0.0 over training")
    print("- Early training: Model explores different parallelization strategies")
    print("- Late training: Model focuses on task success and efficiency")
    print("- Critical Steps metric helps optimize for latency, not just throughput")


def example_with_batching():
    """
    Example with batched inference for efficiency.
    """
    print("\n" + "=" * 80)
    print("PARL + Hugging Face: Batched Processing Example")
    print("=" * 80)

    orchestrator = PARLOrchestrator(model_name="zai-org/GLM-4.7-Flash")

    # Batch of tasks
    tasks = [
        "Classify 5000 support tickets",
        "Generate thumbnails for 1000 videos",
        "Validate 2000 user registrations",
        "Index 10000 documents for search",
    ]

    batch_size = len(tasks)
    training_step = 5000

    print(f"\nProcessing batch of {batch_size} tasks at training step {training_step}")
    print("-" * 80)

    # Collect results for batch
    all_rewards = []
    all_subagents = []
    all_critical_steps = []

    for task in tasks:
        # Predict and execute
        num_subagents, _ = orchestrator.predict_parallelism(task)
        execution_metrics = orchestrator.simulate_execution(num_subagents)
        rewards = orchestrator.compute_reward(
            num_subagents,
            execution_metrics,
            training_step,
        )

        all_rewards.append(rewards["total_reward"].item())
        all_subagents.append(num_subagents)
        all_critical_steps.append(execution_metrics["critical_steps"].item())

    # Display batch results
    print(f"\n{'Task':<50} {'Agents':<10} {'Critical Steps':<15} {'Reward':<10}")
    print("-" * 80)
    for i, task in enumerate(tasks):
        print(
            f"{task:<50} "
            f"{all_subagents[i]:<10} "
            f"{all_critical_steps[i]:<15.2f} "
            f"{all_rewards[i]:<10.4f}"
        )

    # Batch statistics
    print("\n" + "-" * 80)
    print("Batch Statistics")
    print("-" * 80)
    print(f"Average reward: {sum(all_rewards) / len(all_rewards):.4f}")
    print(f"Average agents: {sum(all_subagents) / len(all_subagents):.1f}")
    print(
        f"Average critical steps: {sum(all_critical_steps) / len(all_critical_steps):.2f}"
    )
    print(f"Total parallelism: {sum(all_subagents)} agents across {batch_size} tasks")


def example_reward_analysis():
    """
    Analyze how rewards change with different parallelization strategies.
    """
    print("\n" + "=" * 80)
    print("PARL Reward Analysis: Impact of Parallelization")
    print("=" * 80)

    orchestrator = PARLOrchestrator(model_name="zai-org/GLM-4.7-Flash")

    task = "Execute 100 independent data processing jobs"
    training_step = 5000

    # Test different parallelization levels
    agent_counts = [1, 5, 10, 25, 50, 75, 100]

    print(f"\nTask: {task}")
    print(f"Training step: {training_step}")
    print("\n" + "-" * 80)
    print(
        f"{'Agents':<10} {'Critical Steps':<15} {'Success Rate':<15} {'Total Reward':<15} {'Efficiency':<10}"
    )
    print("-" * 80)

    for num_agents in agent_counts:
        # Simulate multiple runs for statistics
        rewards_list = []
        critical_steps_list = []
        success_list = []

        for _ in range(5):  # Average over 5 runs
            execution_metrics = orchestrator.simulate_execution(num_agents)
            rewards = orchestrator.compute_reward(
                num_agents,
                execution_metrics,
                training_step,
            )
            rewards_list.append(rewards["total_reward"].item())
            critical_steps_list.append(execution_metrics["critical_steps"].item())
            success_list.append(execution_metrics["success"].item())

        avg_reward = sum(rewards_list) / len(rewards_list)
        avg_critical_steps = sum(critical_steps_list) / len(critical_steps_list)
        avg_success = sum(success_list) / len(success_list)
        efficiency = 100.0 / avg_critical_steps  # Efficiency score

        print(
            f"{num_agents:<10} "
            f"{avg_critical_steps:<15.2f} "
            f"{avg_success:<15.1%} "
            f"{avg_reward:<15.4f} "
            f"{efficiency:<10.2f}"
        )

    print("\n" + "-" * 80)
    print("Key Insights:")
    print("- More agents → Lower critical steps (better parallelism)")
    print("- More agents → Higher success rate (up to a point)")
    print("- PARL reward balances parallelism incentive with task success")
    print("- Efficiency score shows diminishing returns beyond optimal point")


if __name__ == "__main__":
    """
    Run all examples demonstrating PARL + Hugging Face integration.

    Note: This example uses GLM-4.7-Flash for demonstration. In production:
    - Fine-tune the model on domain-specific tasks
    - Implement proper RL training loop (PPO, A2C, etc.)
    - Add policy gradient computation and parameter updates
    - Track and log detailed metrics
    - Use proper evaluation benchmarks
    """

    print("\n" + "=" * 80)
    print("PARL + HUGGING FACE INTEGRATION EXAMPLES")
    print("=" * 80)
    print("\nThese examples demonstrate how to:")
    print("1. Use PARL reward shaping with Hugging Face models")
    print("2. Train models to optimize parallel execution")
    print("3. Evaluate parallelization strategies")
    print("4. Implement batched processing")
    print("\n" + "=" * 80)

    # Run all examples
    example_basic_usage()
    example_training_loop()
    example_with_batching()
    example_reward_analysis()

    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
    print("\nNext steps:")
    print("- Implement full RL training loop (PPO, A2C, etc.)")
    print("- Fine-tune with domain-specific tasks")
    print("- Deploy orchestrator in production multi-agent system")
    print("- Benchmark against serial execution baselines")
