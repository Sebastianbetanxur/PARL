"""
PARL Reward Function Implementation in PyTorch

Implements the staged reward shaping from the paper:
R_t = λ_aux(e) · r_parallel + (1 - λ_aux(e)) · (𝟙[success] · Q(τ))

where λ_aux(e) anneals from 0.1 → 0.0 over training.

Requirements:
    pip install torch

This implementation provides:
- Staged reward shaping with lambda annealing
- Instantiation reward (r_parallel) to encourage parallelism
- Task quality computation Q(τ)
- Critical Steps metric for latency evaluation
- Differentiable components for gradient-based optimization
"""

import torch
import torch.nn as nn


class PARLReward(nn.Module):
    """
    Parallel-Agent Reinforcement Learning Reward Function

    Implements staged reward shaping that:
    1. Encourages parallelism early in training (via r_parallel)
    2. Gradually shifts focus toward task success (via Q(τ))
    """

    def __init__(
        self,
        lambda_init: float = 0.1,
        lambda_final: float = 0.0,
        total_training_steps: int = 10000,
        device: str = "cpu",
    ):
        super().__init__()

        self.lambda_init = lambda_init
        self.lambda_final = lambda_final
        self.total_training_steps = total_training_steps
        self.device = device

        # Register lambda as a buffer (not a trainable parameter)
        self.register_buffer("current_step", torch.tensor(0, dtype=torch.long))

    def anneal_lambda(self, training_step: int) -> torch.Tensor:
        """
        Anneal λ_aux from 0.1 → 0.0 over training

        Args:
            training_step: Current training step

        Returns:
            Current λ_aux value as a tensor
        """
        progress = min(1.0, training_step / self.total_training_steps)

        lambda_aux = (
            self.lambda_init + (self.lambda_final - self.lambda_init) * progress
        )

        return torch.tensor(lambda_aux, dtype=torch.float32, device=self.device)

    def compute_instantiation_reward(
        self, num_subagents: torch.Tensor, max_subagents: int = 100
    ) -> torch.Tensor:
        """
        Compute r_parallel: incentivizes subagent instantiation and concurrent execution

        Args:
            num_subagents: Number of subagents instantiated (batch_size,)
            max_subagents: Maximum allowed subagents

        Returns:
            Instantiation reward (batch_size,)
        """
        # Normalize by max capacity
        normalized_count = num_subagents.float() / max_subagents

        # Reward increases with parallelism
        r_parallel = normalized_count

        return r_parallel

    def compute_task_quality(
        self, trajectory_features: torch.Tensor, success_indicators: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Q(τ): end-to-end task quality

        Args:
            trajectory_features: Features extracted from trajectory (batch_size, feature_dim)
            success_indicators: Binary success indicators (batch_size,)

        Returns:
            Task quality scores (batch_size,)
        """
        # In practice, this could be a learned quality function
        # For now, we use a simple weighted combination

        # Base quality from trajectory features (could be learned)
        quality = trajectory_features.mean(dim=-1)

        # Modulate by success
        quality = quality * success_indicators.float()

        return quality.clamp(0.0, 1.0)

    def forward(
        self,
        r_parallel: torch.Tensor,
        success: torch.Tensor,
        task_quality: torch.Tensor,
        training_step: int,
    ) -> torch.Tensor:
        """
        Compute the full PARL reward:
        R_t = λ_aux(e) · r_parallel + (1 - λ_aux(e)) · (𝟙[success] · Q(τ))

        Args:
            r_parallel: Instantiation reward (batch_size,)
            success: Success indicators (batch_size,)
            task_quality: Task quality scores (batch_size,)
            training_step: Current training step

        Returns:
            Total reward (batch_size,)
        """
        # Anneal lambda based on training progress
        lambda_aux = self.anneal_lambda(training_step)

        # Instantiation reward component (encourages parallelism)
        instantiation_reward = lambda_aux * r_parallel

        # Task-level outcome component (encourages success)
        task_outcome = (1.0 - lambda_aux) * (success.float() * task_quality)

        # Total reward
        total_reward = instantiation_reward + task_outcome

        return total_reward

    def compute_full_reward(
        self,
        num_subagents: torch.Tensor,
        trajectory_features: torch.Tensor,
        success: torch.Tensor,
        training_step: int,
        max_subagents: int = 100,
    ) -> dict:
        """
        Compute all reward components in one pass

        Args:
            num_subagents: Number of subagents instantiated (batch_size,)
            trajectory_features: Trajectory features (batch_size, feature_dim)
            success: Success indicators (batch_size,)
            training_step: Current training step
            max_subagents: Maximum allowed subagents

        Returns:
            Dictionary with all reward components
        """
        # Compute individual components
        r_parallel = self.compute_instantiation_reward(num_subagents, max_subagents)
        task_quality = self.compute_task_quality(trajectory_features, success)

        # Compute total reward
        total_reward = self.forward(r_parallel, success, task_quality, training_step)

        return {
            "total_reward": total_reward,
            "r_parallel": r_parallel,
            "task_quality": task_quality,
            "lambda_aux": self.anneal_lambda(training_step),
            "instantiation_component": self.anneal_lambda(training_step) * r_parallel,
            "task_component": (1.0 - self.anneal_lambda(training_step))
            * success.float()
            * task_quality,
        }


class CriticalStepsMetric(nn.Module):
    """
    Critical Steps metric for latency-oriented evaluation

    CriticalSteps = Σ(S_main^(t) + max_i S_sub,i^(t))
    """

    def __init__(self, orchestration_overhead: float = 0.1):
        super().__init__()
        self.orchestration_overhead = orchestration_overhead

    def forward(
        self, main_steps: torch.Tensor, sub_steps: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute critical steps

        Args:
            main_steps: Orchestration overhead at each stage (batch_size, num_stages)
            sub_steps: Subagent steps at each stage (batch_size, num_stages, num_subagents)

        Returns:
            Total critical steps (batch_size,)
        """
        # For each stage, find the slowest subagent
        max_sub_steps, _ = sub_steps.max(dim=-1)  # (batch_size, num_stages)

        # Add orchestration overhead
        critical_steps_per_stage = main_steps + max_sub_steps

        # Sum across stages
        total_critical_steps = critical_steps_per_stage.sum(dim=-1)

        return total_critical_steps


# Example usage and testing
if __name__ == "__main__":
    # Initialize reward function
    reward_fn = PARLReward(
        lambda_init=0.1, lambda_final=0.0, total_training_steps=10000
    )

    # Simulate a batch of episodes
    batch_size = 32
    feature_dim = 64

    # Example data
    num_subagents = torch.randint(1, 50, (batch_size,))
    trajectory_features = torch.randn(batch_size, feature_dim)
    success = torch.bernoulli(torch.ones(batch_size) * 0.7)  # 70% success rate

    print("\nTesting reward at different training stages:\n")

    for step in [0, 2500, 5000, 7500, 10000]:
        rewards = reward_fn.compute_full_reward(
            num_subagents=num_subagents,
            trajectory_features=trajectory_features,
            success=success,
            training_step=step,
        )

        print(f"Training Step {step}:")
        print(f"  λ_aux: {rewards['lambda_aux'].item():.4f}")
        print(f"  Avg Total Reward: {rewards['total_reward'].mean().item():.4f}")
        print(f"  Avg r_parallel: {rewards['r_parallel'].mean().item():.4f}")
        print(f"  Avg Task Quality: {rewards['task_quality'].mean().item():.4f}")
        print(
            f"  Instantiation Component: {rewards['instantiation_component'].mean().item():.4f}"
        )
        print(f"  Task Component: {rewards['task_component'].mean().item():.4f}")
        print()

    print("=" * 80)
    print("\nTesting Critical Steps Metric:\n")

    # Initialize metric
    critical_steps_metric = CriticalStepsMetric()

    # Example data: 3 stages with 10 subagents each
    num_stages = 3
    num_subagents_per_stage = 10

    main_steps = torch.ones(batch_size, num_stages) * 0.1  # Orchestration overhead
    sub_steps = torch.rand(batch_size, num_stages, num_subagents_per_stage) * 2.0

    critical_steps = critical_steps_metric(main_steps, sub_steps)

    print(f"Average Critical Steps: {critical_steps.mean().item():.4f}")
    print(f"Min Critical Steps: {critical_steps.min().item():.4f}")
    print(f"Max Critical Steps: {critical_steps.max().item():.4f}")
