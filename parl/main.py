"""
PARL Reward Function Implementation in PyTorch

Implements the PARL reward from the Kimi K2.5 technical report:

    r_PARL(x,y) = λ1·r_parallel + λ2·r_finish + r_perf(x,y)

where:
- r_parallel: instantiation reward (mitigates serial collapse)
- r_finish: sub-agent finish rate (prevents spurious parallelism)
- r_perf(x,y): task-level outcome (evaluates success and quality of solution)
- λ1 and λ2 are annealed to zero over training so the final policy optimizes r_perf

Requirements:
    pip install torch

This implementation provides:
- Three-term reward with λ1, λ2 annealing
- Instantiation reward (r_parallel) to encourage parallelism
- Finish reward (r_finish) to reward completed subtasks
- Task-level performance (r_perf)
- Critical Steps metric for latency evaluation
- Differentiable components for gradient-based optimization
"""

import torch
import torch.nn as nn


class PARLReward(nn.Module):
    """
    Parallel-Agent Reinforcement Learning Reward Function

    Implements the PARL reward:
    1. r_parallel: incentivizes subagent instantiation (mitigates serial collapse)
    2. r_finish: rewards completed subtasks (prevents spurious parallelism)
    3. r_perf: task-level outcome (primary objective; λ1, λ2 anneal to 0 so this dominates)
    """

    def __init__(
        self,
        lambda1_init: float = 0.1,
        lambda1_final: float = 0.0,
        lambda2_init: float = 0.1,
        lambda2_final: float = 0.0,
        total_training_steps: int = 10000,
        device: str = "cpu",
        *,
        lambda_init: float | None = None,
        lambda_final: float | None = None,
    ):
        super().__init__()
        # Backward compatibility: lambda_init / lambda_final set both λ1 and λ2
        if lambda_init is not None:
            lambda1_init = lambda2_init = lambda_init
        if lambda_final is not None:
            lambda1_final = lambda2_final = lambda_final

        self.lambda1_init = lambda1_init
        self.lambda1_final = lambda1_final
        self.lambda2_init = lambda2_init
        self.lambda2_final = lambda2_final
        self.total_training_steps = total_training_steps
        self.device = device

        self.lambda_init = lambda1_init
        self.lambda_final = lambda1_final

        self.register_buffer("current_step", torch.tensor(0, dtype=torch.long))

    def anneal_lambda1(self, training_step: int) -> torch.Tensor:
        """Anneal λ1 from init → final over training (for r_parallel)."""
        progress = min(1.0, training_step / self.total_training_steps)
        lam = self.lambda1_init + (self.lambda1_final - self.lambda1_init) * progress
        return torch.tensor(lam, dtype=torch.float32, device=self.device)

    def anneal_lambda2(self, training_step: int) -> torch.Tensor:
        """Anneal λ2 from init → final over training (for r_finish)."""
        progress = min(1.0, training_step / self.total_training_steps)
        lam = self.lambda2_init + (self.lambda2_final - self.lambda2_init) * progress
        return torch.tensor(lam, dtype=torch.float32, device=self.device)

    def anneal_lambda(self, training_step: int) -> torch.Tensor:
        """Backward compatibility: returns λ1 (same as anneal_lambda1)."""
        return self.anneal_lambda1(training_step)

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
        normalized_count = num_subagents.float() / max_subagents
        return normalized_count.clamp(0.0, 1.0)

    def compute_finish_reward(
        self,
        completed_subtasks: torch.Tensor,
        assigned_subtasks: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Compute r_finish: sub-agent finish rate (reward for completed subtasks).

        Prevents spurious parallelism: spawning many subagents without meaningful
        task decomposition. Rewards completed subtasks to enforce feasibility and
        guide the policy toward valid decompositions.

        Args:
            completed_subtasks: Number of subtasks completed (batch_size,)
            assigned_subtasks: Number of subtasks assigned (batch_size,)
            eps: Small constant to avoid division by zero

        Returns:
            Finish reward in [0, 1] (batch_size,)
        """
        rate = completed_subtasks.float() / (assigned_subtasks.float() + eps)
        return rate.clamp(0.0, 1.0)

    def compute_task_quality(
        self, trajectory_features: torch.Tensor, success_indicators: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute r_perf: task-level outcome (success and quality of solution y for task x).

        Args:
            trajectory_features: Features extracted from trajectory (batch_size, feature_dim)
            success_indicators: Binary success indicators (batch_size,)

        Returns:
            Task-level performance in [0, 1] (batch_size,)
        """
        quality = trajectory_features.mean(dim=-1)

        # Modulate by success
        quality = quality * success_indicators.float()

        return quality.clamp(0.0, 1.0)

    def forward(
        self,
        r_parallel: torch.Tensor,
        r_finish: torch.Tensor,
        r_perf: torch.Tensor,
        training_step: int,
    ) -> torch.Tensor:
        """
        Compute the full PARL reward:
        r_PARL(x,y) = λ1·r_parallel + λ2·r_finish + r_perf

        λ1 and λ2 are annealed to zero over training so the final policy optimizes r_perf.
        """
        lambda1 = self.anneal_lambda1(training_step)
        lambda2 = self.anneal_lambda2(training_step)
        total_reward = lambda1 * r_parallel + lambda2 * r_finish + r_perf
        return total_reward

    def compute_full_reward(
        self,
        num_subagents: torch.Tensor,
        trajectory_features: torch.Tensor,
        success: torch.Tensor,
        training_step: int,
        max_subagents: int = 100,
        completed_subtasks: torch.Tensor | None = None,
        assigned_subtasks: torch.Tensor | None = None,
    ) -> dict:
        """
        Compute all reward components in one pass.

        Args:
            num_subagents: Number of subagents instantiated (batch_size,)
            trajectory_features: Trajectory features (batch_size, feature_dim)
            success: Success indicators (batch_size,)
            training_step: Current training step
            max_subagents: Maximum allowed subagents
            completed_subtasks: Number of subtasks completed (batch_size,). If None, set to assigned_subtasks so r_finish=1.
            assigned_subtasks: Number of subtasks assigned (batch_size,). If None, set to num_subagents.

        Returns:
            Dictionary with total_reward, r_parallel, r_finish, r_perf, λ1, λ2, and components.
        """
        r_parallel = self.compute_instantiation_reward(num_subagents, max_subagents)
        r_perf = self.compute_task_quality(trajectory_features, success)

        _assigned = (
            assigned_subtasks if assigned_subtasks is not None else num_subagents
        )
        _completed = (
            completed_subtasks if completed_subtasks is not None else _assigned
        )
        r_finish = self.compute_finish_reward(_completed, _assigned)

        total_reward = self.forward(r_parallel, r_finish, r_perf, training_step)
        lambda1 = self.anneal_lambda1(training_step)
        lambda2 = self.anneal_lambda2(training_step)

        return {
            "total_reward": total_reward,
            "r_parallel": r_parallel,
            "r_finish": r_finish,
            "r_perf": r_perf,
            "lambda1": lambda1,
            "lambda2": lambda2,
            "instantiation_component": lambda1 * r_parallel,
            "finish_component": lambda2 * r_finish,
            "task_component": r_perf,
            "task_quality": r_perf,
            "lambda_aux": lambda1,
        }


class CriticalStepsMetric(nn.Module):
    """
    Critical Steps metric for latency-oriented evaluation.

    Per the paper: total critical steps = Σ_t (S_main^(t) + max_i S_sub,i^(t)).
    S_main^(t) is the number of steps taken by the main agent in stage t (typically 1).
    S_sub,i^(t) is the number of steps taken by the i-th subagent in that parallel group.
    The duration of stage t is governed by the longest-running subagent in that cohort.
    """

    def __init__(self, orchestration_overhead: float = 0.1):
        super().__init__()
        # Reserved for future use; pass main_steps in forward (typically 1 per stage).
        self.orchestration_overhead = orchestration_overhead

    def forward(
        self, main_steps: torch.Tensor, sub_steps: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute critical steps: Σ_t (S_main^(t) + max_i S_sub,i^(t)).

        Args:
            main_steps: Main agent steps per stage (batch_size, num_stages). Typically 1 per stage.
            sub_steps: Subagent steps per stage (batch_size, num_stages, num_subagents). For stages with no subagents, use zeros (max_i = 0).

        Returns:
            Total critical steps (batch_size,)
        """
        max_sub_steps = sub_steps.max(dim=-1).values
        # When a stage has no subagents (size 0), max is -inf; treat as 0
        max_sub_steps = max_sub_steps.clamp(min=0.0)
        critical_steps_per_stage = main_steps + max_sub_steps
        return critical_steps_per_stage.sum(dim=-1)
