"""
PARL: Parallel-Agent Reinforcement Learning

A training paradigm that teaches models to decompose complex tasks into parallel
subtasks and coordinate multiple agents simultaneously.

Reward (per Kimi K2.5 report): r_PARL = λ1·r_parallel + λ2·r_finish + r_perf.
λ1, λ2 anneal to zero so the final policy optimizes r_perf.

This package provides:
- PARLReward: Three-term reward (r_parallel, r_finish, r_perf) with λ1/λ2 annealing
- CriticalStepsMetric: Critical steps = Σ_t (S_main^(t) + max_i S_sub,i^(t))

Example:
    >>> import torch
    >>> from parl import PARLReward, CriticalStepsMetric
    >>>
    >>> reward_fn = PARLReward()
    >>> rewards = reward_fn.compute_full_reward(
    ...     num_subagents=torch.tensor([25]),
    ...     trajectory_features=torch.randn(1, 64),
    ...     success=torch.tensor([1.0]),
    ...     training_step=5000
    ... )
    >>> print(rewards['total_reward'])
"""

from parl.main import CriticalStepsMetric, PARLReward

__version__ = "0.1.0"
__author__ = "The Swarm Corporation"
__license__ = "Apache-2.0"

__all__ = ["PARLReward", "CriticalStepsMetric"]
