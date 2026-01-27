"""
PARL: Parallel-Agent Reinforcement Learning

A training paradigm that teaches models to decompose complex tasks into parallel
subtasks and coordinate multiple agents simultaneously.

This package provides:
- PARLReward: Staged reward shaping function for parallel agent training
- CriticalStepsMetric: Latency-oriented evaluation metric

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

from parl.main import PARLReward, CriticalStepsMetric

__version__ = "0.1.0"
__author__ = "The Swarm Corporation"
__license__ = "Apache-2.0"

__all__ = ["PARLReward", "CriticalStepsMetric"]
