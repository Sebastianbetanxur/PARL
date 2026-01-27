"""
Basic Usage Example for PARL

This script demonstrates the basic usage of the PARL reward function
and Critical Steps metric.
"""

import torch

from parl import CriticalStepsMetric, PARLReward


def example_reward_computation():
    """
    Demonstrate basic reward computation with PARL.
    """
    print("=" * 80)
    print("PARL Reward Computation Example")
    print("=" * 80)

    # Initialize the reward function
    reward_fn = PARLReward(
        lambda_init=0.1,
        lambda_final=0.0,
        total_training_steps=10000,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print(f"\nDevice: {reward_fn.device}")
    print(f"Lambda range: {reward_fn.lambda_init} -> {reward_fn.lambda_final}")
    print(f"Training steps: {reward_fn.total_training_steps}")

    # Create sample episode data
    batch_size = 4
    num_subagents = torch.tensor([10, 25, 50, 75])  # Different parallelism levels
    trajectory_features = torch.randn(batch_size, 64)
    success = torch.tensor([1.0, 1.0, 1.0, 0.0])  # Last episode failed

    print(f"\nBatch size: {batch_size}")
    print(f"Number of subagents: {num_subagents.tolist()}")
    print(f"Success indicators: {success.tolist()}")

    # Compute rewards at different training stages
    print("\n" + "-" * 80)
    print("Reward Evolution Across Training")
    print("-" * 80)

    for training_step in [0, 2500, 5000, 7500, 10000]:
        rewards = reward_fn.compute_full_reward(
            num_subagents=num_subagents,
            trajectory_features=trajectory_features,
            success=success,
            training_step=training_step,
            max_subagents=100,
        )

        print(f"\nTraining Step: {training_step}")
        print(f"  λ_aux: {rewards['lambda_aux'].item():.4f}")
        print(f"  Total Reward: {rewards['total_reward'].mean().item():.4f}")
        print(
            f"  Parallelism Component: {rewards['instantiation_component'].mean().item():.4f}"
        )
        print(
            f"  Task Success Component: {rewards['task_component'].mean().item():.4f}"
        )

        # Show per-episode breakdown
        print("  Per-episode total rewards:", end=" ")
        for i, r in enumerate(rewards["total_reward"]):
            print(f"{r.item():.3f}", end=" ")
        print()


def example_critical_steps():
    """
    Demonstrate Critical Steps metric computation.
    """
    print("\n" + "=" * 80)
    print("Critical Steps Metric Example")
    print("=" * 80)

    # Initialize metric
    metric = CriticalStepsMetric(orchestration_overhead=0.1)

    batch_size = 2
    num_stages = 4
    num_subagents = 8

    # Simulate parallel workflow
    # Stage represents a decomposition step, subagents work in parallel
    main_steps = torch.ones(batch_size, num_stages) * 0.1  # Orchestration overhead
    sub_steps = torch.tensor(
        [
            # Episode 1: Varying parallelism per stage
            [
                [2.0, 1.5, 1.0, 1.2, 1.8, 1.3, 1.1, 1.4],  # Stage 1
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # Stage 2 (balanced)
                [3.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # Stage 3 (one slow agent)
                [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],  # Stage 4
            ],
            # Episode 2: Different pattern
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [2.5, 2.0, 1.5, 1.0, 1.2, 1.8, 1.3, 1.1],
                [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
                [2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
        ]
    )

    print(f"\nBatch size: {batch_size}")
    print(f"Number of stages: {num_stages}")
    print(f"Subagents per stage: {num_subagents}")

    # Compute critical steps
    critical_steps = metric(main_steps, sub_steps)

    print("\n" + "-" * 80)
    print("Critical Steps Analysis")
    print("-" * 80)

    for episode_idx in range(batch_size):
        print(f"\nEpisode {episode_idx + 1}:")
        episode_critical_steps = 0

        for stage_idx in range(num_stages):
            stage_main = main_steps[episode_idx, stage_idx].item()
            stage_sub_max = sub_steps[episode_idx, stage_idx].max().item()
            stage_total = stage_main + stage_sub_max

            print(f"  Stage {stage_idx + 1}:")
            print(f"    Orchestration: {stage_main:.2f}")
            print(f"    Max subagent time: {stage_sub_max:.2f}")
            print(f"    Stage critical steps: {stage_total:.2f}")

            episode_critical_steps += stage_total

        print(f"  Total critical steps: {episode_critical_steps:.2f}")
        print(f"  (Computed: {critical_steps[episode_idx].item():.2f})")

    # Compare with serial execution
    total_steps_serial = sub_steps.sum(dim=-1).sum(dim=-1) + main_steps.sum(dim=-1)
    print("\n" + "-" * 80)
    print("Parallel vs Serial Comparison")
    print("-" * 80)

    for episode_idx in range(batch_size):
        speedup = (
            total_steps_serial[episode_idx].item() / critical_steps[episode_idx].item()
        )
        print(f"\nEpisode {episode_idx + 1}:")
        print(f"  Serial execution: {total_steps_serial[episode_idx].item():.2f} steps")
        print(
            f"  Parallel execution: {critical_steps[episode_idx].item():.2f} critical steps"
        )
        print(f"  Speedup: {speedup:.2f}x")


def example_training_simulation():
    """
    Simulate a simple training loop showing reward evolution.
    """
    print("\n" + "=" * 80)
    print("Training Loop Simulation")
    print("=" * 80)

    reward_fn = PARLReward(lambda_init=0.1, lambda_final=0.0, total_training_steps=1000)

    batch_size = 8

    print("\nSimulating 1000 training steps...")
    print("Tracking average reward components over time\n")

    checkpoints = [0, 250, 500, 750, 1000]

    for step in checkpoints:
        # Simulate episodes with varying parallelism and success
        num_subagents = torch.randint(20, 80, (batch_size,))
        trajectory_features = torch.randn(batch_size, 64)

        # Success rate improves over training (simulated)
        success_rate = min(0.5 + step / 2000, 1.0)
        success = torch.bernoulli(torch.ones(batch_size) * success_rate)

        rewards = reward_fn.compute_full_reward(
            num_subagents=num_subagents,
            trajectory_features=trajectory_features,
            success=success,
            training_step=step,
        )

        print(
            f"Step {step:4d} | "
            f"λ={rewards['lambda_aux'].item():.3f} | "
            f"Total={rewards['total_reward'].mean().item():.3f} | "
            f"Parallel={rewards['instantiation_component'].mean().item():.3f} | "
            f"Task={rewards['task_component'].mean().item():.3f} | "
            f"Success={success.mean().item():.2f}"
        )

    print("\nObservations:")
    print("- Lambda (λ) decreases from 0.1 to 0.0 over training")
    print("- Parallel component weight decreases over time")
    print("- Task success component weight increases over time")
    print("- This encourages exploration of parallelism early, then focuses on quality")


if __name__ == "__main__":
    # Run all examples
    example_reward_computation()
    example_critical_steps()
    example_training_simulation()

    print("\n" + "=" * 80)
    print("Examples completed successfully!")
    print("=" * 80)
