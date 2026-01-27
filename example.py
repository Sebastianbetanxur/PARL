import torch

from parl import CriticalStepsMetric, PARLReward

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
