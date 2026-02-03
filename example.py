import torch

from parl import CriticalStepsMetric, PARLReward

# Example usage and testing
if __name__ == "__main__":
    # PARL reward: r_PARL = λ1·r_parallel + λ2·r_finish + r_perf (λ1, λ2 anneal to 0)
    reward_fn = PARLReward(
        lambda_init=0.1, lambda_final=0.0, total_training_steps=10000
    )

    batch_size = 32
    feature_dim = 64

    num_subagents = torch.randint(1, 50, (batch_size,))
    assigned_subtasks = num_subagents * torch.randint(2, 5, (batch_size,))
    completed_subtasks = (assigned_subtasks.float() * 0.85).long().clamp(
        max=assigned_subtasks
    )  # ~85% finish rate so r_finish < 1
    trajectory_features = torch.randn(batch_size, feature_dim)
    success = torch.bernoulli(torch.ones(batch_size) * 0.7)

    print("\nPARL reward at different training stages (λ1, λ2 → 0, so total → r_perf):\n")

    for step in [0, 2500, 5000, 7500, 10000]:
        rewards = reward_fn.compute_full_reward(
            num_subagents=num_subagents,
            trajectory_features=trajectory_features,
            success=success,
            training_step=step,
            completed_subtasks=completed_subtasks,
            assigned_subtasks=assigned_subtasks,
        )

        print(f"Training Step {step}:")
        print(f"  λ1: {rewards['lambda1'].item():.4f}, λ2: {rewards['lambda2'].item():.4f}")
        print(f"  Avg Total Reward: {rewards['total_reward'].mean().item():.4f}")
        print(f"  r_parallel: {rewards['r_parallel'].mean().item():.4f}, r_finish: {rewards['r_finish'].mean().item():.4f}, r_perf: {rewards['r_perf'].mean().item():.4f}")
        print(
            f"  Instantiation: {rewards['instantiation_component'].mean().item():.4f} | Finish: {rewards['finish_component'].mean().item():.4f} | Task: {rewards['task_component'].mean().item():.4f}"
        )
        print()

    print("→ At step 10000: λ1=λ2=0, so total reward = r_perf only (primary objective).")
    print()
    print("=" * 80)
    print("\nCritical Steps Metric (Σ_t (S_main^(t) + max_i S_sub,i^(t))):\n")

    # Initialize metric
    critical_steps_metric = CriticalStepsMetric()

    # Example data: 3 stages with 10 subagents each
    num_stages = 3
    num_subagents_per_stage = 10

    main_steps = torch.ones(batch_size, num_stages)  # S_main^(t) typically 1 per stage
    sub_steps = torch.rand(batch_size, num_stages, num_subagents_per_stage) * 2.0

    critical_steps = critical_steps_metric(main_steps, sub_steps)

    print(f"Average Critical Steps: {critical_steps.mean().item():.4f}")
    print(f"Min Critical Steps: {critical_steps.min().item():.4f}")
    print(f"Max Critical Steps: {critical_steps.max().item():.4f}")
