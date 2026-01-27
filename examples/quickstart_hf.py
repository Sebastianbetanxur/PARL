import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from parl import PARLReward


def main():
    """
    Minimal example: Use PARL reward to train an LLM for parallel task coordination.
    """
    print("Loading Hugging Face model...")

    # 1. Load GLM-4.7-Flash model
    MODEL_PATH = "zai-org/GLM-4.7-Flash"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    device = model.device

    print(f"Model loaded: {MODEL_PATH} on {device}")

    # 2. Initialize PARL reward function
    parl_reward = PARLReward(
        lambda_init=0.1,  # Start with 10% parallelism incentive
        lambda_final=0.0,  # End with 0% (focus on task success)
        total_training_steps=10000,
        device=device,
    )

    print("PARL reward function initialized")

    # 3. Simulate a training episode
    task = "Process 1000 API requests in parallel"
    print(f"\nTask: {task}")

    # Create prompt for the model using chat template
    messages = [
        {
            "role": "user",
            "content": f"Task: {task}\nHow many parallel agents should we use? Answer with only a number between 1 and 100.",
        }
    ]

    # Generate model prediction
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
        )

    # Parse model output
    generated_text = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
    )
    print(f"\nModel output: {generated_text}")

    # Extract predicted number of agents (simplified parsing)
    import re

    numbers = re.findall(r"\d+", generated_text)
    num_subagents = int(numbers[0]) if numbers else 10
    num_subagents = max(1, min(num_subagents, 100))  # Clamp to [1, 100]

    print(f"Predicted subagents: {num_subagents}")

    # 4. Simulate task execution (in practice, this would be real execution)
    # For demo: success probability increases with more agents
    success_prob = min(0.5 + (num_subagents / 100) * 0.5, 0.95)
    success = torch.bernoulli(torch.tensor([success_prob]))

    # Create trajectory features (execution metrics)
    trajectory_features = torch.tensor(
        [
            [
                num_subagents / 100,  # Normalized agent count
                success.item(),  # Task success
                0.8,  # Example: execution efficiency
            ]
        ]
    ).to(device)

    print(f"Success: {'Yes' if success.item() > 0 else 'No'}")

    # 5. Compute PARL reward
    training_step = 5000  # Mid-training

    rewards = parl_reward.compute_full_reward(
        num_subagents=torch.tensor([num_subagents]),
        trajectory_features=trajectory_features,
        success=success,
        training_step=training_step,
        max_subagents=100,
    )

    # 6. Display reward components
    print("\n" + "=" * 60)
    print("PARL Reward Breakdown")
    print("=" * 60)
    print(f"λ_aux (parallelism weight): {rewards['lambda_aux'].item():.4f}")
    print(f"Total reward:              {rewards['total_reward'].item():.4f}")
    print(f"Parallelism component:     {rewards['instantiation_component'].item():.4f}")
    print(f"Task success component:    {rewards['task_component'].item():.4f}")
    print("=" * 60)

    # 7. In a real training loop, you would:
    print("\nNext steps for full training:")
    print("1. Compute policy gradient: grad = reward * log_prob(action)")
    print("2. Update model parameters: optimizer.step()")
    print("3. Repeat for many episodes")
    print("4. Lambda automatically anneals from 0.1 → 0.0 over training")

    # Example: Show reward evolution
    print("\n" + "=" * 60)
    print("Reward Evolution Across Training")
    print("=" * 60)

    for step in [0, 2500, 5000, 7500, 10000]:
        r = parl_reward.compute_full_reward(
            num_subagents=torch.tensor([num_subagents]),
            trajectory_features=trajectory_features,
            success=success,
            training_step=step,
            max_subagents=100,
        )

        print(
            f"Step {step:5d}: λ={r['lambda_aux'].item():.3f}, "
            f"Total={r['total_reward'].item():.4f}, "
            f"Parallel={r['instantiation_component'].item():.4f}, "
            f"Task={r['task_component'].item():.4f}"
        )

    print("\nObserve how:")
    print("- Early training (step 0): High parallelism weight (λ=0.1)")
    print("- Late training (step 10000): Focus on task success (λ=0.0)")


if __name__ == "__main__":
    main()
