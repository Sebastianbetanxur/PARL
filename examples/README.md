# PARL Examples

This directory contains examples demonstrating how to use PARL (Parallel-Agent Reinforcement Learning) with various configurations and models.

## Examples

### 1. Basic Usage (`basic_usage.py`)

Demonstrates the core PARL reward function and Critical Steps metric without any external models.

**Run:**
```bash
python basic_usage.py
```

**What it shows:**
- Reward computation across training stages
- Lambda annealing from 0.1 → 0.0
- Critical Steps metric calculation
- Parallel vs serial execution comparison

### 2. Quick Start with Hugging Face (`quickstart_hf.py`)

Minimal example integrating PARL with the GLM-4.7-Flash model from Hugging Face.

**Requirements:**
```bash
pip install -r requirements.txt
```

**Run:**
```bash
python quickstart_hf.py
```

**What it shows:**
- Loading GLM-4.7-Flash model
- Using PARL reward for parallelization decisions
- Computing rewards based on model predictions
- Reward evolution across training steps

### 3. Full Hugging Face Integration (`huggingface_integration.py`)

Comprehensive example with a complete orchestrator class that demonstrates production-ready integration.

**Requirements:**
```bash
pip install -r requirements.txt
```

**Run:**
```bash
python huggingface_integration.py
```

**What it shows:**
- Complete `PARLOrchestrator` class
- Task decomposition using GLM-4.7-Flash
- Simulated training loop
- Batched processing
- Reward analysis across parallelization levels

**Examples included:**
- `example_basic_usage()` - Basic task decomposition
- `example_training_loop()` - Simulated RL training
- `example_with_batching()` - Batch processing
- `example_reward_analysis()` - Performance analysis

## Model Configuration

All Hugging Face examples use the **GLM-4.7-Flash** model from `zai-org/GLM-4.7-Flash`.

### Why GLM-4.7-Flash?

- Fast inference (optimized for speed)
- Strong reasoning capabilities
- Good at following instructions
- Efficient memory usage

### Using Different Models

You can easily swap in different models by changing the `model_name` parameter:

```python
# Using a different model
orchestrator = PARLOrchestrator(
    model_name="meta-llama/Llama-2-7b-hf",  # Example: LLaMA
)
```

**Compatible models:**
- Any Hugging Face causal language model
- Models with chat template support work best
- Recommended: Instruction-tuned models (e.g., LLaMA, Mistral, etc.)

## Installation

### Full Installation

Install all dependencies for running the Hugging Face examples:

```bash
pip install -r requirements.txt
```

### Basic Installation

For running only the basic example without models:

```bash
pip install open-parl
```

## Understanding PARL Rewards

PARL implements staged reward shaping:

```
R_t = λ_aux(e) · r_parallel + (1 - λ_aux(e)) · (𝟙[success] · Q(τ))
```

**Components:**

1. **λ_aux(e)**: Anneals from 0.1 → 0.0 over training
   - Early training: Encourages parallelism exploration
   - Late training: Focuses on task success

2. **r_parallel**: Instantiation reward
   - Incentivizes spawning more subagents
   - Normalized by max capacity

3. **Q(τ)**: Task quality metric
   - Measures end-to-end success
   - Modulated by success indicator

4. **Critical Steps**: Latency-oriented metric
   - `CriticalSteps = Σ(S_main^(t) + max_i S_sub,i^(t))`
   - Captures true execution time with parallelism

## Example Output

### Quick Start Output

```
Model loaded: zai-org/GLM-4.7-Flash on cuda
PARL reward function initialized

Task: Process 1000 API requests in parallel
Model output: 50

Predicted subagents: 50
Success: Yes

PARL Reward Breakdown
Lambda (λ_aux): 0.0500
Total reward: 0.4523
Parallelism component: 0.0250
Task success component: 0.3998
```

### Training Loop Output

```
Episode    Step     λ        Agents   Success    Reward     Critical Steps
1          0        0.100    25       Yes        0.3245     2.15
5          2500     0.075    40       Yes        0.4567     1.82
10         5000     0.050    50       Yes        0.5123     1.65
15         7500     0.025    60       Yes        0.5891     1.52
20         10000    0.000    75       Yes        0.6234     1.43
```

## Key Concepts

### Lambda Annealing

The auxiliary reward weight λ_aux decreases over training:
- **Step 0**: λ = 0.1 (10% weight on parallelism)
- **Step 5000**: λ = 0.05 (5% weight on parallelism)
- **Step 10000**: λ = 0.0 (0% weight, full focus on task success)

### Critical Steps Metric

Unlike total steps, Critical Steps accounts for parallel execution:

**Serial execution:**
```
Total Steps = Sum of all subagent steps = 100 steps
```

**Parallel execution (10 agents):**
```
Critical Steps = Orchestration + Max(subagent steps) = 15 steps
Speedup = 100 / 15 = 6.7x
```

## Advanced Usage

### Custom Trajectory Features

Extend trajectory features with domain-specific metrics:

```python
trajectory_features = torch.tensor([[
    execution_metrics["critical_steps"].item(),
    num_subagents / max_subagents,
    execution_metrics["success"].item(),
    # Add custom features:
    memory_usage / max_memory,
    network_latency,
    cache_hit_rate,
]])
```

### Custom Reward Functions

Extend the PARL reward with domain-specific bonuses:

```python
def compute_custom_reward(self, base_reward, metrics):
    # Add efficiency bonus
    efficiency_bonus = 1.0 / (1.0 + metrics["critical_steps"])

    # Add cost penalty
    cost_penalty = -0.01 * metrics["num_subagents"]

    # Combine
    total_reward = base_reward + efficiency_bonus + cost_penalty

    return total_reward
```

### Production Training Loop

For production RL training, implement:

1. **Policy Gradients**:
   ```python
   log_probs = model.get_log_probs(actions)
   loss = -(rewards * log_probs).mean()
   loss.backward()
   optimizer.step()
   ```

2. **Baseline Subtraction** (reduce variance):
   ```python
   advantages = rewards - rewards.mean()
   loss = -(advantages * log_probs).mean()
   ```

3. **PPO Clipping** (stable updates):
   ```python
   ratio = torch.exp(log_probs - old_log_probs)
   clipped_ratio = torch.clamp(ratio, 1-epsilon, 1+epsilon)
   loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
   ```

## Troubleshooting

### CUDA Out of Memory

If you encounter OOM errors:

```python
# Use a smaller model
orchestrator = PARLOrchestrator(
    model_name="gpt2",  # Smaller model
)

# Or use CPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
```

### Slow Inference

For faster inference:

```python
# Use Flash Attention
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",  # Requires flash-attn package
)
```

### Model Not Following Instructions

Try adjusting the prompt or using a better instruction-tuned model:

```python
messages = [
    {
        "role": "system",
        "content": "You are an expert at decomposing tasks into parallel subtasks."
    },
    {
        "role": "user",
        "content": f"Task: {task}\nHow many parallel agents (1-100)? Answer with only a number."
    }
]
```

## Next Steps

1. **Experiment with different models** - Try LLaMA, Mistral, or other models
2. **Implement full RL training** - Add PPO or A2C training loop
3. **Fine-tune on your domain** - Collect task-specific data and fine-tune
4. **Deploy in production** - Integrate with your multi-agent system
5. **Benchmark performance** - Compare against serial baselines

## Resources

- [PARL Repository](https://github.com/The-Swarm-Corporation/PARL)
- [Kimi K2.5 Technical Report](https://www.kimi.com/blog/kimi-k2-5.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PyTorch RL Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)

## Support

For issues or questions:
- Open an issue: https://github.com/The-Swarm-Corporation/PARL/issues
- Check existing examples and documentation
- Review the test suite in `tests/` directory
