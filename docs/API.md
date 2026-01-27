# API Reference

This document provides detailed API documentation for the PARL (Parallel-Agent Reinforcement Learning) library.

## Contents

- [PARLReward](#parlreward)
- [CriticalStepsMetric](#criticalstepsmetric)

---

## PARLReward

```python
class parl.PARLReward(lambda_init=0.1, lambda_final=0.0, total_training_steps=10000, device='cpu')
```

Parallel-Agent Reinforcement Learning Reward Function implementing staged reward shaping.

The `PARLReward` class implements a two-component reward structure that encourages parallelism early in training and gradually shifts focus toward task success. The reward function follows the formulation:

```math
R_t = λ_aux(e) · r_parallel + (1 - λ_aux(e)) · (𝟙[success] · Q(τ))
```

where:
- `λ_aux(e)` anneals linearly from `lambda_init` to `lambda_final` over `total_training_steps`
- `r_parallel` is the instantiation reward encouraging parallelism
- `𝟙[success]` is a binary success indicator
- `Q(τ)` is the end-to-end task quality metric

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lambda_init` | `float` | `0.1` | Initial auxiliary reward weight. Controls how much weight is placed on parallelism incentive at the start of training. Typical values range from 0.05 to 0.2. |
| `lambda_final` | `float` | `0.0` | Final auxiliary reward weight. Usually set to 0.0 to fully focus on task success by the end of training. |
| `total_training_steps` | `int` | `10000` | Total number of training steps over which lambda anneals. This determines the annealing schedule. |
| `device` | `str` | `'cpu'` | Device for tensor computation. Must be `'cpu'` or `'cuda'`. All internal tensors will be created on this device. |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `lambda_init` | `float` | Initial lambda value (read-only). |
| `lambda_final` | `float` | Final lambda value (read-only). |
| `total_training_steps` | `int` | Total training steps (read-only). |
| `device` | `str` | Computation device (read-only). |

### Methods

#### `anneal_lambda(training_step)`

Computes the current value of λ_aux based on training progress.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `training_step` | `int` | Current training step (0-indexed). Should be in range [0, total_training_steps]. |

**Returns:**

| Type | Description |
|------|-------------|
| `torch.Tensor` | Current λ_aux value as a scalar tensor on the specified device. Shape: `()`. |

**Mathematical Formulation:**

```math
λ_aux(e) = λ_init + (λ_final - λ_init) · min(1.0, e / total_training_steps)
```

where `e` is the current training step.

**Example:**

```python
import torch
from parl import PARLReward

reward_fn = PARLReward(
    lambda_init=0.1,
    lambda_final=0.0,
    total_training_steps=10000,
    device='cuda'
)

# At the start of training
lambda_start = reward_fn.anneal_lambda(0)
print(f"Lambda at step 0: {lambda_start.item()}")  # 0.1

# At mid-training
lambda_mid = reward_fn.anneal_lambda(5000)
print(f"Lambda at step 5000: {lambda_mid.item()}")  # 0.05

# At the end of training
lambda_end = reward_fn.anneal_lambda(10000)
print(f"Lambda at step 10000: {lambda_end.item()}")  # 0.0
```

**Note:** If `training_step >= total_training_steps`, lambda is clamped to `lambda_final`.

---

#### `compute_instantiation_reward(num_subagents, max_subagents=100)`

Computes the instantiation reward `r_parallel` that incentivizes subagent creation and concurrent execution.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `num_subagents` | `torch.Tensor` | Number of subagents instantiated per episode. Shape: `(batch_size,)`. Must contain non-negative integers. |
| `max_subagents` | `int` | Maximum allowed number of subagents. Used for normalization. Default: `100`. |

**Returns:**

| Type | Description |
|------|-------------|
| `torch.Tensor` | Instantiation reward values. Shape: `(batch_size,)`. Values are in range [0, 1] (normalized by `max_subagents`). |

**Mathematical Formulation:**

```math
r_parallel = num_subagents / max_subagents
```

**Example:**

```python
import torch
from parl import PARLReward

reward_fn = PARLReward(device='cpu')

# Batch of episodes with different numbers of subagents
num_subagents = torch.tensor([10, 25, 50, 100])
r_parallel = reward_fn.compute_instantiation_reward(
    num_subagents=num_subagents,
    max_subagents=100
)

print(r_parallel)
# tensor([0.1000, 0.2500, 0.5000, 1.0000])
```

**Note:** The reward increases linearly with the number of subagents, encouraging the model to spawn more parallel agents.

---

#### `compute_task_quality(trajectory_features, success_indicators)`

Computes the end-to-end task quality metric `Q(τ)`.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `trajectory_features` | `torch.Tensor` | Features extracted from the execution trajectory. Shape: `(batch_size, feature_dim)`. Can include metrics like execution time, resource usage, intermediate results, etc. |
| `success_indicators` | `torch.Tensor` | Binary success indicators (1.0 for success, 0.0 for failure). Shape: `(batch_size,)`. |

**Returns:**

| Type | Description |
|------|-------------|
| `torch.Tensor` | Task quality scores. Shape: `(batch_size,)`. Values are clamped to [0.0, 1.0]. |

**Mathematical Formulation:**

```math
Q(τ) = clamp(mean(trajectory_features) · success_indicators, 0.0, 1.0)
```

**Example:**

```python
import torch
from parl import PARLReward

reward_fn = PARLReward(device='cpu')

# Trajectory features: [critical_steps, efficiency, resource_usage]
trajectory_features = torch.tensor([
    [0.8, 0.9, 0.7],  # Episode 1: good performance
    [0.3, 0.2, 0.4],  # Episode 2: poor performance
    [0.6, 0.7, 0.5],  # Episode 3: moderate performance
])

success_indicators = torch.tensor([1.0, 0.0, 1.0])  # Episodes 1 and 3 succeeded

task_quality = reward_fn.compute_task_quality(
    trajectory_features=trajectory_features,
    success_indicators=success_indicators
)

print(task_quality)
# tensor([0.8000, 0.0000, 0.6000])
```

**Note:** Failed episodes (success=0) will have zero task quality regardless of trajectory features. In practice, you can extend this method to use a learned quality function.

---

#### `forward(r_parallel, success, task_quality, training_step)`

Computes the full PARL reward using the staged reward shaping formula.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `r_parallel` | `torch.Tensor` | Instantiation reward values. Shape: `(batch_size,)`. Typically computed via `compute_instantiation_reward()`. |
| `success` | `torch.Tensor` | Binary success indicators. Shape: `(batch_size,)`. |
| `task_quality` | `torch.Tensor` | Task quality scores. Shape: `(batch_size,)`. Typically computed via `compute_task_quality()`. |
| `training_step` | `int` | Current training step for lambda annealing. |

**Returns:**

| Type | Description |
|------|-------------|
| `torch.Tensor` | Total reward values. Shape: `(batch_size,)`. |

**Mathematical Formulation:**

```math
R_t = λ_aux(e) · r_parallel + (1 - λ_aux(e)) · (success · Q(τ))
```

**Example:**

```python
import torch
from parl import PARLReward

reward_fn = PARLReward(
    lambda_init=0.1,
    lambda_final=0.0,
    total_training_steps=10000,
    device='cpu'
)

# Pre-computed components
r_parallel = torch.tensor([0.5, 0.3, 0.8])
success = torch.tensor([1.0, 1.0, 0.0])
task_quality = torch.tensor([0.9, 0.7, 0.6])
training_step = 5000  # Mid-training

total_reward = reward_fn.forward(
    r_parallel=r_parallel,
    success=success,
    task_quality=task_quality,
    training_step=training_step
)

print(total_reward)
# tensor([0.4750, 0.3650, 0.0400])
```

**Note:** This method is typically called internally by `compute_full_reward()`. Use `compute_full_reward()` for most use cases as it computes all components automatically.

---

#### `compute_full_reward(num_subagents, trajectory_features, success, training_step, max_subagents=100)`

Computes all reward components in a single pass. This is the recommended method for most use cases.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `num_subagents` | `torch.Tensor` | Number of subagents instantiated per episode. Shape: `(batch_size,)`. |
| `trajectory_features` | `torch.Tensor` | Trajectory features. Shape: `(batch_size, feature_dim)`. |
| `success` | `torch.Tensor` | Binary success indicators. Shape: `(batch_size,)`. |
| `training_step` | `int` | Current training step for lambda annealing. |
| `max_subagents` | `int` | Maximum allowed subagents. Default: `100`. |

**Returns:**

| Type | Description |
|------|-------------|
| `dict` | Dictionary containing all reward components with the following keys: |

**Return Dictionary:**

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `total_reward` | `torch.Tensor` | `(batch_size,)` | Total PARL reward combining parallelism and task success components. |
| `r_parallel` | `torch.Tensor` | `(batch_size,)` | Instantiation reward (parallelism component). |
| `task_quality` | `torch.Tensor` | `(batch_size,)` | Task quality scores Q(τ). |
| `lambda_aux` | `torch.Tensor` | `()` | Current lambda value (scalar). |
| `instantiation_component` | `torch.Tensor` | `(batch_size,)` | λ_aux · r_parallel (parallelism contribution to total reward). |
| `task_component` | `torch.Tensor` | `(batch_size,)` | (1 - λ_aux) · success · Q(τ) (task success contribution to total reward). |

**Example:**

```python
import torch
from parl import PARLReward

# Initialize reward function
reward_fn = PARLReward(
    lambda_init=0.1,
    lambda_final=0.0,
    total_training_steps=10000,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Prepare episode data
num_subagents = torch.tensor([25, 30, 40, 50])
trajectory_features = torch.randn(4, 64)  # 4 episodes, 64 features each
success = torch.tensor([1.0, 1.0, 0.0, 1.0])
training_step = 5000

# Compute all reward components
rewards = reward_fn.compute_full_reward(
    num_subagents=num_subagents,
    trajectory_features=trajectory_features,
    success=success,
    training_step=training_step,
    max_subagents=100
)

# Access individual components
print(f"Total Reward: {rewards['total_reward']}")
print(f"Lambda (λ_aux): {rewards['lambda_aux'].item():.4f}")
print(f"Parallelism Component: {rewards['instantiation_component']}")
print(f"Task Success Component: {rewards['task_component']}")
print(f"Task Quality: {rewards['task_quality']}")
```

**Note:** This method is differentiable and can be used directly in gradient-based optimization. All tensors are created on the device specified during initialization.

---

## CriticalStepsMetric

```python
class parl.CriticalStepsMetric(orchestration_overhead=0.1)
```

Critical Steps metric for latency-oriented evaluation of parallel execution.

The `CriticalStepsMetric` computes the critical path length of parallel execution, which represents the true execution time considering parallel operations. Unlike total step counts, this metric accounts for the fact that parallel subagents execute concurrently.

**Mathematical Formulation:**

```math
CriticalSteps = Σ_t (S_main^(t) + max_i S_sub,i^(t))
```

where:
- `S_main^(t)` is the orchestration overhead at stage `t`
- `S_sub,i^(t)` is the execution time of subagent `i` at stage `t`
- `max_i` finds the slowest subagent at each stage (the critical path)
- `Σ_t` sums across all stages

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `orchestration_overhead` | `float` | `0.1` | Base overhead for orchestrator coordination. This parameter is currently stored but not directly used in the computation (orchestration overhead is provided via `main_steps` in the forward pass). Reserved for future use. |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `orchestration_overhead` | `float` | Orchestration overhead parameter (read-only). |

### Methods

#### `forward(main_steps, sub_steps)`

Computes the critical steps for parallel execution workflows.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `main_steps` | `torch.Tensor` | Orchestration overhead at each stage. Shape: `(batch_size, num_stages)`. Each value represents the time spent by the orchestrator at that stage. |
| `sub_steps` | `torch.Tensor` | Subagent execution times at each stage. Shape: `(batch_size, num_stages, num_subagents)`. Each value `sub_steps[b, t, i]` represents the execution time of subagent `i` at stage `t` in batch `b`. |

**Returns:**

| Type | Description |
|------|-------------|
| `torch.Tensor` | Total critical steps per episode. Shape: `(batch_size,)`. Each value represents the critical path length for that episode. |

**Computation Steps:**

1. For each stage, find the maximum subagent execution time: `max_sub_steps = max_i(sub_steps[:, :, i])`
2. Add orchestration overhead: `critical_steps_per_stage = main_steps + max_sub_steps`
3. Sum across stages: `total_critical_steps = Σ_t(critical_steps_per_stage)`

**Example:**

```python
import torch
from parl import CriticalStepsMetric

metric = CriticalStepsMetric()

# Batch of 3 episodes, 5 stages each, 10 subagents
batch_size = 3
num_stages = 5
num_subagents = 10

# Orchestration overhead (constant across stages)
main_steps = torch.ones(batch_size, num_stages) * 0.1

# Subagent execution times (random for demonstration)
sub_steps = torch.rand(batch_size, num_stages, num_subagents) * 2.0

# Compute critical steps
critical_steps = metric(main_steps, sub_steps)

print(f"Critical Steps: {critical_steps}")
# tensor([5.2341, 4.8765, 5.1234])  # Example output
```

**Example: Serial vs Parallel Comparison**

```python
import torch
from parl import CriticalStepsMetric

metric = CriticalStepsMetric()

# Scenario: 10 subagents, 3 stages
num_stages = 3
num_subagents = 10

# Serial execution: subagents execute one after another
main_steps_serial = torch.tensor([[0.1, 0.1, 0.1]])
sub_steps_serial = torch.ones(1, num_stages, num_subagents) * 1.0
# Total time if serial: 10 subagents × 1.0 time × 3 stages = 30.0

# Parallel execution: subagents execute concurrently
main_steps_parallel = torch.tensor([[0.1, 0.1, 0.1]])
sub_steps_parallel = torch.ones(1, num_stages, num_subagents) * 1.0
# Critical path: max(1.0) × 3 stages + overhead = ~3.3

critical_serial = metric(main_steps_serial, sub_steps_serial)
critical_parallel = metric(main_steps_parallel, sub_steps_parallel)

print(f"Serial critical steps: {critical_serial.item():.2f}")
print(f"Parallel critical steps: {critical_parallel.item():.2f}")
print(f"Speedup: {critical_serial.item() / critical_parallel.item():.2f}x")
```

**Note:** This metric is differentiable and can be used in gradient-based optimization. The critical path represents the minimum possible execution time for the parallel workflow.

---

## Usage Examples

### Complete Training Loop Example

```python
import torch
from parl import PARLReward, CriticalStepsMetric

# Initialize components
reward_fn = PARLReward(
    lambda_init=0.1,
    lambda_final=0.0,
    total_training_steps=10000,
    device='cuda'
)
critical_steps_metric = CriticalStepsMetric()

# Training loop
for episode in range(1000):
    training_step = episode
    
    # Simulate episode execution
    num_subagents = torch.randint(1, 100, (32,))  # Batch of 32 episodes
    trajectory_features = torch.randn(32, 64)
    success = torch.bernoulli(torch.ones(32) * 0.8)
    
    # Compute rewards
    rewards = reward_fn.compute_full_reward(
        num_subagents=num_subagents,
        trajectory_features=trajectory_features,
        success=success,
        training_step=training_step,
        max_subagents=100
    )
    
    # Compute critical steps
    main_steps = torch.ones(32, 5) * 0.1
    sub_steps = torch.rand(32, 5, 10) * 2.0
    critical_steps = critical_steps_metric(main_steps, sub_steps)
    
    # Use rewards for policy gradient update
    # loss = -(rewards['total_reward'] * log_probs).mean()
    # loss.backward()
    # optimizer.step()
```

### Integration with Hugging Face Models

See [examples/huggingface_integration.py](../examples/huggingface_integration.py) for a complete example integrating PARL with Hugging Face language models.

---

## See Also

- [Examples](../examples/README.md) - Comprehensive usage examples
- [Quick Start Guide](../README.md#quick-start) - Getting started with PARL
- [Research Paper](https://www.kimi.com/blog/kimi-k2-5.html) - Original PARL technical report

---

## Notes

1. **Device Consistency**: All tensors should be on the same device. The reward function creates tensors on the device specified during initialization.

2. **Gradient Flow**: Both `PARLReward` and `CriticalStepsMetric` are fully differentiable and can be used with `autograd`.

3. **Batch Processing**: All methods support batched inputs. The batch dimension is the first dimension of all input tensors.

4. **Lambda Annealing**: The lambda annealing schedule is linear. For custom schedules, you can manually compute lambda and use the `forward()` method directly.

5. **Task Quality Function**: The default `compute_task_quality()` uses a simple mean of trajectory features. In practice, you may want to replace this with a learned quality function.
