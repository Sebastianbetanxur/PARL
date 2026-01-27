# PARL: Parallel-Agent Reinforcement Learning

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

> **⚠️ Disclaimer**: This is an **open-source community implementation** of PARL (Parallel-Agent Reinforcement Learning) based on the Kimi K2.5 technical report. This is **NOT an official implementation** from Kimi AI or any affiliated organization. This project is maintained independently by The Swarm Corporation and the open-source community.

Open-source implementation of **PARL (Parallel-Agent Reinforcement Learning)**, a novel training paradigm that enables AI models to decompose complex tasks into parallel subtasks and coordinate multiple agents simultaneously.

## Overview

PARL is a training methodology that addresses the critical challenge of **serial collapse** in multi-agent systems, where models default to sequential execution despite having parallel computational capacity. By implementing staged reward shaping and a latency-oriented evaluation metric, PARL trains models to efficiently orchestrate up to 100 sub-agents across 1,500+ coordinated steps.

### Key Features

- **Staged Reward Shaping**: Dynamic reward annealing that encourages parallelism early in training and gradually shifts focus toward task success
- **Instantiation Reward**: Incentivizes subagent creation and concurrent execution
- **Critical Steps Metric**: Latency-oriented evaluation inspired by parallel computation's critical path concept
- **Differentiable Components**: Fully compatible with gradient-based optimization
- **Orchestrator-Subagent Architecture**: Trainable coordinator with frozen execution agents

## Architecture

```
┌─────────────────────────────────────────────┐
│         Orchestrator Agent                  │
│  (Trainable Central Coordinator)            │
│  - Decomposes tasks into subtasks           │
│  - Manages parallel execution               │
│  - Coordinates subagent workflows           │
└──────────────┬──────────────────────────────┘
               │
               ├──────────┬──────────┬─────────┐
               │          │          │         │
          ┌────▼───┐ ┌───▼────┐ ┌──▼────┐  ┌─▼──────┐
          │Subagent│ │Subagent│ │Subagent│  │Subagent│
          │   1    │ │   2    │ │   3    │  │  ...N  │
          └────────┘ └────────┘ └────────┘  └────────┘
           (Frozen)   (Frozen)   (Frozen)    (Frozen)
```

## Reward Function

PARL implements a two-component reward structure:

```
R_t = λ_aux(e) · r_parallel + (1 - λ_aux(e)) · (𝟙[success] · Q(τ))
```

Where:
- `λ_aux(e)`: Anneals from 0.1 → 0.0 over training
- `r_parallel`: Instantiation reward encouraging parallelism
- `𝟙[success]`: Binary success indicator
- `Q(τ)`: End-to-end task quality metric

### Critical Steps Metric

Instead of counting total steps, PARL uses a latency-oriented metric:

```
CriticalSteps = Σ(S_main^(t) + max_i S_sub,i^(t))
```

This metric captures the true execution time considering parallel operations.

## Installation


```bash
pip3 install -U open-parl
```

## Quick Start

```python
import torch
from parl import PARLReward, CriticalStepsMetric

# Initialize the reward function
reward_fn = PARLReward(
    lambda_init=0.1,
    lambda_final=0.0,
    total_training_steps=10000,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Prepare episode data
num_subagents = torch.tensor([25, 30, 40])  # Number of subagents per episode
trajectory_features = torch.randn(3, 64)     # Trajectory features
success = torch.tensor([1.0, 1.0, 0.0])      # Success indicators
training_step = 5000                          # Current training step

# Compute rewards
rewards = reward_fn.compute_full_reward(
    num_subagents=num_subagents,
    trajectory_features=trajectory_features,
    success=success,
    training_step=training_step,
    max_subagents=100
)

print(f"Total Reward: {rewards['total_reward']}")
print(f"Lambda (λ_aux): {rewards['lambda_aux']:.4f}")
print(f"Parallelism Component: {rewards['instantiation_component']}")
print(f"Task Success Component: {rewards['task_component']}")

# Evaluate using Critical Steps metric
critical_steps_metric = CriticalStepsMetric()

main_steps = torch.ones(3, 5) * 0.1  # Orchestration overhead
sub_steps = torch.rand(3, 5, 10)      # Subagent steps

critical_steps = critical_steps_metric(main_steps, sub_steps)
print(f"Critical Steps: {critical_steps}")
```

## Examples

| Example | File | Description | Requirements |
|---------|------|-------------|-------------|
| **Basic Usage** | `basic_usage.py` | Core PARL reward function and Critical Steps metric without external models | `open-parl` only |
| **Quick Start (HF)** | `quickstart_hf.py` | Minimal integration with Hugging Face GLM-4.7-Flash model | `transformers`, `torch`, `accelerate` |
| **Full HF Integration** | `huggingface_integration.py` | Complete orchestrator class with training loop, batching, and reward analysis | `transformers`, `torch`, `accelerate` |

## API Reference

For detailed API documentation, see [docs/API.md](docs/API.md).

## Experiments

Run the example training simulation:

```bash
python -m parl.main
```

This will demonstrate reward evolution across training stages and critical steps computation.

## Testing

Run the comprehensive test suite:

```bash
# Using pytest
pytest tests/ -v

# With coverage report
pytest tests/ --cov=parl --cov-report=html

# Run specific test file
pytest tests/test_parl.py -v
```

## Research Paper

This is an **unofficial open-source implementation** based on the technical report:

> **"PARL: Parallel-Agent Reinforcement Learning for Large Language Models"**
> Kimi AI Research Team, 2026

For technical details and experimental results from the original research, see: [Kimi K2.5 Technical Report](https://www.kimi.com/blog/kimi-k2-5.html)

**Note**: This implementation is not affiliated with, endorsed by, or officially connected to Kimi AI. It is an independent open-source project developed by the community based on publicly available information.

## Citation

If you use PARL in your research, please cite:

```bibtex
@article{parl2026,
  title={PARL: Parallel-Agent Reinforcement Learning for Large Language Models},
  author={Kimi AI Research Team},
  journal={Technical Report},
  year={2026},
  url={https://www.kimi.com/blog/kimi-k2-5.html}
}
```



## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code passes all tests and follows PEP 8 style guidelines.


## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This is an **unofficial open-source implementation** inspired by the Kimi K2.5 technical report
- Built on PyTorch's efficient tensor operations
- Thanks to the open-source ML community
- This project is not affiliated with or endorsed by Kimi AI

## Contact

- **Repository**: [github.com/The-Swarm-Corporation/PARL](https://github.com/The-Swarm-Corporation/PARL)
- **Issues**: [github.com/The-Swarm-Corporation/PARL/issues](https://github.com/The-Swarm-Corporation/PARL/issues)

---

**Made with ⚡ by The Swarm Corporation**
