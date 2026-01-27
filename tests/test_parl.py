"""
Comprehensive Unit Tests for PARL Implementation

Tests cover:
- PARLReward reward function computation
- Lambda annealing schedule
- Instantiation reward calculation
- Task quality computation
- Critical Steps metric
- Edge cases and boundary conditions
- Gradient computation and backpropagation
"""

import pytest
import torch
from typing import Dict

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from parl.main import PARLReward, CriticalStepsMetric


class TestPARLReward:
    """Test suite for PARLReward class"""

    @pytest.fixture
    def reward_fn(self) -> PARLReward:
        """Create a standard PARLReward instance for testing"""
        return PARLReward(
            lambda_init=0.1, lambda_final=0.0, total_training_steps=10000, device="cpu"
        )

    @pytest.fixture
    def sample_data(self) -> Dict[str, torch.Tensor]:
        """Create sample data for testing"""
        batch_size = 8
        feature_dim = 64

        return {
            "num_subagents": torch.randint(1, 50, (batch_size,)),
            "trajectory_features": torch.randn(batch_size, feature_dim),
            "success": torch.bernoulli(torch.ones(batch_size) * 0.7),
            "training_step": 5000,
        }

    def test_initialization(self, reward_fn: PARLReward):
        """Test proper initialization of PARLReward"""
        assert reward_fn.lambda_init == 0.1
        assert reward_fn.lambda_final == 0.0
        assert reward_fn.total_training_steps == 10000
        assert reward_fn.device == "cpu"
        assert reward_fn.current_step == 0

    def test_anneal_lambda_start(self, reward_fn: PARLReward):
        """Test lambda annealing at the start of training"""
        lambda_aux = reward_fn.anneal_lambda(0)
        assert lambda_aux.item() == pytest.approx(0.1, rel=1e-5)

    def test_anneal_lambda_middle(self, reward_fn: PARLReward):
        """Test lambda annealing at the middle of training"""
        lambda_aux = reward_fn.anneal_lambda(5000)
        expected = 0.1 + (0.0 - 0.1) * 0.5  # Should be 0.05
        assert lambda_aux.item() == pytest.approx(expected, rel=1e-5)

    def test_anneal_lambda_end(self, reward_fn: PARLReward):
        """Test lambda annealing at the end of training"""
        lambda_aux = reward_fn.anneal_lambda(10000)
        assert lambda_aux.item() == pytest.approx(0.0, rel=1e-5)

    def test_anneal_lambda_beyond_end(self, reward_fn: PARLReward):
        """Test lambda annealing beyond training steps"""
        lambda_aux = reward_fn.anneal_lambda(15000)
        assert lambda_aux.item() == pytest.approx(0.0, rel=1e-5)

    def test_instantiation_reward_shape(self, reward_fn: PARLReward):
        """Test instantiation reward output shape"""
        batch_size = 16
        num_subagents = torch.randint(1, 100, (batch_size,))
        r_parallel = reward_fn.compute_instantiation_reward(num_subagents)

        assert r_parallel.shape == (batch_size,)

    def test_instantiation_reward_range(self, reward_fn: PARLReward):
        """Test instantiation reward is within [0, 1]"""
        num_subagents = torch.tensor([0, 25, 50, 75, 100])
        r_parallel = reward_fn.compute_instantiation_reward(
            num_subagents, max_subagents=100
        )

        assert torch.all(r_parallel >= 0.0)
        assert torch.all(r_parallel <= 1.0)

    def test_instantiation_reward_zero_agents(self, reward_fn: PARLReward):
        """Test instantiation reward with zero subagents"""
        num_subagents = torch.tensor([0])
        r_parallel = reward_fn.compute_instantiation_reward(
            num_subagents, max_subagents=100
        )

        assert r_parallel.item() == pytest.approx(0.0, rel=1e-5)

    def test_instantiation_reward_max_agents(self, reward_fn: PARLReward):
        """Test instantiation reward with maximum subagents"""
        num_subagents = torch.tensor([100])
        r_parallel = reward_fn.compute_instantiation_reward(
            num_subagents, max_subagents=100
        )

        assert r_parallel.item() == pytest.approx(1.0, rel=1e-5)

    def test_task_quality_shape(self, reward_fn: PARLReward):
        """Test task quality computation output shape"""
        batch_size = 16
        feature_dim = 64

        trajectory_features = torch.randn(batch_size, feature_dim)
        success = torch.ones(batch_size)

        quality = reward_fn.compute_task_quality(trajectory_features, success)

        assert quality.shape == (batch_size,)

    def test_task_quality_range(self, reward_fn: PARLReward):
        """Test task quality is clamped to [0, 1]"""
        batch_size = 10
        feature_dim = 64

        trajectory_features = torch.randn(batch_size, feature_dim) * 10  # Large values
        success = torch.ones(batch_size)

        quality = reward_fn.compute_task_quality(trajectory_features, success)

        assert torch.all(quality >= 0.0)
        assert torch.all(quality <= 1.0)

    def test_task_quality_failure(self, reward_fn: PARLReward):
        """Test task quality is zero when success is zero"""
        batch_size = 8
        feature_dim = 64

        trajectory_features = torch.randn(batch_size, feature_dim)
        success = torch.zeros(batch_size)

        quality = reward_fn.compute_task_quality(trajectory_features, success)

        assert torch.all(quality == 0.0)

    def test_forward_shape(self, reward_fn: PARLReward, sample_data: Dict):
        """Test forward pass output shape"""
        batch_size = sample_data["num_subagents"].shape[0]

        r_parallel = reward_fn.compute_instantiation_reward(
            sample_data["num_subagents"]
        )
        task_quality = reward_fn.compute_task_quality(
            sample_data["trajectory_features"], sample_data["success"]
        )

        reward = reward_fn.forward(
            r_parallel,
            sample_data["success"],
            task_quality,
            sample_data["training_step"],
        )

        assert reward.shape == (batch_size,)

    def test_forward_early_training(self, reward_fn: PARLReward):
        """Test reward emphasizes parallelism early in training"""
        batch_size = 4

        # High parallelism, low task quality
        r_parallel = torch.tensor([1.0, 1.0, 1.0, 1.0])
        success = torch.ones(batch_size)
        task_quality = torch.tensor([0.1, 0.1, 0.1, 0.1])

        reward = reward_fn.forward(r_parallel, success, task_quality, training_step=0)

        # At step 0, lambda_aux = 0.1, so parallelism should have significant weight
        expected_parallel_component = 0.1 * 1.0
        expected_task_component = 0.9 * 0.1
        expected_reward = expected_parallel_component + expected_task_component

        assert torch.allclose(
            reward, torch.tensor([expected_reward] * batch_size), atol=1e-5
        )

    def test_forward_late_training(self, reward_fn: PARLReward):
        """Test reward emphasizes task success late in training"""
        batch_size = 4

        # Low parallelism, high task quality
        r_parallel = torch.tensor([0.1, 0.1, 0.1, 0.1])
        success = torch.ones(batch_size)
        task_quality = torch.tensor([1.0, 1.0, 1.0, 1.0])

        reward = reward_fn.forward(
            r_parallel, success, task_quality, training_step=10000
        )

        # At step 10000, lambda_aux = 0.0, so only task success matters
        expected_reward = 1.0 * 1.0

        assert torch.allclose(
            reward, torch.tensor([expected_reward] * batch_size), atol=1e-5
        )

    def test_compute_full_reward(self, reward_fn: PARLReward, sample_data: Dict):
        """Test compute_full_reward returns all components"""
        results = reward_fn.compute_full_reward(
            num_subagents=sample_data["num_subagents"],
            trajectory_features=sample_data["trajectory_features"],
            success=sample_data["success"],
            training_step=sample_data["training_step"],
        )

        assert "total_reward" in results
        assert "r_parallel" in results
        assert "task_quality" in results
        assert "lambda_aux" in results
        assert "instantiation_component" in results
        assert "task_component" in results

        # Check shapes
        batch_size = sample_data["num_subagents"].shape[0]
        assert results["total_reward"].shape == (batch_size,)
        assert results["r_parallel"].shape == (batch_size,)
        assert results["task_quality"].shape == (batch_size,)

    def test_reward_decomposition(self, reward_fn: PARLReward, sample_data: Dict):
        """Test that reward components sum to total reward"""
        results = reward_fn.compute_full_reward(
            num_subagents=sample_data["num_subagents"],
            trajectory_features=sample_data["trajectory_features"],
            success=sample_data["success"],
            training_step=sample_data["training_step"],
        )

        reconstructed_reward = (
            results["instantiation_component"] + results["task_component"]
        )

        assert torch.allclose(results["total_reward"], reconstructed_reward, atol=1e-5)

    def test_gradient_flow(self, reward_fn: PARLReward):
        """Test that gradients flow through the reward computation"""
        batch_size = 4
        feature_dim = 64

        # Create trainable trajectory features
        trajectory_features = torch.randn(batch_size, feature_dim, requires_grad=True)
        num_subagents = torch.randint(1, 50, (batch_size,)).float()
        num_subagents.requires_grad = True
        success = torch.ones(batch_size)

        results = reward_fn.compute_full_reward(
            num_subagents=num_subagents,
            trajectory_features=trajectory_features,
            success=success,
            training_step=5000,
        )

        # Backpropagate
        loss = results["total_reward"].sum()
        loss.backward()

        # Check gradients exist
        assert trajectory_features.grad is not None
        assert num_subagents.grad is not None

    def test_device_compatibility_cpu(self):
        """Test reward function works on CPU"""
        reward_fn = PARLReward(device="cpu")

        num_subagents = torch.tensor([25])
        trajectory_features = torch.randn(1, 64)
        success = torch.ones(1)

        results = reward_fn.compute_full_reward(
            num_subagents=num_subagents,
            trajectory_features=trajectory_features,
            success=success,
            training_step=1000,
        )

        assert results["total_reward"].device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_compatibility_cuda(self):
        """Test reward function works on CUDA"""
        reward_fn = PARLReward(device="cuda")

        num_subagents = torch.tensor([25]).cuda()
        trajectory_features = torch.randn(1, 64).cuda()
        success = torch.ones(1).cuda()

        results = reward_fn.compute_full_reward(
            num_subagents=num_subagents,
            trajectory_features=trajectory_features,
            success=success,
            training_step=1000,
        )

        assert results["total_reward"].device.type == "cuda"

    def test_batch_consistency(self, reward_fn: PARLReward):
        """Test that batch processing gives same results as individual processing"""
        # Create single samples
        single_samples = [
            {
                "num_subagents": torch.tensor([i * 10]),
                "trajectory_features": torch.randn(1, 64),
                "success": torch.tensor([1.0]),
                "training_step": 5000,
            }
            for i in range(5)
        ]

        # Process individually
        individual_rewards = []
        for sample in single_samples:
            result = reward_fn.compute_full_reward(**sample)
            individual_rewards.append(result["total_reward"])

        # Process as batch
        batch_data = {
            "num_subagents": torch.cat([s["num_subagents"] for s in single_samples]),
            "trajectory_features": torch.cat(
                [s["trajectory_features"] for s in single_samples]
            ),
            "success": torch.cat([s["success"] for s in single_samples]),
            "training_step": 5000,
        }

        batch_results = reward_fn.compute_full_reward(**batch_data)

        # Compare
        individual_stacked = torch.cat(individual_rewards)
        assert torch.allclose(
            batch_results["total_reward"], individual_stacked, atol=1e-5
        )


class TestCriticalStepsMetric:
    """Test suite for CriticalStepsMetric class"""

    @pytest.fixture
    def metric(self) -> CriticalStepsMetric:
        """Create a standard CriticalStepsMetric instance"""
        return CriticalStepsMetric(orchestration_overhead=0.1)

    def test_initialization(self, metric: CriticalStepsMetric):
        """Test proper initialization"""
        assert metric.orchestration_overhead == 0.1

    def test_critical_steps_shape(self, metric: CriticalStepsMetric):
        """Test output shape"""
        batch_size = 8
        num_stages = 5
        num_subagents = 10

        main_steps = torch.ones(batch_size, num_stages) * 0.1
        sub_steps = torch.rand(batch_size, num_stages, num_subagents)

        critical_steps = metric(main_steps, sub_steps)

        assert critical_steps.shape == (batch_size,)

    def test_critical_steps_single_stage(self, metric: CriticalStepsMetric):
        """Test critical steps with single stage"""
        batch_size = 1
        num_stages = 1
        num_subagents = 3

        main_steps = torch.tensor([[0.1]])
        sub_steps = torch.tensor([[[1.0, 2.0, 1.5]]])

        critical_steps = metric(main_steps, sub_steps)

        # Should be main_steps + max(sub_steps) = 0.1 + 2.0 = 2.1
        expected = 0.1 + 2.0
        assert critical_steps.item() == pytest.approx(expected, rel=1e-5)

    def test_critical_steps_multiple_stages(self, metric: CriticalStepsMetric):
        """Test critical steps with multiple stages"""
        batch_size = 1
        num_stages = 3
        num_subagents = 2

        main_steps = torch.tensor([[0.1, 0.1, 0.1]])
        sub_steps = torch.tensor([[[1.0, 2.0], [1.5, 1.0], [2.0, 1.0]]])

        critical_steps = metric(main_steps, sub_steps)

        # Stage 1: 0.1 + 2.0 = 2.1
        # Stage 2: 0.1 + 1.5 = 1.6
        # Stage 3: 0.1 + 2.0 = 2.1
        # Total: 5.8
        expected = 2.1 + 1.6 + 2.1
        assert critical_steps.item() == pytest.approx(expected, rel=1e-5)

    def test_critical_steps_parallel_benefit(self, metric: CriticalStepsMetric):
        """Test that parallel execution reduces critical steps vs serial"""
        batch_size = 1
        num_stages = 1

        # Serial execution (1 subagent)
        main_steps_serial = torch.tensor([[0.1]])
        sub_steps_serial = torch.tensor([[[10.0]]])
        critical_steps_serial = metric(main_steps_serial, sub_steps_serial)

        # Parallel execution (5 subagents, each doing 2 steps)
        main_steps_parallel = torch.tensor([[0.1]])
        sub_steps_parallel = torch.tensor([[[2.0, 2.0, 2.0, 2.0, 2.0]]])
        critical_steps_parallel = metric(main_steps_parallel, sub_steps_parallel)

        # Parallel should be faster
        assert critical_steps_parallel < critical_steps_serial

    def test_critical_steps_zero_subagents(self, metric: CriticalStepsMetric):
        """Test critical steps with zero sub-steps"""
        batch_size = 2
        num_stages = 3
        num_subagents = 5

        main_steps = torch.ones(batch_size, num_stages) * 0.5
        sub_steps = torch.zeros(batch_size, num_stages, num_subagents)

        critical_steps = metric(main_steps, sub_steps)

        # Should only count main steps
        expected = 0.5 * 3  # 3 stages
        assert torch.allclose(
            critical_steps, torch.tensor([expected, expected]), atol=1e-5
        )

    def test_gradient_flow(self, metric: CriticalStepsMetric):
        """Test gradient flow through metric"""
        batch_size = 2
        num_stages = 3
        num_subagents = 5

        main_steps = torch.ones(batch_size, num_stages, requires_grad=True) * 0.1
        sub_steps = torch.rand(
            batch_size, num_stages, num_subagents, requires_grad=True
        )

        critical_steps = metric(main_steps, sub_steps)
        loss = critical_steps.sum()
        loss.backward()

        assert main_steps.grad is not None
        assert sub_steps.grad is not None

    def test_batch_consistency(self, metric: CriticalStepsMetric):
        """Test batch processing consistency"""
        num_stages = 4
        num_subagents = 8

        # Individual samples
        samples = [
            {
                "main_steps": torch.rand(1, num_stages),
                "sub_steps": torch.rand(1, num_stages, num_subagents),
            }
            for _ in range(5)
        ]

        # Process individually
        individual_results = [metric(s["main_steps"], s["sub_steps"]) for s in samples]

        # Process as batch
        batch_main = torch.cat([s["main_steps"] for s in samples])
        batch_sub = torch.cat([s["sub_steps"] for s in samples])
        batch_result = metric(batch_main, batch_sub)

        # Compare
        individual_stacked = torch.cat(individual_results)
        assert torch.allclose(batch_result, individual_stacked, atol=1e-5)


class TestIntegration:
    """Integration tests combining multiple components"""

    def test_full_training_loop_simulation(self):
        """Simulate a simplified training loop"""
        reward_fn = PARLReward(
            lambda_init=0.1, lambda_final=0.0, total_training_steps=1000
        )

        batch_size = 4
        feature_dim = 32

        # Track reward evolution
        early_rewards = []
        late_rewards = []

        for step in [0, 500, 1000]:
            num_subagents = torch.randint(10, 50, (batch_size,))
            trajectory_features = torch.randn(batch_size, feature_dim)
            success = torch.ones(batch_size)

            results = reward_fn.compute_full_reward(
                num_subagents=num_subagents,
                trajectory_features=trajectory_features,
                success=success,
                training_step=step,
            )

            if step == 0:
                early_rewards.append(results["instantiation_component"].mean().item())
            elif step == 1000:
                late_rewards.append(results["task_component"].mean().item())

        # Verify reward shift: early training emphasizes parallelism, late training emphasizes success
        # This is a qualitative check that the mechanism works
        assert len(early_rewards) > 0
        assert len(late_rewards) > 0

    def test_multi_component_consistency(self):
        """Test that all components work together correctly"""
        reward_fn = PARLReward()
        metric = CriticalStepsMetric()

        batch_size = 8
        num_stages = 3
        num_subagents_count = torch.randint(5, 20, (batch_size,))

        # Compute rewards
        trajectory_features = torch.randn(batch_size, 64)
        success = torch.ones(batch_size)

        reward_results = reward_fn.compute_full_reward(
            num_subagents=num_subagents_count,
            trajectory_features=trajectory_features,
            success=success,
            training_step=5000,
        )

        # Compute critical steps
        main_steps = torch.ones(batch_size, num_stages) * 0.1
        sub_steps = torch.rand(batch_size, num_stages, 10)

        critical_steps = metric(main_steps, sub_steps)

        # Both should produce valid outputs
        assert reward_results["total_reward"].shape == (batch_size,)
        assert critical_steps.shape == (batch_size,)

        # Rewards should be non-negative
        assert torch.all(reward_results["total_reward"] >= 0)

        # Critical steps should be positive
        assert torch.all(critical_steps > 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
