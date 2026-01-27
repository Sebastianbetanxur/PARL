# Contributing to PARL

Thank you for your interest in contributing to PARL (Parallel-Agent Reinforcement Learning)! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. Please be considerate and professional in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/PARL.git
   cd PARL
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/The-Swarm-Corporation/PARL.git
   ```

## Development Setup

### Using Poetry (Recommended)

```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

### Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

### Install Pre-commit Hooks

```bash
poetry run pre-commit install
```

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

- **Bug fixes**: Fix issues and improve stability
- **New features**: Add new functionality
- **Documentation**: Improve or add documentation
- **Tests**: Add or improve test coverage
- **Examples**: Create usage examples and tutorials
- **Performance**: Optimize existing code

### Contribution Workflow

1. **Create a branch** for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** with clear, focused commits

3. **Write or update tests** for your changes

4. **Run tests** to ensure everything works:
   ```bash
   poetry run pytest tests/ -v
   ```

5. **Format your code**:
   ```bash
   poetry run black parl/ tests/
   poetry run isort parl/ tests/
   ```

6. **Check for issues**:
   ```bash
   poetry run flake8 parl/ tests/
   poetry run mypy parl/
   ```

7. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add feature description"
   ```

8. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

9. **Open a Pull Request** on GitHub

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line length**: 88 characters (Black default)
- **Imports**: Organized with isort
- **Type hints**: Use type annotations where appropriate
- **Docstrings**: Google-style docstrings for all public APIs

### Example Docstring

```python
def compute_reward(
    r_parallel: torch.Tensor,
    success: torch.Tensor,
    task_quality: torch.Tensor,
    training_step: int
) -> torch.Tensor:
    """
    Compute the PARL reward.

    Args:
        r_parallel: Instantiation reward (batch_size,)
        success: Success indicators (batch_size,)
        task_quality: Task quality scores (batch_size,)
        training_step: Current training step

    Returns:
        Total reward (batch_size,)

    Example:
        >>> r_parallel = torch.tensor([0.5, 0.7])
        >>> success = torch.tensor([1.0, 1.0])
        >>> task_quality = torch.tensor([0.8, 0.9])
        >>> reward = compute_reward(r_parallel, success, task_quality, 5000)
    """
    # Implementation here
```

### Commit Message Convention

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions or modifications
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

Example:
```
feat: add support for heterogeneous subagent architectures

- Implement subagent type specification
- Add tests for mixed agent types
- Update documentation
```

## Testing

### Running Tests

```bash
# Run all tests
poetry run pytest tests/ -v

# Run specific test file
poetry run pytest tests/test_parl.py -v

# Run with coverage
poetry run pytest tests/ --cov=parl --cov-report=html

# Run specific test
poetry run pytest tests/test_parl.py::TestPARLReward::test_anneal_lambda_start -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files with `test_` prefix
- Use descriptive test names that explain what is being tested
- Include edge cases and boundary conditions
- Aim for high test coverage (>90%)

### Test Structure

```python
def test_feature_description():
    """Test that feature works correctly under normal conditions"""
    # Arrange
    input_data = create_test_data()

    # Act
    result = function_under_test(input_data)

    # Assert
    assert result.shape == expected_shape
    assert torch.allclose(result, expected_value)
```

## Pull Request Process

### Before Submitting

- [ ] Tests pass locally
- [ ] Code is formatted (Black, isort)
- [ ] No linting errors (flake8, mypy)
- [ ] Documentation is updated
- [ ] Commit messages follow conventions
- [ ] Branch is up to date with main

### PR Description Template

```markdown
## Description
Brief description of changes

## Motivation
Why is this change necessary?

## Changes
- Change 1
- Change 2

## Testing
How was this tested?

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Code formatted
- [ ] No linting errors
```

### Review Process

1. Automated tests will run on your PR
2. A maintainer will review your code
3. Address any feedback or requested changes
4. Once approved, your PR will be merged

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

- **Description**: Clear description of the bug
- **Steps to reproduce**: Minimal steps to reproduce the issue
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Environment**: Python version, PyTorch version, OS
- **Error messages**: Full error traceback if applicable

### Feature Requests

When requesting features, please include:

- **Use case**: Why is this feature needed?
- **Proposed solution**: How should it work?
- **Alternatives**: Other approaches considered
- **Additional context**: Any relevant information

## Development Tips

### Debugging

```python
# Enable PyTorch anomaly detection
torch.autograd.set_detect_anomaly(True)

# Add breakpoints
import pdb; pdb.set_trace()
```

### Performance Profiling

```python
import torch.profiler as profiler

with profiler.profile() as prof:
    # Your code here
    pass

print(prof.key_averages().table(sort_by="cpu_time_total"))
```

## Questions?

- Open an issue for questions
- Check existing issues and PRs
- Read the documentation

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.

---

Thank you for contributing to PARL!
