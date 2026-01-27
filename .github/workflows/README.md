# GitHub Actions Workflows

This directory contains all CI/CD workflows for the PARL project.

## Workflows Overview

### 1. **ci.yml** - Main CI Pipeline
Runs on every push and pull request to `main` and `develop` branches.

**Jobs:**
- **Quality Check**: Runs Ruff, Black, and isort to ensure code quality
- **Tests**: Runs pytest across multiple OS (Ubuntu, macOS, Windows) and Python versions (3.9-3.12)
- **Build**: Creates distribution packages

### 2. **lint.yml** - Comprehensive Linting
Dedicated linting workflow with all code quality tools.

**Checks:**
- Ruff linter and formatter
- Black code formatter
- isort import sorting
- mypy type checking

### 3. **ruff.yml** - Ruff Only
Fast workflow that only runs Ruff checks. Useful for quick feedback.

### 4. **black.yml** - Black Only
Dedicated Black code formatting checks.

### 5. **tests.yml** (Existing)
Original test workflow with comprehensive testing and linting.

### 6. **publish.yml** - PyPI Publishing
Publishes the package to PyPI on releases or manual trigger.

**Features:**
- Automatic publishing on GitHub releases
- Manual workflow dispatch with option to publish to Test PyPI
- Creates GitHub release assets

## Setup Instructions

### PyPI Publishing Setup

To enable automatic PyPI publishing, you need to configure the following secrets in your GitHub repository:

1. **For PyPI (Production)**
   - Go to https://pypi.org/manage/account/token/
   - Create a new API token
   - Add it to GitHub Secrets as `PYPI_API_TOKEN`

2. **For Test PyPI (Testing)**
   - Go to https://test.pypi.org/manage/account/token/
   - Create a new API token
   - Add it to GitHub Secrets as `TEST_PYPI_API_TOKEN`

3. **Add Secrets to GitHub**
   - Go to your repository settings
   - Navigate to `Settings > Secrets and variables > Actions`
   - Click "New repository secret"
   - Add `PYPI_API_TOKEN` and `TEST_PYPI_API_TOKEN`

### Codecov Setup (Optional)

For coverage reporting:
1. Go to https://codecov.io/
2. Connect your GitHub repository
3. Get the Codecov token
4. Add it as `CODECOV_TOKEN` in GitHub Secrets

## Publishing to PyPI

### Automatic Publishing (Recommended)
1. Create a new release on GitHub
2. The `publish.yml` workflow will automatically build and publish to PyPI
3. Distribution files will be attached to the GitHub release

### Manual Publishing
1. Go to the "Actions" tab
2. Select "Publish to PyPI" workflow
3. Click "Run workflow"
4. Choose whether to publish to Test PyPI or production PyPI

## Development Workflow

### Before Committing
Run these commands locally:
```bash
# Format code
poetry run black parl/ tests/ examples/
poetry run isort parl/ tests/ examples/

# Lint
poetry run ruff check parl/ tests/ examples/ --fix

# Type check
poetry run mypy parl/

# Run tests
poetry run pytest tests/
```

### Pre-commit Hooks (Recommended)
Install pre-commit hooks to automatically run checks:
```bash
poetry run pre-commit install
```

## Workflow Status Badges

Add these badges to your README.md:

```markdown
[![CI](https://github.com/The-Swarm-Corporation/PARL/actions/workflows/ci.yml/badge.svg)](https://github.com/The-Swarm-Corporation/PARL/actions/workflows/ci.yml)
[![Ruff](https://github.com/The-Swarm-Corporation/PARL/actions/workflows/ruff.yml/badge.svg)](https://github.com/The-Swarm-Corporation/PARL/actions/workflows/ruff.yml)
[![Black](https://github.com/The-Swarm-Corporation/PARL/actions/workflows/black.yml/badge.svg)](https://github.com/The-Swarm-Corporation/PARL/actions/workflows/black.yml)
[![PyPI](https://img.shields.io/pypi/v/open-parl.svg)](https://pypi.org/project/open-parl/)
[![Python Versions](https://img.shields.io/pypi/pyversions/open-parl.svg)](https://pypi.org/project/open-parl/)
```

## Troubleshooting

### Poetry Lock File Issues
If you encounter poetry.lock issues:
```bash
poetry lock --no-update
git add poetry.lock
git commit -m "Update poetry.lock"
```

### Test Failures
- Check the Actions tab for detailed logs
- Run tests locally: `poetry run pytest tests/ -v`
- Ensure all dependencies are installed: `poetry install --with dev`

### Publishing Failures
- Verify your PyPI tokens are correct
- Check that the version in `pyproject.toml` is unique
- Ensure the package builds successfully: `poetry build`
