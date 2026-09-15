# Contributing to SentinelEdge Core

Thank you for your interest in contributing to the **Intel Physical AI Challenge** project! This guide covers everything you need to get started.

---

## Table of Contents

1. [Setting Up the Development Environment](#1-setting-up-the-development-environment)
2. [Running Tests](#2-running-tests)
3. [Code Style Guidelines](#3-code-style-guidelines)
4. [Recording a New Teleoperation Dataset Episode](#4-recording-a-new-teleoperation-dataset-episode)
5. [Adding a New Task to task_suite.py](#5-adding-a-new-task-to-task_suitepy)
6. [Pull Request Guidelines](#6-pull-request-guidelines)

---

## 1. Setting Up the Development Environment

We use **Conda** to manage dependencies for reproducibility.

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/)
- Git

### Steps

```bash
# Clone the repository
git clone https://github.com/your-org/sentineledge-core.git
cd sentineledge-core

# Create and activate the conda environment
conda env create -f environment.yml
conda activate intel-vla-challenge

# Verify the setup with smoke tests
python -c 'from simulation.env import BimanualDinnerEnv; print("Env import OK")'
python -c 'from models.vla_policy import get_policy; m = get_policy(); print("Policy import OK")'
python -c 'from scripts.metrics import MetricsLogger; print("Metrics import OK")'
```

### Updating the Environment

If `environment.yml` changes after you have already created the environment:

```bash
conda env update -f environment.yml --prune
```

---

## 2. Running Tests

### Smoke Tests (Fast)

These verify that all core modules can be imported successfully:

```bash
python -c 'from simulation.env import BimanualDinnerEnv; print("Env import OK")'
python -c 'from models.vla_policy import get_policy; m = get_policy(); print("Policy import OK")'
python -c 'from scripts.metrics import MetricsLogger; print("Metrics import OK")'
```

### Full Evaluation

Run a complete evaluation across 10 random seeds to benchmark policy performance:

```bash
python scripts/evaluate.py --seeds 10
```

### CI Pipeline

All smoke tests are automatically run on every push and pull request via GitHub Actions. Check the **Actions** tab on GitHub to see results.

---

## 3. Code Style Guidelines

### PEP 8

All Python code must conform to [PEP 8](https://pep8.org/). Use a linter before committing:

```bash
pip install flake8
flake8 . --max-line-length=100 --exclude=.git,__pycache__,build,dist
```

### Docstrings

Every public function, class, and module **must** include a docstring. We follow the [Google docstring style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings):

```python
def compute_reward(achieved: bool, time_elapsed: float) -> float:
    """Compute the shaped reward for a task episode.

    Args:
        achieved: Whether the task goal was successfully completed.
        time_elapsed: Wall-clock seconds elapsed since episode start.

    Returns:
        A scalar reward value in the range [0.0, 1.0].
    """
    ...
```

### Type Hints

Use type hints for all function signatures:

```python
def load_config(path: str) -> dict:
    ...
```

### Imports

Organize imports in three groups (stdlib -> third-party -> local), separated by blank lines:

```python
import os
import sys

import numpy as np
import torch

from simulation.env import BimanualDinnerEnv
```

---

## 4. Recording a New Teleoperation Dataset Episode

Use the teleoperation script to capture a new demonstration episode.

### Prerequisites

- Two SO-101 robot arms connected and calibrated
- The `intel-vla-challenge` conda environment active

### Steps

```bash
# Start a recording session (replace TASK_NAME with e.g. "dinner_table_setup")
python scripts/record_episode.py \
    --task TASK_NAME \
    --output data/episodes/ \
    --num-episodes 5

# The script will:
# 1. Launch the MuJoCo simulation (or real robot interface)
# 2. Wait for you to perform the bimanual task
# 3. Save observations, actions, and metadata to data/episodes/<TASK_NAME>_<timestamp>.hdf5
```

### Dataset Format

Each HDF5 file contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `observations/qpos` | `(T, 14)` | Joint positions for both arms |
| `observations/images/top` | `(T, H, W, 3)` | Top-view RGB frames |
| `actions` | `(T, 14)` | Recorded joint velocity commands |
| `metadata/task` | string | Task name |
| `metadata/seed` | int | Random seed used |

---

## 5. Adding a New Task to task_suite.py

Tasks are defined in `scripts/task_suite.py`. Each task is a Python dataclass describing goals, object placements, and reward functions.

### Step-by-Step

1. **Open** `scripts/task_suite.py`.

2. **Define your task** by subclassing `BaseTask`:

```python
from scripts.task_suite import BaseTask, register_task

@register_task("my_new_task")
class MyNewTask(BaseTask):
    """Stack a cup on a plate using both arms.

    Args:
        seed: Random seed for domain randomization.
    """

    name: str = "my_new_task"
    description: str = "Stack a cup on a plate."

    def get_initial_state(self, seed: int) -> dict:
        """Return randomized initial object placements."""
        rng = np.random.default_rng(seed)
        return {
            "cup_pos": rng.uniform([0.3, -0.1, 0.8], [0.5, 0.1, 0.9]),
            "plate_pos": rng.uniform([-0.5, -0.1, 0.8], [-0.3, 0.1, 0.9]),
        }

    def compute_reward(self, obs: dict) -> float:
        """Return shaped reward based on cup-on-plate proximity."""
        ...

    def is_success(self, obs: dict) -> bool:
        """Return True when cup is stably placed on plate."""
        ...
```

3. **Register** the task: the `@register_task("my_new_task")` decorator automatically adds it to the global task registry.

4. **Write a smoke test** in your PR description showing that the task can be instantiated and reset:

```python
from scripts.task_suite import get_task
task = get_task("my_new_task")
state = task.get_initial_state(seed=42)
print(state)
```

5. **Add domain randomization** parameters to `simulation/env.py` if your task introduces new objects.

---

## 6. Pull Request Guidelines

### Before Opening a PR

- [ ] All smoke tests pass locally
- [ ] Code conforms to PEP 8 (`flake8` returns no errors)
- [ ] All new public functions/classes have docstrings
- [ ] `environment.yml` is updated if you added new dependencies
- [ ] `CHALLENGE_CHECKLIST.md` statuses are updated if a deliverable changed

### PR Title Format

Use the following prefixes:

| Prefix | When to use |
|--------|-------------|
| `feat:` | New feature or task |
| `fix:` | Bug fix |
| `docs:` | Documentation only |
| `refactor:` | Code restructuring without behaviour change |
| `ci:` | CI/CD pipeline changes |
| `data:` | Dataset or recording changes |

Example: `feat: add stacking task to task_suite`

### PR Description Template

```markdown
## Summary
Brief description of what this PR does.

## Changes
- Added `MyNewTask` to `scripts/task_suite.py`
- Updated domain randomization parameters in `simulation/env.py`

## Testing
Paste smoke test output here.

## Checklist
- [ ] Smoke tests pass
- [ ] PEP 8 compliant
- [ ] Docstrings added
```

### Review Process

1. At least **one reviewer** must approve before merging.
2. All CI checks must be **green**.
3. Squash-merge into `main` with a descriptive commit message.

---

*Happy hacking! 🤖*
