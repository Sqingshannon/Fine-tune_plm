"""Reproducibility utilities — deterministic seeding for all RNG sources."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(sample_seed: int, model_seed: int) -> None:
    """Set all relevant RNG seeds for fully reproducible training runs.

    Sets seeds for Python ``random``, NumPy, and PyTorch (CPU + all GPUs).
    Also enables cuDNN deterministic mode and disables the benchmark flag.

    Args:
        sample_seed: Seed for data sampling (Python ``random`` + NumPy).
        model_seed: Seed for model weight initialisation (PyTorch).

    Example:
        >>> seed_everything(sample_seed=0, model_seed=1)
    """
    random.seed(sample_seed)
    np.random.seed(sample_seed)
    torch.manual_seed(model_seed)
    torch.cuda.manual_seed_all(model_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False