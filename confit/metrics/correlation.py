"""Correlation metrics for fitness prediction evaluation.

Single source of truth — eliminates the duplicated, diverged ``spearman()``
and ``compute_stat()`` functions that existed in both ``data_utils.py`` and
``stat_utils.py``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import bootstrap, spearmanr


def spearman(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute Spearman rank correlation, with a zero-variance guard.

    Returns 0.0 when either array has near-zero variance (degenerate case
    that would otherwise produce NaN from scipy).

    Args:
        y_pred: Predicted scores, shape ``(N,)``.
        y_true: Ground-truth scores, shape ``(N,)``.

    Returns:
        Spearman ρ in [-1, 1], or 0.0 for degenerate inputs.
    """
    if np.var(y_pred) < 1e-6 or np.var(y_true) < 1e-6:
        return 0.0
    return float(spearmanr(y_pred, y_true)[0])


def compute_stat(
    sr: np.ndarray,
) -> Tuple[float, float, list]:
    """Compute mean, std, and bootstrap 95% confidence interval of Spearman scores.

    Args:
        sr: Array of per-seed Spearman correlation values.

    Returns:
        Tuple of ``(mean, std, confidence_interval)`` where
        ``confidence_interval`` is a two-element list ``[lower, upper]``.
    """
    sr = np.asarray(sr)
    mean = float(np.mean(sr))
    std = float(np.std(sr))
    ci = list(bootstrap((sr,), np.mean).confidence_interval)
    return mean, std, ci