"""Metrics subpackage — single source of truth for correlation statistics."""

from psifit.metrics.correlation import spearman, compute_stat

__all__ = ["spearman", "compute_stat"]