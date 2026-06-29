"""Inference aggregator — ensemble averaging and optional retrieval blending.

Replaces the script-level code in the original ``inference.py`` that ran
at import time and had no ``if __name__`` guard.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from psifit.metrics.correlation import spearman


class InferenceAggregator:
    """Aggregates per-seed predictions and computes ensemble Spearman correlation.

    Supports an optional VAE-ELBO retrieval blend::

        retrieval_score = alpha * ensemble_avg + (1 - alpha) * elbo

    Args:
        predicted_dir: Root directory containing per-dataset prediction folders.
        data_dir: Root directory containing per-dataset ``test.csv`` files.
        results_dir: Root directory where ``summary.csv`` is written.
        alpha: Blending weight for retrieval (default 0.8).

    Example:
        >>> agg = InferenceAggregator(
        ...     predicted_dir=Path("predicted"),
        ...     data_dir=Path("data"),
        ...     results_dir=Path("results"),
        ... )
        >>> sr = agg.aggregate("PTEN_HUMAN", shot=96, use_retrieval=False)
    """

    def __init__(
        self,
        predicted_dir: Path = Path("predicted"),
        data_dir: Path = Path("data"),
        results_dir: Path = Path("results"),
        alpha: float = 0.8,
    ) -> None:
        self.predicted_dir = Path(predicted_dir)
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.alpha = alpha

    def aggregate(
        self,
        dataset: str,
        shot: int,
        use_retrieval: bool = True,
    ) -> Optional[float]:
        """Aggregate per-seed predictions and compute Spearman correlation.

        Reads ``predicted/<dataset>/pred.csv``, averages across seed columns,
        merges with ground-truth from ``data/<dataset>/test.csv``, and writes
        the result to ``results/<dataset>/summary.csv``.

        Args:
            dataset: Dataset identifier string.
            shot: Training shot count (stored in the output summary).
            use_retrieval: When ``True``, blend ensemble scores with VAE-ELBO
                scores from ``data/<dataset>/vae_elbo.csv``.

        Returns:
            Spearman ρ, or ``None`` if prediction file does not exist.
        """
        pred_path = self.predicted_dir / dataset / "pred.csv"
        if not pred_path.exists():
            return None

        summary = self._load_existing_summary(dataset)
        pred = pd.read_csv(pred_path, index_col=0)
        pred = pred.drop_duplicates(subset="PID")

        seed_cols = self._detect_seed_columns(pred)
        ensemble_avg = pred[seed_cols].mean(axis=1)
        pred = pred.copy()
        pred["avg"] = ensemble_avg

        test = pd.read_csv(self.data_dir / dataset / "test.csv", index_col=0)
        perf = pd.merge(pred[["avg", "PID"]], test[["PID", "log_fitness"]], on="PID")

        if use_retrieval:
            sr = self._blend_with_retrieval(perf, dataset)
        else:
            sr = self._score_ensemble(perf)

        out = pd.DataFrame({"spearman": sr, "shot": shot}, index=[dataset])
        summary = pd.concat([summary, out], axis=0)

        self._write_summary(dataset, summary)
        return sr

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_existing_summary(self, dataset: str) -> pd.DataFrame:
        """Load an existing summary CSV or return an empty DataFrame."""
        path = self.results_dir / dataset / "summary.csv"
        if path.exists():
            return pd.read_csv(path, index_col=0)
        return pd.DataFrame()

    @staticmethod
    def _detect_seed_columns(pred: pd.DataFrame) -> List[str]:
        """Return column names that are seed indices (``'1'`` – ``'5'``)."""
        return [c for c in pred.columns if c in [str(i) for i in range(1, 6)]]

    def _blend_with_retrieval(self, perf: pd.DataFrame, dataset: str) -> float:
        """Blend ensemble average with VAE-ELBO retrieval scores.

        Args:
            perf: DataFrame with ``avg`` and ``log_fitness`` columns.
            dataset: Dataset identifier (used to locate ``vae_elbo.csv``).

        Returns:
            Spearman ρ of the blended score against ground truth.
        """
        elbo = pd.read_csv(self.data_dir / dataset / "vae_elbo.csv", index_col=0)
        perf = pd.merge(perf, elbo, on="PID")
        perf["retrieval"] = (
            self.alpha * perf["avg"] + (1 - self.alpha) * perf["elbo"]
        )
        scores = np.asarray(perf["retrieval"])
        gscores = np.asarray(perf["log_fitness"])
        return spearman(scores, gscores)

    @staticmethod
    def _score_ensemble(perf: pd.DataFrame) -> float:
        """Compute Spearman ρ of ensemble average against ground truth."""
        scores = np.asarray(perf["avg"])
        gscores = np.asarray(perf["log_fitness"])
        return spearman(scores, gscores)

    def _write_summary(self, dataset: str, summary: pd.DataFrame) -> None:
        """Write the summary DataFrame to ``results/<dataset>/summary.csv``."""
        out_dir = self.results_dir / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_dir / "summary.csv")