"""K-shot train/test splitting and 5-fold cross-validation partitioning.

Replaces the bare ``sample_data()`` and ``split_train()`` functions from
the original ``data_utils.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class DataSplitter:
    """Creates k-shot train/test splits and 5-fold validation partitions.

    Workflow
    --------
    1. :meth:`sample` — randomly samples a held-out test set and a k-shot
       training set from ``data.csv``, writing ``train.csv`` and ``test.csv``.
    2. :meth:`split_folds` — divides ``train.csv`` into five equal chunks
       (``train_1.csv`` … ``train_5.csv``) for cross-validation.

    Both methods are idempotent when ``test.csv`` already exists — the
    caller is expected to guard against re-running (see :class:`ExperimentRunner`).

    Args:
        data_root: Root directory that contains ``<dataset_name>/data.csv``.
        test_fraction: Fraction of the full pool reserved as the test set.

    Example:
        >>> splitter = DataSplitter(data_root=Path("data_rerun_fixed"))
        >>> splitter.sample("PTEN_HUMAN", seed=0, shot=96)
        >>> splitter.split_folds("PTEN_HUMAN")
    """

    def __init__(
        self,
        data_root: Path = Path("data"),
        test_fraction: float = 0.2,
    ) -> None:
        self.data_root = Path(data_root)
        self.test_fraction = test_fraction

    def sample(self, dataset_name: str, seed: int, shot: int) -> None:
        """Sample a k-shot training set and a held-out test set.

        Reads ``data_root / dataset_name / data.csv`` and writes
        ``train.csv`` and ``test.csv`` to the same directory.

        Args:
            dataset_name: Identifier for the dataset subdirectory.
            seed: Random seed for reproducible sampling.
            shot: Number of labelled examples in the training set.

        Raises:
            AssertionError: If the sampled training set size does not equal *shot*.
        """
        dataset_dir = self.data_root / dataset_name
        df = pd.read_csv(dataset_dir / "data.csv", index_col=0)

        test_data = df.sample(frac=self.test_fraction, random_state=seed)
        train_pool = df.drop(test_data.index)
        kshot_data = train_pool.sample(n=shot, random_state=seed)

        assert len(kshot_data) == shot, (
            f"Expected {shot} training examples, got {len(kshot_data)}."
        )

        kshot_data.to_csv(dataset_dir / "train.csv", index=False)
        test_data.to_csv(dataset_dir / "test.csv", index=False)

    def split_folds(self, dataset_name: str, n_folds: int = 5) -> None:
        """Split ``train.csv`` into *n_folds* equal chunks for cross-validation.

        Writes ``train_1.csv`` … ``train_{n_folds}.csv`` alongside ``train.csv``.

        Args:
            dataset_name: Identifier for the dataset subdirectory.
            n_folds: Number of folds (default 5).
        """
        dataset_dir = self.data_root / dataset_name
        train = pd.read_csv(dataset_dir / "train.csv")
        fold_size = int(np.ceil(len(train) / n_folds))
        start = 0
        for i in range(1, n_folds):
            chunk = train[start : start + fold_size]
            chunk.to_csv(dataset_dir / f"train_{i}.csv", index=False)
            start += fold_size
        train[start:].to_csv(dataset_dir / f"train_{n_folds}.csv", index=False)

    def ensure_splits_exist(
        self, dataset_name: str, seed: int, shot: int
    ) -> None:
        """Run :meth:`sample` and :meth:`split_folds` if splits are missing.

        Idempotent — does nothing when ``test.csv`` already exists.

        Args:
            dataset_name: Identifier for the dataset subdirectory.
            seed: Random seed forwarded to :meth:`sample`.
            shot: k-shot size forwarded to :meth:`sample`.
        """
        test_path = self.data_root / dataset_name / "test.csv"
        if not test_path.exists():
            self.sample(dataset_name, seed=seed, shot=shot)
            self.split_folds(dataset_name)