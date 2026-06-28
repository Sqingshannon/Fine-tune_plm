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
    1. :meth:`sample` — samples a fixed held-out test set (controlled by
       *test_seed*) and a k-shot training set (controlled by *model_seed*)
       from ``data.csv``, writing ``train.csv`` and ``test.csv``.
    2. :meth:`split_folds` — divides ``train.csv`` into five equal chunks
       (``train_1.csv`` … ``train_5.csv``) for validation fold selection.

    Output files are written to
    ``data_root / dataset_name / seed_{model_seed}_shot{shot} /``.
    ``data.csv`` is read from *source_dir* (defaults to *data_root*).

    Args:
        data_root: Root directory under which per-dataset seed subdirectories
            are created, e.g. ``data_rerun_fixed/``.
        test_fraction: Fraction of the full pool reserved as the test set.
        source_dir: Root directory containing ``{dataset}/data.csv`` to read
            from.  Defaults to *data_root* when not provided.

    Example:
        >>> splitter = DataSplitter(
        ...     data_root=Path("data_rerun_fixed"),
        ...     source_dir=Path("data_rerun_fixed"),
        ... )
        >>> splitter.ensure_splits_exist("PTEN_HUMAN", test_seed=0, model_seed=2, shot=96)
        # writes to data_rerun_fixed/PTEN_HUMAN/seed_2_shot96/
    """

    def __init__(
        self,
        data_root: Path = Path("data_rerun_fixed"),
        test_fraction: float = 0.2,
        source_dir: Optional[Path] = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.test_fraction = test_fraction
        self.source_dir = Path(source_dir) if source_dir is not None else self.data_root

    def seed_dir(self, dataset_name: str, model_seed: int, shot: int) -> Path:
        """Return the output directory for a given dataset, model_seed, and shot.

        Public so callers (e.g. the train entry point) can locate the split
        files without duplicating this path logic.
        """
        return self.data_root / dataset_name / f"seed_{model_seed}_shot{shot}"

    def sample(
        self,
        dataset_name: str,
        test_seed: int,
        model_seed: int,
        shot: int,
    ) -> None:
        """Sample a k-shot training set and a held-out test set.

        Reads ``source_dir / dataset_name / data.csv`` and writes
        ``train.csv`` and ``test.csv`` to
        ``data_root / dataset_name / seed_{model_seed}_shot{shot} /``.

        Args:
            dataset_name: Identifier for the dataset subdirectory.
            test_seed: Seed for the test split — keep fixed across runs so
                the test set never changes.
            model_seed: Seed for k-shot training sampling — varies per run;
                matches the ``seed{model_seed}`` checkpoint folder name.
            shot: Number of labelled examples in the training set. If the
                training pool (after holding out the test set) is smaller
                than *shot*, the entire pool is used instead.
        """
        df = pd.read_csv(self.source_dir / dataset_name / "data.csv", index_col=0)

        out_dir = self.seed_dir(dataset_name, model_seed, shot)
        out_dir.mkdir(parents=True, exist_ok=True)

        test_data  = df.sample(frac=self.test_fraction, random_state=test_seed)
        train_pool = df.drop(test_data.index)
        n = min(shot, len(train_pool))
        kshot_data = train_pool.sample(n=n, random_state=model_seed)

        kshot_data.to_csv(out_dir / "train.csv")
        test_data.to_csv(out_dir / "test.csv")

    def split_folds(
        self, dataset_name: str, model_seed: int, shot: int, n_folds: int = 5
    ) -> None:
        """Split ``train.csv`` into *n_folds* equal chunks for validation.

        Writes ``train_1.csv`` … ``train_{n_folds}.csv`` into
        ``data_root / dataset_name / seed_{model_seed}_shot{shot} /``.

        Args:
            dataset_name: Identifier for the dataset subdirectory.
            model_seed: Identifies the seed subdirectory to write into.
            shot: k-shot size identifying the seed subdirectory to write into.
            n_folds: Number of folds (default 5).
        """
        seed_dir = self.seed_dir(dataset_name, model_seed, shot)
        train = pd.read_csv(seed_dir / "train.csv")
        fold_size = int(np.ceil(len(train) / n_folds))
        start = 0
        for i in range(1, n_folds):
            chunk = train[start : start + fold_size]
            chunk.to_csv(seed_dir / f"train_{i}.csv", index=False)
            start += fold_size
        train[start:].to_csv(seed_dir / f"train_{n_folds}.csv", index=False)

    def ensure_splits_exist(
        self,
        dataset_name: str,
        test_seed: int,
        model_seed: int,
        shot: int,
    ) -> None:
        """Run :meth:`sample` and :meth:`split_folds` if splits are missing.

        Idempotent — does nothing when ``test.csv`` already exists in
        ``data_root / dataset_name / seed_{model_seed}_shot{shot} /``.

        Args:
            dataset_name: Identifier for the dataset subdirectory.
            test_seed: Forwarded to :meth:`sample`.
            model_seed: Forwarded to :meth:`sample` and :meth:`split_folds`.
            shot: k-shot size forwarded to :meth:`sample` and :meth:`split_folds`;
                also part of the output directory so different shot values
                for the same dataset/model_seed never collide.
        """
        test_path = self.seed_dir(dataset_name, model_seed, shot) / "test.csv"
        if not test_path.exists():
            self.sample(dataset_name, test_seed=test_seed, model_seed=model_seed, shot=shot)
            self.split_folds(dataset_name, model_seed=model_seed, shot=shot)
