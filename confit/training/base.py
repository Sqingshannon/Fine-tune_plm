"""Abstract base trainer interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from torch.utils.data import DataLoader


class BaseTrainer(ABC):
    """Abstract base class for all ConFit trainers.

    Defines the minimal contract that every trainer must satisfy.
    Concrete subclasses (e.g. :class:`~confit.training.trainer.ConFitTrainer`)
    implement the full training logic.
    """

    @abstractmethod
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_dir: Path,
    ) -> float:
        """Train the model and return the best validation Spearman correlation.

        Args:
            train_loader: DataLoader over the labelled training set.
            val_loader: DataLoader over the held-out validation fold.
            save_dir: Directory path where the best checkpoint is written.

        Returns:
            Best validation Spearman ρ achieved during training.
        """