"""YAML → TrainingConfig loader.

Provides a single, reusable entry point for loading and validating config files,
replacing the scattered ``yaml.load(...)`` + manual dict casts in the original code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import yaml

from psifit.config.schema import TrainingConfig


class ConfigLoader:
    """Loads and validates a YAML config file into a ``TrainingConfig``.

    Example:
        >>> cfg = ConfigLoader.load("config/training_config.yaml")
        >>> cfg.per_device_batch_size
        8
    """

    @staticmethod
    def load(path: Union[str, Path]) -> TrainingConfig:
        """Load a YAML file and validate it as a ``TrainingConfig``.

        Args:
            path: Filesystem path to the ``.yaml`` config file.

        Returns:
            A fully validated and frozen ``TrainingConfig`` instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
            pydantic.ValidationError: If any field fails validation.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open("r", encoding="utf-8") as fh:
            raw: dict = yaml.safe_load(fh)

        return TrainingConfig(**raw)