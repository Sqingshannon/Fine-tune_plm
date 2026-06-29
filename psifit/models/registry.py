"""ESM model variant registry.

Replaces the repeated ``if config['model'] == 'ESM-1v'`` blocks scattered
across ``train.py`` with a single, extensible enum + lookup table.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict


class ModelVariant(str, Enum):
    """Supported ESM model variants.

    Each member's value is the canonical string used in YAML config files.

    Attributes:
        ESM_1V: ESM-1v (650M, UR90S).
        ESM_1B: ESM-1b (650M, UR50S).
        ESM_2:  ESM-2  (15B,  UR50D).
    """

    ESM_1V = "ESM-1v"
    ESM_1B = "ESM-1b"
    ESM_2 = "ESM-2"


class ModelRegistry:
    """Maps :class:`ModelVariant` values to HuggingFace model hub IDs.

    The registry is intentionally a thin data structure; actual model
    construction is delegated to :class:`~psifit.models.factory.ESMModelFactory`.

    Example:
        >>> hub_id = ModelRegistry.hub_id(ModelVariant.ESM_1V, model_seed=2)
        'facebook/esm1v_t33_650M_UR90S_2'
    """

    _STATIC_IDS: Dict[ModelVariant, str] = {
        ModelVariant.ESM_1B: "facebook/esm1b_t33_650M_UR50S",
        ModelVariant.ESM_2: "facebook/esm2_t48_15B_UR50D",
    }

    @classmethod
    def hub_id(cls, variant: ModelVariant, model_seed: int = 1) -> str:
        """Return the HuggingFace hub model ID for *variant*.

        For ESM-1v the *model_seed* suffix (1–5) is appended per the
        original ensemble convention.

        Args:
            variant: The model variant enum member.
            model_seed: Seed index used only for ESM-1v (1–5).

        Returns:
            A fully-qualified HuggingFace hub model identifier string.

        Raises:
            KeyError: If *variant* is not registered.
        """
        if variant == ModelVariant.ESM_1V:
            return f"facebook/esm1v_t33_650M_UR90S_{model_seed}"
        try:
            return cls._STATIC_IDS[variant]
        except KeyError:
            raise KeyError(f"No hub ID registered for model variant: {variant!r}")

    @classmethod
    def from_string(cls, name: str) -> ModelVariant:
        """Resolve a config-file string to a :class:`ModelVariant`.

        Args:
            name: Raw string from YAML config (e.g. ``'ESM-1v'``).

        Returns:
            The matching :class:`ModelVariant` enum member.

        Raises:
            ValueError: If *name* does not match any registered variant.
        """
        try:
            return ModelVariant(name)
        except ValueError:
            valid = [v.value for v in ModelVariant]
            raise ValueError(
                f"Unknown model variant '{name}'. Valid options: {valid}"
            )