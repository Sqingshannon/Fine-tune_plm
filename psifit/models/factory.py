"""ESM model factory — constructs backbone, regularisation copy, and tokenizer.

Replaces the repeated model-creation blocks in the original ``train.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Tuple

from transformers import EsmForMaskedLM, EsmTokenizer

from psifit.models.registry import ModelRegistry, ModelVariant


@dataclass
class ModelBundle:
    """Container for a constructed ESM backbone, regularisation copy, and tokenizer.

    Attributes:
        backbone: The trainable ESM model (will have LoRA applied).
        reg_model: A frozen copy of the same checkpoint for KL regularisation.
        tokenizer: The matching ESM tokenizer.
        variant: The :class:`ModelVariant` that was instantiated.
    """

    backbone: EsmForMaskedLM
    reg_model: EsmForMaskedLM
    tokenizer: EsmTokenizer
    variant: ModelVariant


class BaseModelFactory(ABC):
    """Abstract factory interface for model bundle construction.

    Subclass and implement :meth:`build` to support new model families.
    """

    @abstractmethod
    def build(self, variant: ModelVariant, model_seed: int = 1) -> ModelBundle:
        """Build and return a :class:`ModelBundle` for *variant*.

        Args:
            variant: The model variant to construct.
            model_seed: Seed index for ESM-1v ensemble (1–5). Ignored by
                model families that don't use seed-indexed checkpoints.

        Returns:
            A fully initialised :class:`ModelBundle`.
        """


class ESMModelFactory(BaseModelFactory):
    """Concrete factory for ESM protein language models.

    Loads both a trainable backbone and a frozen regularisation copy from
    the HuggingFace model hub, matching the original ``train.py`` pattern.

    Example:
        >>> factory = ESMModelFactory()
        >>> bundle = factory.build(ModelVariant.ESM_1V, model_seed=1)
        >>> bundle.tokenizer.mask_token_id
        32
    """

    def build(self, variant: ModelVariant, model_seed: int = 1) -> ModelBundle:
        """Instantiate ESM backbone, regularisation model, and tokenizer.

        The regularisation model is immediately frozen (``requires_grad=False``)
        and set to ``eval()`` mode.

        Args:
            variant: ESM variant to load.
            model_seed: Seed index (1–5) used only for ESM-1v.

        Returns:
            A :class:`ModelBundle` with ``backbone``, ``reg_model``, and
            ``tokenizer`` fields populated.
        """
        hub_id = ModelRegistry.hub_id(variant, model_seed=model_seed)

        backbone: EsmForMaskedLM = EsmForMaskedLM.from_pretrained(hub_id)
        reg_model: EsmForMaskedLM = EsmForMaskedLM.from_pretrained(hub_id)
        tokenizer: EsmTokenizer = EsmTokenizer.from_pretrained(hub_id)

        for param in reg_model.parameters():
            param.requires_grad = False
        reg_model.eval()

        return ModelBundle(
            backbone=backbone,
            reg_model=reg_model,
            tokenizer=tokenizer,
            variant=variant,
        )