"""Learnable SPURS-DDG scaling strategies for ConFit.

Replaces the monolithic ``AModule`` class from ``train.py``.  Each scaling
mode is now an independent Strategy that satisfies a common abstract interface,
making it trivial to add new modes without touching existing code.

Design pattern: Strategy (each subclass is a distinct algorithm).

Public API:
    - :class:`ScalingMode`               — enum of supported modes
    - :class:`BaseScalingStrategy`       — ABC that all strategies implement
    - :class:`NoScalingStrategy`         — pass-through (a = 0)
    - :class:`SingleScalingStrategy`     — single learnable scalar
    - :class:`PositionSpecificStrategy`  — per-position learnable vector
    - :class:`ContextSpecificStrategy`   — MLP that predicts a from (ESM logits, DDG)
    - :class:`AModule`                   — composes one strategy; drop-in replacement
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------


class ScalingMode(str, Enum):
    """Supported SPURS-DDG scaling modes.

    Attributes:
        NONE: No scaling (a = 0). Pure ESM score.
        SINGLE: One global learnable scalar shared across all positions.
        POSITION_SPECIFIC: Per-position learnable vector of length L.
        CONTEXT_SPECIFIC: MLP that predicts a per-sample scalar from
            ESM logits and DDG features.
    """

    NONE = "none"
    SINGLE = "single"
    POSITION_SPECIFIC = "position-specific"
    CONTEXT_SPECIFIC = "context-specific"


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseScalingStrategy(ABC, nn.Module):
    """Abstract base for SPURS-DDG scaling strategies.

    Each concrete strategy encapsulates one specific way of computing the
    scaling coefficient ``a`` that is multiplied with ``spurs_ddg`` before
    being added to the ESM masked-marginal score.

    Concrete subclasses must implement :meth:`forward`.
    """

    @property
    @abstractmethod
    def mode(self) -> ScalingMode:
        """The :class:`ScalingMode` this strategy implements."""

    @abstractmethod
    def forward(
        self,
        esm_logits: Optional[Tensor] = None,
        ddg_features: Optional[Tensor] = None,
        mut_pos: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        """Compute the scaling coefficient tensor ``a``.

        Args:
            esm_logits: ESM logits at the mutation position,
                shape ``(batch, 20)``. Required for context-specific mode.
            ddg_features: SPURS DDG feature vector at the mutation position,
                shape ``(batch, 20)``. Required for context-specific mode.
            mut_pos: Long tensor of mutation position indices (0-indexed),
                shape ``(batch,)``. Required for position-specific mode.

        Returns:
            Scaling coefficient tensor, or ``None`` for the no-op mode.
        """


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------


class NoScalingStrategy(BaseScalingStrategy):
    """Pass-through strategy that contributes zero SPURS correction.

    Keeps a registered zero buffer so that ``AModule.combined_way`` branching
    in :class:`~confit.scoring.masked_marginal.MaskedMarginalScorer` can
    still call ``A.A`` without an AttributeError.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("A", torch.zeros(1))

    @property
    def mode(self) -> ScalingMode:
        """Return :attr:`ScalingMode.NONE`."""
        return ScalingMode.NONE

    def forward(
        self,
        esm_logits: Optional[Tensor] = None,
        ddg_features: Optional[Tensor] = None,
        mut_pos: Optional[Tensor] = None,
    ) -> None:
        """Return ``None`` — no scaling is applied."""
        return None


class SingleScalingStrategy(BaseScalingStrategy):
    """Single global learnable scalar ``a`` shared across all positions.

    Args:
        a_init: Initial value of the scalar parameter.
    """

    def __init__(self, a_init: float) -> None:
        super().__init__()
        self.A = nn.Parameter(torch.tensor(a_init))

    @property
    def mode(self) -> ScalingMode:
        """Return :attr:`ScalingMode.SINGLE`."""
        return ScalingMode.SINGLE

    def forward(
        self,
        esm_logits: Optional[Tensor] = None,
        ddg_features: Optional[Tensor] = None,
        mut_pos: Optional[Tensor] = None,
    ) -> Tensor:
        """Return the global scalar parameter.

        Returns:
            Scalar tensor ``A``.
        """
        return self.A


class PositionSpecificStrategy(BaseScalingStrategy):
    """Per-position learnable vector ``A`` of shape ``(L,)``.

    Args:
        n_positions: Sequence length L (number of positions).
        a_init: Initial fill value for all positions.
    """

    def __init__(self, n_positions: int, a_init: float) -> None:
        super().__init__()
        self.A = nn.Parameter(torch.full((n_positions,), a_init))

    @property
    def mode(self) -> ScalingMode:
        """Return :attr:`ScalingMode.POSITION_SPECIFIC`."""
        return ScalingMode.POSITION_SPECIFIC

    def forward(
        self,
        esm_logits: Optional[Tensor] = None,
        ddg_features: Optional[Tensor] = None,
        mut_pos: Optional[Tensor] = None,
    ) -> Tensor:
        """Look up the scaling coefficients for the given mutation positions.

        Args:
            mut_pos: Long tensor of 0-indexed positions, shape ``(batch,)``.

        Returns:
            Scaling coefficients, shape ``(batch,)``.

        Raises:
            ValueError: If ``mut_pos`` is not provided.
        """
        if mut_pos is None:
            raise ValueError("mut_pos must be provided for position-specific mode.")
        return self.A[mut_pos]


class ContextSpecificStrategy(BaseScalingStrategy):
    """MLP that predicts a per-sample scaling coefficient from ESM + DDG features.

    Architecture: ``Linear(20→H) + ReLU`` for both ESM and DDG streams,
    summed, then ``Linear(H→1)``.

    Args:
        hidden_size: Width of the hidden layer (default 20).
    """

    def __init__(self, hidden_size: int = 20) -> None:
        super().__init__()
        self.lin1 = nn.Linear(20, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    @property
    def mode(self) -> ScalingMode:
        """Return :attr:`ScalingMode.CONTEXT_SPECIFIC`."""
        return ScalingMode.CONTEXT_SPECIFIC

    def forward(
        self,
        esm_logits: Optional[Tensor] = None,
        ddg_features: Optional[Tensor] = None,
        mut_pos: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict a scaling coefficient from ESM logits and DDG features.

        Args:
            esm_logits: ESM logit slice at mutation positions, shape ``(N, 20)``.
            ddg_features: SPURS DDG slice at mutation positions, shape ``(N, 20)``.

        Returns:
            Predicted scaling coefficient, shape ``(N, 1)``.

        Raises:
            ValueError: If either ``esm_logits`` or ``ddg_features`` is None.
        """
        if esm_logits is None or ddg_features is None:
            raise ValueError(
                "esm_logits and ddg_features must be provided for "
                "context-specific mode."
            )
        x_esm = self.relu(self.lin1(esm_logits))
        x_ddg = self.relu(self.lin2(ddg_features))
        return self.lin3(x_esm + x_ddg)


# ---------------------------------------------------------------------------
# AModule — composes a strategy; drop-in replacement for the original
# ---------------------------------------------------------------------------


class AModule(nn.Module):
    """Wrapper that composes a :class:`BaseScalingStrategy`.

    This class is the public-facing ``A`` object consumed by both the
    :class:`~confit.scoring.masked_marginal.MaskedMarginalScorer` and the
    :class:`~confit.training.trainer.ConFitTrainer`.  It exposes the same
    attributes (``mode``, ``combined_way``, ``A``) that the original code
    relied on, ensuring full backward compatibility.

    Args:
        mode: Scaling mode as a ``str`` or :class:`ScalingMode` enum member.
        spurs_ddg_shape: Shape of the SPURS DDG tensor ``(L, 20)``; used to
            infer ``n_positions`` for position-specific mode.
        a_init: Initial scalar value (used for single / position-specific).
        combined_way: How DDG is injected — ``'scores'`` or ``'logits'``.
        hidden_size: MLP hidden size for context-specific mode.

    Example:
        >>> A = AModule(mode="single", spurs_ddg_shape=(350, 20), a_init=0.1)
        >>> A.mode
        ScalingMode.SINGLE
    """

    def __init__(
        self,
        mode: str,
        spurs_ddg_shape: Tuple[int, int],
        a_init: float,
        combined_way: Optional[str] = None,
        hidden_size: int = 20,
    ) -> None:
        super().__init__()
        self.combined_way = combined_way
        self._strategy = self._build_strategy(
            ScalingMode(mode), spurs_ddg_shape, a_init, hidden_size
        )

    # Expose .mode and .A for backward compatibility with scorer / trainer
    @property
    def mode(self) -> ScalingMode:
        """The active :class:`ScalingMode`."""
        return self._strategy.mode

    @property
    def A(self) -> Optional[nn.Parameter]:
        """Direct access to the ``A`` parameter of the underlying strategy."""
        return getattr(self._strategy, "A", None)

    def forward(
        self,
        esm_i: Optional[Tensor] = None,
        ddg_i: Optional[Tensor] = None,
        mut_pos: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        """Delegate forward pass to the underlying strategy.

        Args:
            esm_i: ESM feature slice, shape ``(N, 20)``.
            ddg_i: DDG feature slice, shape ``(N, 20)``.
            mut_pos: Mutation position indices, shape ``(N,)``.

        Returns:
            Scaling coefficients or ``None`` (for no-op mode).
        """
        return self._strategy(
            esm_logits=esm_i,
            ddg_features=ddg_i,
            mut_pos=mut_pos,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_strategy(
        mode: ScalingMode,
        spurs_ddg_shape: Tuple[int, int],
        a_init: float,
        hidden_size: int,
    ) -> BaseScalingStrategy:
        """Instantiate the correct strategy for *mode*.

        Args:
            mode: Selected :class:`ScalingMode`.
            spurs_ddg_shape: ``(n_positions, n_aa)`` shape of the DDG tensor.
            a_init: Initial scalar value.
            hidden_size: MLP hidden width for context-specific mode.

        Returns:
            A concrete :class:`BaseScalingStrategy` instance.

        Raises:
            ValueError: If *mode* is not a valid :class:`ScalingMode`.
        """
        if mode == ScalingMode.NONE:
            return NoScalingStrategy()
        if mode == ScalingMode.SINGLE:
            return SingleScalingStrategy(a_init=a_init)
        if mode == ScalingMode.POSITION_SPECIFIC:
            return PositionSpecificStrategy(
                n_positions=spurs_ddg_shape[0], a_init=a_init
            )
        if mode == ScalingMode.CONTEXT_SPECIFIC:
            return ContextSpecificStrategy(hidden_size=hidden_size)
        raise ValueError(f"Unsupported ScalingMode: {mode!r}")