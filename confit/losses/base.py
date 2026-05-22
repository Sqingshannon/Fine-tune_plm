"""Abstract base class for all ConFit loss functions."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


class BaseLoss(ABC, nn.Module):
    """Abstract base for ConFit loss functions.

    All losses are ``nn.Module`` subclasses so they integrate naturally with
    :class:`~accelerate.Accelerator` and can be moved to a specific device.
    Concrete subclasses must implement :meth:`forward`.
    """

    @abstractmethod
    def forward(self, *args: Tensor, **kwargs: Tensor) -> Tensor:
        """Compute and return the scalar loss tensor.

        Args:
            *args: Positional tensors (concrete signatures vary per subclass).
            **kwargs: Keyword tensors.

        Returns:
            Scalar loss tensor on the same device as the inputs.
        """