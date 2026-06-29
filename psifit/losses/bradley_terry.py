"""Bradley-Terry pairwise ranking loss.

Replaces the bare ``BT_loss()`` function from the original ``stat_utils.py``.
The original function hardcoded ``.cuda()``, making it incompatible with CPU
and multi-device setups.  This implementation uses ``torch.zeros_like`` to
initialise the accumulator on the correct device automatically.
"""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from psifit.losses.base import BaseLoss


class BradleyTerryLoss(BaseLoss):
    """Pairwise Bradley-Terry ranking loss.

    For every pair (i, j) where ``golden_score[i] > golden_score[j]``, the loss
    penalises predicted scores that disagree with that ordering::

        L = Σ_{i<j} log(1 + exp(score_loser - score_winner))

    Args:
        None — no hyperparameters.

    Example:
        >>> loss_fn = BradleyTerryLoss()
        >>> loss = loss_fn(scores, golden_scores)
    """

    def forward(self, scores: Tensor, golden_scores: Tensor) -> Tensor:
        """Compute the Bradley-Terry pairwise ranking loss.

        Args:
            scores: Predicted fitness scores, shape ``(batch,)``.
            golden_scores: Ground-truth fitness scores, shape ``(batch,)``.

        Returns:
            Scalar loss tensor on the same device as *scores*.
        """
        # pairwise diffs: diff[i,j] = scores[i] - scores[j]
        diff        = scores.unsqueeze(1)        - scores.unsqueeze(0)
        golden_diff = golden_scores.unsqueeze(1) - golden_scores.unsqueeze(0)

        # loser_score - winner_score for each pair
        arg = torch.where(golden_diff > 0, -diff, diff)

        # upper triangle only (each pair counted once)
        mask = torch.ones(len(scores), len(scores), dtype=torch.bool, device=scores.device).triu(diagonal=1)
        return F.softplus(arg[mask]).sum()