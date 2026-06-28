"""KL-divergence regularisation loss.

Keeps the fine-tuned ESM model close to the frozen reference checkpoint by
minimising the KL divergence between their per-token probability distributions.

Replaces the bare ``KLloss()`` function from ``stat_utils.py``.  The original
hardcoded ``.cuda()`` for the accumulator tensor; this implementation uses
``torch.zeros_like`` for automatic device placement.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from confit.losses.base import BaseLoss


class KLRegularizationLoss(BaseLoss):
    """Per-sequence KL divergence between fine-tuned and reference ESM logits.

    For each sequence in the batch the KL divergence is computed over the
    tokens within the valid (non-padding) region and accumulated into a mean.

    Args:
        None — uses ``KLDivLoss(reduction='mean')`` internally.

    Example:
        >>> reg_loss = KLRegularizationLoss()
        >>> loss = reg_loss(logits, logits_ref, seq_tokens, attention_mask)
    """

    def __init__(self) -> None:
        super().__init__()
        self._kl = nn.KLDivLoss(reduction="mean")

    def forward(
        self,
        logits: Tensor,
        logits_reg: Tensor,
        seq: Tensor,
        att_mask: Tensor,
    ) -> Tensor:
        """Compute per-batch KL regularisation loss.

        Args:
            logits: Logits from the fine-tuned model, shape ``(B, L, V)``.
            logits_reg: Logits from the frozen reference model, shape ``(B, L, V)``.
            seq: Token IDs of the input sequences, shape ``(B, L)``.
            att_mask: Attention mask (1 = valid token), shape ``(B, L)``.

        Returns:
            Scalar mean KL loss tensor on the same device as *logits*.
        """
        batch_size = seq.shape[0]
        loss = torch.zeros(1, device=logits.device, dtype=logits.dtype).squeeze()

        probs = torch.softmax(logits, dim=-1)
        probs_reg = torch.softmax(logits_reg, dim=-1)

        for i in range(batch_size):
            seq_len = int(torch.sum(att_mask[i]).item())
            token_range = torch.arange(seq_len, device=seq.device)
            token_ids = seq[i, :seq_len]

            pred_i = probs[i][token_range, token_ids]
            ref_i = probs_reg[i][token_range, token_ids]

            loss = loss + self._kl(ref_i.log(), pred_i)

        return loss