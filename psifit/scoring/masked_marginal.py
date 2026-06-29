"""Masked-marginal probability scorer with optional SPURS-DDG correction.

Replaces the ``compute_score()`` function that was misplaced in ``stat_utils.py``.
This is a forward-pass operation — it belongs here, not in a metrics module.

Two DDG injection modes are controlled by ``AModule.combined_way``:

* ``'logits'`` — scale and add DDG to the ESM logit tensor before log-softmax.
* ``'scores'`` — compute the standard masked-marginal score, then add
  ``a * ddg_value`` as a post-hoc correction.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import torch
from torch import Tensor

from psifit.models.scaling import AModule, ScalingMode


class MaskedMarginalScorer:
    """Computes mutational fitness proxy via masked marginal log-probabilities.

    For each mutant in a batch the scorer:

    1. Clones the input token sequence and masks the mutation position.
    2. Runs the ESM forward pass to obtain per-position logits.
    3. Optionally integrates the SPURS DDG correction (``'logits'`` mode).
    4. Computes ``log P(mut | context) − log P(wt | context)``.
    5. Optionally adds the SPURS DDG correction (``'scores'`` mode).

    Args:
        tokenizer: HuggingFace ESM tokenizer (provides ``mask_token_id``).
        a_module: :class:`~psifit.models.scaling.AModule` instance.
        spurs_ddg: SPURS DDG tensor, shape ``(L, 20)``.
        aa_token_ids: Token IDs for the 20 canonical amino acids, shape ``(20,)``.

    Example:
        >>> scorer = MaskedMarginalScorer(tokenizer, A, spurs_ddg, aa_token_ids)
        >>> scores, logits = scorer.score(model, seq, mask, wt, pos)
    """

    def __init__(
        self,
        tokenizer: Any,
        a_module: AModule,
        spurs_ddg: Tensor,
        aa_token_ids: Tensor,
    ) -> None:
        self.tokenizer = tokenizer
        self.a_module = a_module
        self.spurs_ddg = spurs_ddg
        self.aa_token_ids = aa_token_ids

    def score(
        self,
        model: Any,
        seq: Tensor,
        mask: Tensor,
        wt: Tensor,
        pos: List[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """Compute masked-marginal scores for a batch of mutations.

        Args:
            model: ESM model (may be PEFT-wrapped).
            seq: Mutant token IDs including BOS/EOS, shape ``(B, L+2)``.
            mask: Attention mask, shape ``(B, L+2)``.
            wt: Wild-type token IDs, shape ``(B, L+2)``.
            pos: List of B tensors, each containing mutation position index(es)
                (0-indexed, relative to the raw sequence without BOS/EOS).

        Returns:
            Tuple ``(scores, logits)``
            * ``scores``: per-sample fitness proxy, shape ``(B,)``.
            * ``logits``: ESM output logits, shape ``(B, L+2, V)``.
        """
        device = seq.device
        batch_size = seq.shape[0]

        pos_tensor = torch.tensor(
            [p[0].item() for p in pos], dtype=torch.long, device=device
        )

        masked_seq = self._mask_positions(seq, pos_tensor, batch_size, device)
        out = model(masked_seq, mask, output_hidden_states=True)
        logits: Tensor = out.logits

        if self._use_logits_mode():
            logits = self._apply_logits_correction(logits, pos_tensor, batch_size, device)

        scores = self._delta_log_probs(logits, seq, wt, pos_tensor, batch_size, device)

        if self._use_scores_mode():
            scores = self._apply_scores_correction(
                scores, logits, seq, pos_tensor, batch_size, device
            )

        return scores, logits

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _use_logits_mode(self) -> bool:
        """True when logits-mode DDG correction should be applied."""
        return (
            self.a_module is not None
            and self.a_module.combined_way == "logits"
            and self.a_module.mode != ScalingMode.NONE
        )

    def _use_scores_mode(self) -> bool:
        """True when scores-mode DDG correction should be applied."""
        return self.a_module is not None and self.a_module.combined_way == "scores"

    def _mask_positions(
        self,
        seq: Tensor,
        pos: Tensor,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        """Return a copy of *seq* with mutation positions replaced by [MASK]."""
        masked = seq.clone()
        batch_idx = torch.arange(batch_size, device=device)
        masked[batch_idx, pos + 1] = self.tokenizer.mask_token_id
        return masked

    def _apply_logits_correction(
        self,
        logits: Tensor,
        pos: Tensor,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        """Add scaled DDG to the amino-acid columns of *logits*."""
        seq_len = logits.shape[1] - 2
        aa_ids = self.aa_token_ids.to(device)
        aligned_ddg = self.spurs_ddg.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        aligned_logits = logits[:, 1 : seq_len + 1, aa_ids]

        A = self.a_module
        if A.mode == ScalingMode.SINGLE:
            scaled_ddg = A.A * aligned_ddg
        elif A.mode == ScalingMode.POSITION_SPECIFIC:
            a = A.A.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, -1).to(device)
            scaled_ddg = a * aligned_ddg
        else:  # CONTEXT_SPECIFIC
            flat_esm = aligned_logits.reshape(-1, 20)
            flat_ddg = aligned_ddg.reshape(-1, 20)
            a = A(esm_i=flat_esm, ddg_i=flat_ddg).reshape(batch_size, seq_len, 1)
            scaled_ddg = a * aligned_ddg

        corrected = logits.clone()
        corrected[:, 1 : seq_len + 1, aa_ids] = (
            corrected[:, 1 : seq_len + 1, aa_ids] + scaled_ddg
        )
        return corrected

    @staticmethod
    def _delta_log_probs(
        logits: Tensor,
        seq: Tensor,
        wt: Tensor,
        pos: Tensor,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        """Compute log P(mut) − log P(wt) at each mutation site."""
        log_probs = torch.log_softmax(logits, dim=-1)
        batch_idx = torch.arange(batch_size, device=device)
        p = pos + 1  # +1 for BOS token offset

        mut_token = seq[batch_idx, p]
        wt_token = wt[batch_idx, p]
        return log_probs[batch_idx, p, mut_token] - log_probs[batch_idx, p, wt_token]

    def _apply_scores_correction(
        self,
        scores: Tensor,
        logits: Tensor,
        seq: Tensor,
        pos: Tensor,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        """Add ``a * ddg_value`` post-hoc to masked-marginal scores."""
        aa_ids = self.aa_token_ids.to(device)
        batch_idx = torch.arange(batch_size, device=device)
        p = pos + 1

        mut_token = seq[batch_idx, p]
        mut_idx = (aa_ids == mut_token.unsqueeze(1)).nonzero(as_tuple=True)[1]
        ddg_value = self.spurs_ddg[pos, mut_idx]

        A = self.a_module
        if A.mode == ScalingMode.SINGLE:
            a = A.A.expand(batch_size).to(device)
        elif A.mode == ScalingMode.POSITION_SPECIFIC:
            a = A(mut_pos=pos).to(device)
        elif A.mode == ScalingMode.CONTEXT_SPECIFIC:
            esm_i = logits[batch_idx, p][:, aa_ids]
            ddg_i = self.spurs_ddg[pos]
            a = A(esm_i=esm_i, ddg_i=ddg_i).squeeze(-1)
        else:  # NONE
            a = A.A.expand(batch_size).to(device)

        return scores + a * ddg_value