"""ConFit evaluation loop.

Decoupled from the trainer so it can be reused for standalone inference,
hyperparameter search, or test-set evaluation without touching the training code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from confit.metrics.correlation import spearman
from confit.scoring.masked_marginal import MaskedMarginalScorer


@dataclass
class EvaluationResult:
    """Container for the outputs of a single evaluation pass.

    Attributes:
        spearman_correlation: Spearman ρ between predicted and ground-truth scores.
        scores: Predicted scores array, shape ``(N,)``. Populated when
            ``collect_outputs=True``.
        ground_truth: Ground-truth fitness scores, shape ``(N,)``. Populated when
            ``collect_outputs=True``.
        sequence_ids: Sequence PID array, shape ``(N,)``. Populated when
            ``is_test=True``.
        mutation_ids: Mutation index array, shape ``(N,)``. Populated when
            ``collect_outputs=True``.
    """

    spearman_correlation: float
    scores: Optional[np.ndarray] = None
    ground_truth: Optional[np.ndarray] = None
    sequence_ids: Optional[np.ndarray] = None
    mutation_ids: Optional[np.ndarray] = None


class ConFitEvaluator:
    """Runs a full evaluation loop over a DataLoader and returns an EvaluationResult.

    Args:
        scorer: :class:`~confit.scoring.masked_marginal.MaskedMarginalScorer`
            that computes per-sample fitness proxies.
        accelerator: :class:`~accelerate.Accelerator` instance for distributed
            gather operations.
        tokenizer: ESM tokenizer (passed through to the scorer).

    Example:
        >>> evaluator = ConFitEvaluator(scorer, accelerator, tokenizer)
        >>> result = evaluator.evaluate(model, val_loader)
        >>> print(result.spearman_correlation)
    """

    def __init__(
        self,
        scorer: MaskedMarginalScorer,
        accelerator: object,
        tokenizer: object,
    ) -> None:
        self.scorer = scorer
        self.accelerator = accelerator
        self.tokenizer = tokenizer

    def evaluate(
        self,
        model: object,
        loader: DataLoader,
        is_test: bool = False,
    ) -> EvaluationResult:
        """Run the evaluation loop and compute Spearman correlation.

        Args:
            model: ESM model (may be PEFT-wrapped).
            loader: DataLoader yielding batches in the canonical 8-tuple format.
            is_test: When ``True``, collects sequence PIDs in addition to scores.

        Returns:
            An :class:`EvaluationResult` with all populated fields.
        """
        model.eval()  # type: ignore[union-attr]
        acc = self.accelerator

        score_list: List[np.ndarray] = []
        gscore_list: List[np.ndarray] = []
        mutation_list: List[np.ndarray] = []
        seq_list: List[np.ndarray] = []

        with torch.no_grad():
            for batch in loader:
                seq, mask, wt, _wt_mask, pos, golden_score, pid, mutation = batch

                scores, _logits = self.scorer.score(model, seq, mask, wt, pos)
                scores = scores.to(acc.device)  # type: ignore[union-attr]

                scores = acc.gather(scores)  # type: ignore[union-attr]
                golden_score = acc.gather(golden_score)  # type: ignore[union-attr]
                mutation = acc.gather(mutation)  # type: ignore[union-attr]

                score_list.extend(np.asarray(scores.cpu()))
                gscore_list.extend(np.asarray(golden_score.cpu()))
                mutation_list.extend(np.asarray(mutation.cpu()))

                if is_test:
                    pid = pid.to(acc.device)  # type: ignore[union-attr]
                    pid = acc.gather(pid)  # type: ignore[union-attr]
                    seq_list.extend([s.cpu() for s in pid])

        scores_arr = np.asarray(score_list)
        gscores_arr = np.asarray(gscore_list)
        mutations_arr = np.asarray(mutation_list)
        sr = spearman(scores_arr, gscores_arr)

        return EvaluationResult(
            spearman_correlation=sr,
            scores=scores_arr,
            ground_truth=gscores_arr,
            mutation_ids=mutations_arr,
            sequence_ids=np.asarray(seq_list) if is_test else None,
        )