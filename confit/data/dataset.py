"""PyTorch Dataset for protein mutation fitness data.

Replaces the original ``Mutation_Set`` class from ``data_utils.py``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from torch import Tensor
from torch.utils.data import Dataset


class MutationDataset(Dataset):
    """Dataset of protein mutations with associated fitness scores.

    Each sample contains the mutant sequence tokens, wild-type sequence tokens,
    mutation position(s), the experimental fitness score, a sequence identifier,
    and a numeric mutation index.

    The wild-type sequence is loaded from ``data/<fname>/wt.fasta``.

    Args:
        data: DataFrame with columns ``seq``, ``log_fitness``, ``mutated_position``,
              and ``PID``.
        fname: Dataset name; used to locate ``data/<fname>/wt.fasta``.
        tokenizer: HuggingFace-compatible ESM tokenizer.
        max_seq_len: Maximum sequence length passed to the tokenizer (default 1024).
        data_root: Root path containing ``<fname>/wt.fasta``
            (default ``Path("data")``).

    Example:
        >>> ds = MutationDataset(train_df, "PTEN_HUMAN", tokenizer)
        >>> seq_tok, att_mask, wt_tok, wt_mask, pos, score, pid, mut_id = ds[0]
    """

    def __init__(
        self,
        data: pd.DataFrame,
        fname: str,
        tokenizer: Any,
        max_seq_len: int = 1024,
        data_root: str = "data",
    ) -> None:
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        encoded_mut = tokenizer(
            list(self.data["seq"]),
            padding=False,
            truncation=False,
            max_length=max_seq_len,
        )
        self.seq: List[List[int]] = encoded_mut["input_ids"]
        self.attention_mask: List[List[int]] = encoded_mut["attention_mask"]

        wt_path = os.path.join(data_root, fname, "wt.fasta")
        wt_seq = str(next(SeqIO.parse(wt_path, "fasta")).seq)  
        encoded_wt = tokenizer(
            [wt_seq] * len(self.data),
            padding=False,
            truncation=False,
            max_length=max_seq_len,
        )
        self.target: List[List[int]] = encoded_wt["input_ids"]
        self.tgt_mask: List[List[int]] = encoded_wt["attention_mask"]

        self.score: Tensor = torch.tensor(
            np.array(self.data["log_fitness"]), dtype=torch.float32
        )
        self.pid: np.ndarray = np.asarray(self.data["PID"])
        self.position: List[List[int]] = self._parse_positions()
        self.mutation_id: Tensor = torch.tensor(np.arange(len(self.data)))

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the total number of mutations in the dataset."""
        return len(self.score)

    def __getitem__(self, idx: int) -> List[Any]:
        """Return a single sample.

        Returns:
            A list of eight elements:
            ``[seq_ids, att_mask, wt_ids, wt_mask, pos, score, pid, mut_id]``
        """
        return [
            self.seq[idx],
            self.attention_mask[idx],
            self.target[idx],
            self.tgt_mask[idx],
            self.position[idx],
            self.score[idx],
            self.pid[idx],
            self.mutation_id[idx],
        ]

    def collate_fn(self, batch: List[List[Any]]) -> Tuple[Tensor, ...]:
        """Collate a list of samples into a padded batch of tensors.

        Args:
            batch: List of samples returned by ``__getitem__``.

        Returns:
            Tuple of ``(seq, att_mask, tgt, tgt_mask, pos, score, pid, mutation)``.
        """
        seq = torch.tensor(np.array([s[0] for s in batch]))
        att_mask = torch.tensor(np.array([s[1] for s in batch]))
        tgt = torch.tensor(np.array([s[2] for s in batch]))
        tgt_mask = torch.tensor(np.array([s[3] for s in batch]))
        pos = [torch.tensor(s[4]) for s in batch]
        score = torch.tensor(np.array([s[5] for s in batch]), dtype=torch.float32)
        pid = torch.tensor(np.array([s[6] for s in batch]))
        mutation = torch.tensor(np.array([s[7] for s in batch]))
        return seq, att_mask, tgt, tgt_mask, pos, score, pid, mutation

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_positions(self) -> List[List[int]]:
        """Parse the ``mutated_position`` column into lists of integers."""
        raw = list(self.data["mutated_position"])
        positions: List[List[int]] = []
        for entry in raw:
            if not isinstance(entry, str):
                positions.append([int(entry)])
            else:
                positions.append([int(p) for p in entry.split(",")])
        return positions