"""ProteinGym data preprocessing into the ConFit canonical format.

Replaces the bare ``data_restruct()`` function from the original ``data_check.py``.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import pandas as pd
from Bio import SeqIO


class DataPreprocessor:
    """Converts raw ProteinGym DMS data into the canonical ConFit layout.

    Expected input layout (``input_dir / dms_id /``)::

        wildtype.fasta       — wild-type protein sequence (FASTA)
        proteingym_dms.tsv   — DMS assay data with columns:
                               mutant, DMS_score, [mutated_sequence (optional)]

    Produced output layout (``output_base / dms_id /``)::

        wt.fasta             — copy of wildtype.fasta
        data.csv             — full mutation pool with columns:
                               seq, log_fitness, n_mut, mutant, PID,
                               mutated_position

    ``sample_data`` / ``split_train`` in :mod:`confit.data.splitter` create the
    actual ``test.csv`` and ``train_i.csv`` splits from ``data.csv`` at run time.

    Args:
        input_dir: Root directory containing raw ProteinGym datasets.
        output_base: Root directory where processed datasets are written.

    Example:
        >>> proc = DataPreprocessor(
        ...     input_dir=Path("/data/proteingym"),
        ...     output_base=Path("data_rerun_fixed"),
        ... )
        >>> proc.prepare("PTEN_HUMAN")
    """

    def __init__(
        self,
        input_dir: Path = Path("/work/yunan/PsiFit/data/proteingym"),
        output_base: Path = Path("./data"),
    ) -> None:
        self.input_dir = Path(input_dir)
        self.output_base = Path(output_base)

    def prepare(self, dms_id: str) -> Path:
        """Prepare a single DMS dataset, writing to ``output_base / dms_id``.

        Idempotent — skips processing if the output directory already exists.

        Args:
            dms_id: Dataset identifier matching a subdirectory in ``input_dir``.

        Returns:
            Path to the output directory for this dataset.

        Raises:
            FileNotFoundError: If the input dataset directory does not exist.
            ValueError: If mutation notation is malformed or position mismatches WT.
        """
        print(f"[DataPreprocessor] output_base={self.output_base} | dataset={dms_id}")

        output_dir = self.output_base / dms_id
        if output_dir.exists():
            print(f"   {output_dir} already exists. Skipping.")
            return output_dir

        input_dms_dir = self.input_dir / dms_id
        if not input_dms_dir.exists():
            raise FileNotFoundError(
                f"Input dataset directory not found: {input_dms_dir}"
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"   Created: {output_dir}")

        wt_seq = self._load_wildtype(input_dms_dir)
        df = self._load_dms(input_dms_dir, wt_seq)

        shutil.copy(input_dms_dir / "wildtype.fasta", output_dir / "wt.fasta")
        df.to_csv(output_dir / "data.csv", index=True)
        print(f"   Successfully prepared {dms_id} → {output_dir}\n")

        return output_dir

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_wildtype(self, dataset_dir: Path) -> str:
        """Parse the wild-type FASTA and return the sequence string."""
        fasta_path = dataset_dir / "wildtype.fasta"
        return str(next(SeqIO.parse(fasta_path, "fasta")).seq)

    def _load_dms(self, dataset_dir: Path, wt_seq: str) -> pd.DataFrame:
        """Load and normalise the DMS TSV into the canonical DataFrame."""
        df = pd.read_csv(dataset_dir / "proteingym_dms.tsv", sep="\t")

        if "mutated_sequence" not in df.columns:
            df["mutated_sequence"] = df["mutant"].apply(
                lambda m: self._apply_mutations(m, wt_seq)
            )

        df = df.rename(columns={"mutated_sequence": "seq", "DMS_score": "log_fitness"})
        df = df.reset_index()
        df["mutated_position"] = df["mutant"].apply(self._extract_positions)
        df["n_mut"] = df["mutant"].apply(lambda x: len(x.split(":")))
        df["PID"] = df.index.astype(str)

        return df[["seq", "log_fitness", "n_mut", "mutant", "PID", "mutated_position"]]

    @staticmethod
    def _apply_mutations(mutant: str, wt_seq: str) -> str:
        """Apply colon-separated mutation notation to a wild-type sequence.

        Args:
            mutant: Colon-separated mutations in the form ``W123M``.
            wt_seq: Wild-type sequence string.

        Returns:
            Mutated sequence string.

        Raises:
            ValueError: On malformed notation or position mismatch.
        """
        seq_list = list(wt_seq)
        for mut in mutant.split(":"):
            if len(mut) < 3:
                raise ValueError(f"Malformed mutant token: '{mut}'")
            wild_aa, pos_str, mut_aa = mut[0], mut[1:-1], mut[-1]
            pos = int(pos_str)
            if seq_list[pos - 1] != wild_aa:
                raise ValueError(
                    f"Mismatch at pos {pos}: expected {wild_aa}, "
                    f"found {seq_list[pos - 1]}"
                )
            seq_list[pos - 1] = mut_aa
        return "".join(seq_list)

    @staticmethod
    def _extract_positions(mutant: str) -> object:
        """Extract 0-indexed mutation position(s) from mutation notation.

        Args:
            mutant: Colon-separated mutation string.

        Returns:
            A single ``int`` for single-site mutations, or a comma-separated
            ``str`` of integers for multi-site mutations.
        """
        positions = [int(m[1:-1]) - 1 for m in mutant.split(":")]
        return positions[0] if len(positions) == 1 else ",".join(map(str, positions))