"""Experiment runner — orchestrates dataset iteration and subprocess dispatch.

Replaces the ~270-line script-level code in the original ``run.py`` that
executed at import time and hardcoded an absolute ``BASE`` path.

The runner reads dataset metadata, builds the combination grid, and launches
``accelerate launch confit/train_entry.py ...`` for each combination via
``subprocess``.  All path resolution is relative so the code is portable.
"""

from __future__ import annotations

import itertools
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import pandas as pd
from Bio import SeqIO


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Combo = Tuple[str, float, str, str]


class RunMode(str, Enum):
    """Operating mode for the experiment runner.

    Attributes:
        MAIN: Standard run with the configured shot count.
        RERUN_SHOT64: Re-run shot-64 affected combinations.
        RERUN_SHOT96: Re-run shot-96 affected combinations.
        RERUN_ALL_FAILED: Re-run all missing combinations.
        RERUN_ALLNONE_FAILED: Re-run failed none+scores+full datasets.
        CHECK_STATUS: Print missing combination status and exit.
    """

    MAIN = "main"
    RERUN_SHOT64 = "rerun_shot64"
    RERUN_SHOT96 = "rerun_shot96"
    RERUN_ALL_FAILED = "rerun_all_failed"
    RERUN_ALLNONE_FAILED = "rerun_allnone_failed"
    CHECK_STATUS = "check_status"


@dataclass
class RunnerConfig:
    """Configuration for :class:`ExperimentRunner`.

    Attributes:
        proteingym_dir: Root directory containing raw ProteinGym datasets.
        base_dir: Project root (used to resolve checkpoint/predicted paths).
        run_suffix: Suffix appended to ``checkpoint_`` and ``predicted_`` dirs.
        quarter: Dataset quarter index (1–4) to process.
        shot: Number of training examples (k-shot).
        mode: Operating mode controlling which combos are run.
        check_shot: Shot count used in check_status mode.
        max_datasets: Limit to the first N datasets (for quick testing).
        last_datasets: Limit to the last N datasets.
        sleep_between_runs: Seconds to sleep between subprocess calls.
        accel_config: Path to the accelerate config YAML.
        train_script: Path to the training entry-point script.
        training_config: Path to the training YAML config.
    """

    proteingym_dir: Path = Path("/work/yunan/PsiFit/data/proteingym")
    base_dir: Path = Path(".")
    run_suffix: str = "rerun_fixed"
    quarter: int = 1
    shot: int = 96
    mode: RunMode = RunMode.MAIN
    check_shot: int = 96
    max_datasets: Optional[int] = None
    last_datasets: Optional[int] = None
    sleep_between_runs: float = 3.0
    accel_config: Path = Path("config/parallel_config.yaml")
    train_script: Path = Path("confit/train_entry.py")
    training_config: Path = Path("config/training_config.yaml")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class ExperimentRunner:
    """Iterates over datasets and combinations, launching training subprocesses.

    Args:
        config: :class:`RunnerConfig` controlling all runner behaviour.

    Example:
        >>> cfg = RunnerConfig(quarter=1, shot=96, mode=RunMode.MAIN)
        >>> runner = ExperimentRunner(cfg)
        >>> runner.run()
    """

    _ALL_A_TYPES = ["single", "position-specific", "context-specific", "none"]
    _ALL_A_INITS = [-1.0, 0.1]
    _ALL_COMBINED_WAYS = ["scores", "logits"]
    _ALL_TRAIN_MODES = ["full", "a_only"]
    _MAX_SEQ_LEN = 1022

    def __init__(self, config: RunnerConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute the full experiment loop for the configured quarter and mode."""
        df = self._load_dataset_index()
        df = self._apply_quarter_slice(df)
        df = self._apply_dataset_limit(df)

        combos, shot, skip_done = self._build_combos()

        print(f"Total valid combinations after filtering: {len(combos)}\n")

        if self.config.mode == RunMode.CHECK_STATUS:
            self._check_status(df, combos)
            sys.exit(0)

        self._launch_all(df, combos, shot, skip_done)

    # ------------------------------------------------------------------
    # Dataset index
    # ------------------------------------------------------------------

    def _load_dataset_index(self) -> pd.DataFrame:
        """Load and filter the ProteinGym dataset index.

        Returns:
            DataFrame with columns ``dms_id`` and ``seq_length``, sorted by
            ``seq_length`` ascending, excluding Tsuboyama datasets and those
            longer than :attr:`_MAX_SEQ_LEN`.
        """
        records = []
        for dataset_dir in self.config.proteingym_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            fasta_path = dataset_dir / "wildtype.fasta"
            if not fasta_path.exists():
                continue
            try:
                seq = str(next(SeqIO.parse(fasta_path, "fasta")).seq)
                records.append({"dms_id": dataset_dir.name, "seq_length": len(seq)})
            except Exception as exc:  # noqa: BLE001
                print(f"Error reading {fasta_path}: {exc}")

        df = pd.DataFrame(records)
        df = df[~df["dms_id"].str.contains("Tsuboyama")]
        df = df[df["seq_length"] <= self._MAX_SEQ_LEN]
        df.sort_values("seq_length", ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)

        print("=" * 80)
        print(f"Found {len(df)} datasets with seq_length <= {self._MAX_SEQ_LEN}")
        print(df[["dms_id", "seq_length"]].to_string(index=False))
        print("=" * 80)

        return df

    def _apply_quarter_slice(self, df: pd.DataFrame) -> pd.DataFrame:
        """Slice the dataset index to the configured quarter.

        Args:
            df: Full dataset index.

        Returns:
            Subset for the configured :attr:`RunnerConfig.quarter`.
        """
        n = len(df)
        q = self.config.quarter - 1
        starts = [0, n // 4, n // 2, 3 * n // 4]
        ends = [n // 4, n // 2, 3 * n // 4, n]
        sliced = df.iloc[starts[q] : ends[q]].reset_index(drop=True)
        print(
            f"Quarter {self.config.quarter}: datasets "
            f"{starts[q]+1}–{ends[q]} ({len(sliced)} total)"
        )
        return sliced

    def _apply_dataset_limit(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply optional ``max_datasets`` / ``last_datasets`` limits.

        Args:
            df: Quarter-sliced dataset index.

        Returns:
            Potentially further-limited DataFrame.
        """
        cfg = self.config
        if cfg.max_datasets is not None and cfg.max_datasets > 0:
            df = df.head(cfg.max_datasets).reset_index(drop=True)
            print(f"→ SAMPLE MODE: limiting to first {len(df)} datasets")
        elif cfg.last_datasets is not None and cfg.last_datasets > 0:
            df = df.tail(cfg.last_datasets).reset_index(drop=True)
            print(f"→ SAMPLE MODE: limiting to last {len(df)} datasets")
        print("=" * 80)
        return df

    # ------------------------------------------------------------------
    # Combination grid
    # ------------------------------------------------------------------

    def _build_combos(self) -> Tuple[List[Combo], int, bool]:
        """Build the list of hyper-parameter combinations for the active mode.

        Returns:
            Tuple of ``(combos, shot, skip_done)`` where *skip_done* is
            ``True`` when already-completed runs should be silently skipped.
        """
        mode = self.config.mode
        skip_done = False

        base_combos: List[Combo] = []
        for combo in itertools.product(
            self._ALL_A_TYPES,
            self._ALL_A_INITS,
            self._ALL_COMBINED_WAYS,
            self._ALL_TRAIN_MODES,
        ):
            a_type, a_init, combined_way, train_mode = combo
            if a_type == "none":
                if combined_way == "scores" and train_mode == "full" and a_init == 0.1:
                    base_combos.append(combo)
            else:
                base_combos.append(combo)

        if mode == RunMode.MAIN:
            return base_combos, self.config.shot, False

        if mode == RunMode.RERUN_SHOT64:
            return [("context-specific", -1.0, "scores", "a_only")], 64, False

        if mode == RunMode.RERUN_SHOT96:
            full_affected: List[Combo] = [
                ("single", 0.1, "logits", "full"),
                ("single", -1.0, "logits", "full"),
                ("none", 0.1, "scores", "full"),
                ("none", -1.0, "scores", "full"),
            ]
            a_only_affected: List[Combo] = list(itertools.product(
                ["single", "position-specific", "context-specific"],
                [0.1, -1.0], ["logits", "scores"], ["a_only"],
            ))
            return full_affected + a_only_affected, 96, False

        if mode == RunMode.RERUN_ALL_FAILED:
            skip_done = True
            return base_combos, 64, skip_done

        if mode == RunMode.RERUN_ALLNONE_FAILED:
            return [("none", 0.1, "scores", "full")], 96, False

        if mode == RunMode.CHECK_STATUS:
            return base_combos, self.config.check_shot, False

        raise ValueError(f"Unsupported RunMode: {mode!r}")

    # ------------------------------------------------------------------
    # Status check
    # ------------------------------------------------------------------

    def _check_status(self, df: pd.DataFrame, combos: List[Combo]) -> None:
        """Print a per-dataset summary of done vs. missing combinations.

        Args:
            df: Dataset index DataFrame.
            combos: List of (a_type, a_init, combined_way, train_mode) tuples.
        """
        for shot in [64, 96]:
            print(f"\n{'='*80}")
            print(
                f"Checking shot={shot}, quarter={self.config.quarter} | "
                f"{len(df)} datasets x {len(combos)} combos"
            )
            print("=" * 80)
            total_done = total_missing = 0
            for _, row in df.iterrows():
                dataset = row["dms_id"]
                missing = [
                    c for c in combos
                    if not self._is_combo_done(dataset, shot, 1, c)
                ]
                done_count = len(combos) - len(missing)
                total_done += done_count
                total_missing += len(missing)
                if missing:
                    print(f"\n  {dataset}  [{done_count}/{len(combos)} done]")
                    for a_type, a_init, combined_way, train_mode in missing:
                        print(
                            f"    MISSING: {a_type} | {a_init} | "
                            f"{combined_way} | {train_mode}"
                        )
                else:
                    print(f"  {dataset}  [ALL DONE]")
            print(f"\n{'='*80}")
            print(
                f"SUMMARY shot={shot}: {total_done} done, "
                f"{total_missing} missing out of "
                f"{len(df)*len(combos)} total"
            )
            print("=" * 80)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _launch_all(
        self,
        df: pd.DataFrame,
        combos: List[Combo],
        shot: int,
        skip_done: bool,
    ) -> None:
        """Iterate datasets × combos and launch training subprocesses.

        Args:
            df: Dataset index DataFrame.
            combos: Hyper-parameter combinations to run.
            shot: k-shot size for this run.
            skip_done: If ``True``, silently skip completed combinations.
        """
        total_runs = len(df) * len(combos)
        global_run = 0

        for idx, row in df.iterrows():
            dms_id = row["dms_id"]
            length = row["seq_length"]
            print(f"\n{'='*100}")
            print(f"DATASET {idx+1}/{len(df)} → {dms_id} (len={length})")
            print(f"Running {len(combos)} combinations")
            print(f"{'='*100}\n")

            for combo in combos:
                global_run += 1
                a_type, a_init, combined_way, train_mode = combo

                if skip_done and self._is_combo_done(dms_id, shot, 1, combo):
                    print(
                        f"  [{global_run}/{total_runs}] → SKIP (already done): "
                        f"a_type={a_type} | a_init={a_init} | "
                        f"combined_way={combined_way} | train_mode={train_mode}"
                    )
                    continue

                print(
                    f"  [{global_run}/{total_runs}] → a_type={a_type} | "
                    f"a_init={a_init} | combined_way={combined_way} | "
                    f"train_mode={train_mode}"
                )
                self._launch_one(dms_id, shot, combo)
                time.sleep(self.config.sleep_between_runs)

        print("\n" + "=" * 80)
        print(f"FINISHED! Ran {global_run} trainings on {len(df)} datasets.")
        print("=" * 80)

    def _launch_one(self, dms_id: str, shot: int, combo: Combo) -> None:
        """Launch a single training run as a subprocess.

        Args:
            dms_id: Dataset identifier.
            shot: k-shot count.
            combo: ``(a_type, a_init, combined_way, train_mode)`` tuple.
        """
        a_type, a_init, combined_way, train_mode = combo
        cfg = self.config
        cmd = [
            "accelerate", "launch",
            "--config_file", str(cfg.accel_config),
            "--main_process_port", str(29500 + cfg.quarter),
            str(cfg.train_script),
            "--config",       str(cfg.training_config),
            "--dataset",      dms_id,
            "--a_type",       a_type,
            "--a_init",       str(a_init),
            "--combined_way", combined_way,
            "--train_mode",   train_mode,
            "--sample_seed",  "0",
            "--model_seed",   "1",
            "--shot",         str(shot),
            "--run_suffix",   cfg.run_suffix,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=False)
            print("SUCCESS\n")
        except subprocess.CalledProcessError as exc:
            print(f"FAILED (code {exc.returncode})\n")
        except FileNotFoundError:
            print("FAILED (train script not found)\n")
            sys.exit(1)

    # ------------------------------------------------------------------
    # Completion checks
    # ------------------------------------------------------------------

    def _predicted_folder(
        self,
        dataset: str,
        shot: int,
        seed: int,
        a_type: str,
        a_init: float,
        combined_way: str,
        train_mode: str,
    ) -> Path:
        """Return the expected predicted-output folder path."""
        name = (
            f"shot{shot}_seed{seed}_mode{a_type}_ainit{a_init}"
            f"_combined{combined_way}_trainmode{train_mode}"
        )
        return (
            self.config.base_dir
            / f"predicted_{self.config.run_suffix}"
            / dataset
            / name
        )

    def _checkpoint_folder(
        self,
        dataset: str,
        shot: int,
        seed: int,
        a_type: str,
        a_init: float,
        combined_way: str,
        train_mode: str,
    ) -> Path:
        """Return the expected checkpoint folder path."""
        name = (
            f"mode{a_type}_ainit{a_init}"
            f"_combined{combined_way}_trainmode{train_mode}"
        )
        return (
            self.config.base_dir
            / f"checkpoint_{self.config.run_suffix}"
            / dataset
            / f"shot{shot}"
            / f"seed{seed}"
            / name
        )

    def _is_combo_done(
        self, dataset: str, shot: int, seed: int, combo: Combo
    ) -> bool:
        """Return True if both predicted output and checkpoint exist.

        Args:
            dataset: Dataset identifier.
            shot: k-shot count.
            seed: Model seed.
            combo: ``(a_type, a_init, combined_way, train_mode)`` tuple.

        Returns:
            ``True`` when the run is considered complete.
        """
        a_type, a_init, combined_way, train_mode = combo
        pred = self._predicted_folder(
            dataset, shot, seed, a_type, a_init, combined_way, train_mode
        )
        ckpt = self._checkpoint_folder(
            dataset, shot, seed, a_type, a_init, combined_way, train_mode
        )
        return (pred / "pred.csv").exists() and ckpt.exists() and any(ckpt.iterdir())

    def _get_failed_datasets(
        self, tail_part: str, log_dir: str = "logs"
    ) -> set:
        """Parse log files to identify datasets with failed runs.

        Args:
            tail_part: Suffix of log filenames (e.g. ``'allnone'``).
            log_dir: Directory containing ``q{1..4}_{tail_part}.log`` files.

        Returns:
            Set of dataset identifiers that appear as FAILED in any log.
        """
        failed: set = set()
        log_root = Path(log_dir)
        for q in [1, 2, 3, 4]:
            log_file = log_root / f"q{q}_{tail_part}.log"
            if not log_file.exists():
                continue
            current_dataset: Optional[str] = None
            with open(log_file) as fh:
                for line in fh:
                    line = line.strip()
                    if "DATASET" in line and "→" in line and "(len=" in line:
                        current_dataset = (
                            line.split("→")[1].split("(len=")[0].strip()
                        )
                    elif line.startswith("FAILED") and current_dataset:
                        failed.add(current_dataset)
        return failed