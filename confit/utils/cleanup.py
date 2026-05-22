"""Artifact cleaner — migrates old-format predicted and checkpoint directories.

Replaces the bare functions from ``clean_job.py``.  Encapsulates the detection
logic for buggy combinations in a single private method, and exposes the two
public operations (:meth:`clean_predicted` and :meth:`clean_checkpoint`)
through a cohesive class.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


class ArtifactCleaner:
    """Migrates old-format checkpoint and prediction directories.

    Applies three rules to directories under ``predicted/`` and ``checkpoint/``:

    * **Rename** — correct old-format combinations get a ``shot96_`` prefix added.
    * **Delete (buggy)** — combinations known to have bugs are removed.
    * **Delete (replaced)** — old-format entries where the new-format target
      already exists are removed.

    Buggy combinations (always deleted):
      - ``trainmodea_only``  (Issue 5)
      - ``modenone``         (Issue 3)
      - ``modesingle`` + ``combinedlogits`` (Issue 1)

    Args:
        base_dir: Project root directory containing ``predicted/`` and
            ``checkpoint/`` subdirectories.
        dry_run: When ``True``, print actions without modifying anything.

    Example:
        >>> cleaner = ArtifactCleaner(base_dir=Path("/work/shannon/fine-tune_plm"))
        >>> cleaner.clean_predicted()
        >>> cleaner.clean_checkpoint()
    """

    def __init__(
        self,
        base_dir: Path = Path("."),
        dry_run: bool = False,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.dry_run = dry_run

    def clean_predicted(self) -> None:
        """Rename or delete old-format entries in the ``predicted/`` directory."""
        predicted_dir = self.base_dir / "predicted"
        if not predicted_dir.exists():
            return

        for dataset_dir in predicted_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            renamed = deleted = 0
            for entry in dataset_dir.iterdir():
                first_term = entry.name.split("_")[0]
                if first_term in ("shot64", "shot96"):
                    continue
                if self._is_correct(entry.name):
                    new_path = dataset_dir / ("shot96_" + entry.name)
                    if not new_path.exists():
                        print(f"  RENAME: {entry.name}  →  {new_path.name}")
                        if not self.dry_run:
                            entry.rename(new_path)
                        renamed += 1
                    else:
                        print(f"  DELETE (replaced): {entry.name}")
                        if not self.dry_run:
                            self._remove(entry)
                        deleted += 1
                else:
                    print(f"  DELETE (buggy):    {entry.name}")
                    if not self.dry_run:
                        self._remove(entry)
                    deleted += 1
            print(f"{dataset_dir.name}: renamed {renamed}, deleted {deleted}")
        print("cleaning in predicted/ done.")

    def clean_checkpoint(self) -> None:
        """Rename or delete old-format entries in the ``checkpoint/`` directory."""
        checkpoint_dir = self.base_dir / "checkpoint"
        if not checkpoint_dir.exists():
            return

        for dataset_dir in checkpoint_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            renamed = deleted = 0
            for seed_subdir in dataset_dir.iterdir():
                if not seed_subdir.is_dir():
                    continue
                if seed_subdir.name in ("shot64", "shot96"):
                    continue
                seed_name = seed_subdir.name
                for combo_dir in seed_subdir.iterdir():
                    if not combo_dir.is_dir():
                        continue
                    if self._is_correct(combo_dir.name):
                        target = (
                            dataset_dir / "shot96" / seed_name / combo_dir.name
                        )
                        if not target.exists():
                            print(
                                f"  MOVE: {seed_name}/{combo_dir.name}  →  "
                                f"shot96/{seed_name}/{combo_dir.name}"
                            )
                            if not self.dry_run:
                                target.parent.mkdir(parents=True, exist_ok=True)
                                shutil.move(str(combo_dir), str(target))
                            renamed += 1
                        else:
                            print(f"  DELETE (replaced): {seed_name}/{combo_dir.name}")
                            if not self.dry_run:
                                shutil.rmtree(combo_dir)
                            deleted += 1
                    else:
                        print(f"  DELETE (buggy):    {seed_name}/{combo_dir.name}")
                        if not self.dry_run:
                            shutil.rmtree(combo_dir)
                        deleted += 1
                if not self.dry_run and seed_subdir.exists():
                    if not any(seed_subdir.iterdir()):
                        seed_subdir.rmdir()
            print(f"{dataset_dir.name}: renamed {renamed}, deleted {deleted}")
        print("cleaning in checkpoint/ done.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_correct(name: str) -> bool:
        """Return True if this old-format combination is bug-free.

        Buggy combinations (should always be deleted):
          - ``trainmodea_only``  (Issue 5)
          - ``modenone``         (Issue 3)
          - ``modesingle`` + ``combinedlogits`` (Issue 1)

        Args:
            name: Directory name to inspect.

        Returns:
            ``True`` if the combination should be kept (renamed), ``False``
            if it should be deleted.
        """
        if "trainmodea_only" in name:
            return False
        if "modenone" in name:
            return False
        if "modesingle" in name and "combinedlogits" in name:
            return False
        return True

    @staticmethod
    def _remove(path: Path) -> None:
        """Remove a file or directory tree.

        Args:
            path: File or directory to remove.
        """
        if path.is_dir():
            shutil.rmtree(path)
        else:
            os.remove(path)