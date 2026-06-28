"""Artifact cleanup CLI — thin wrapper around ArtifactCleaner."""

from __future__ import annotations

import argparse
from pathlib import Path

from confit.utils.cleanup import ArtifactCleaner


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--base_dir", type=str, default="/work/shannon/fine-tune_plm")
    args = p.parse_args()

    if args.dry_run:
        print("=== DRY RUN — no files will be changed ===\n")

    cleaner = ArtifactCleaner(base_dir=Path(args.base_dir), dry_run=args.dry_run)
    cleaner.clean_predicted()
    cleaner.clean_checkpoint()