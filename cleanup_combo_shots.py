"""Delete prior runs of one specific (a_type, a_init, combined_way, train_mode)
combo across a given set of shots, before re-running the shot sweep.

Scoped narrowly on purpose: only removes the exact checkpoint/predicted
folders matching this combo + these shots, for all datasets and seeds 1-5.
Does not touch any other combo's results, and does not touch data_rerun_fixed/
splits (those are independently keyed by shot now and regenerate themselves).

Usage:
  python cleanup_combo_shots.py              # dry run (default) — just prints
  python cleanup_combo_shots.py --confirm    # actually deletes
"""

import shutil
import sys
from pathlib import Path

A_TYPE       = "position-specific"
A_INIT       = -1.0
COMBINED_WAY = "scores"
TRAIN_MODE   = "full"
SHOTS        = [48, 64, 96, 144, 192, 240]
SEEDS        = [1, 2, 3, 4, 5]

BASE       = Path("/work/shannon/fine-tune_plm")
CKPT_ROOT  = BASE / "checkpoint_rerun_fixed"
PRED_ROOT  = BASE / "predicted_rerun_fixed"

CONFIRM = "--confirm" in sys.argv

mode_name = f"mode{A_TYPE}_ainit{A_INIT}_combined{COMBINED_WAY}_trainmode{TRAIN_MODE}"


def rmtree(path: Path) -> None:
    print(("[DRY RUN] DELETE " if not CONFIRM else "DELETE ") + str(path.relative_to(BASE)))
    if CONFIRM:
        shutil.rmtree(str(path))


def main() -> None:
    ckpt_count = 0
    pred_count = 0

    if CKPT_ROOT.exists():
        for dataset_dir in sorted(CKPT_ROOT.iterdir()):
            if not dataset_dir.is_dir():
                continue
            for shot in SHOTS:
                for seed in SEEDS:
                    target = dataset_dir / f"shot{shot}" / f"seed{seed}" / mode_name
                    if target.exists():
                        rmtree(target)
                        ckpt_count += 1

    if PRED_ROOT.exists():
        for dataset_dir in sorted(PRED_ROOT.iterdir()):
            if not dataset_dir.is_dir():
                continue
            for shot in SHOTS:
                for seed in SEEDS:
                    target = dataset_dir / f"shot{shot}_seed{seed}_{mode_name}"
                    if target.exists():
                        rmtree(target)
                        pred_count += 1

    print()
    print(f"{'Would delete' if not CONFIRM else 'Deleted'}: "
          f"{ckpt_count} checkpoint dirs, {pred_count} predicted dirs")
    if not CONFIRM:
        print("Dry run only — rerun with --confirm to actually delete.")


if __name__ == "__main__":
    main()
