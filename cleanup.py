"""Project structure cleanup script.

What this does:
  1. For each dataset in data_rerun_fixed/:
       - Copy spurs_prediction.tsv from data/{dataset}/ (if present)
       - Move existing train*.csv / test.csv into seed_1/ subfolder
  2. Delete all dataset folders in data/ — keeps checkpoints/, dataset/,
       enzyme/, inference_example/
  3. Delete checkpoint/ and predicted/ if --delete-old is passed

Usage:
  python cleanup.py --dry-run              # preview only
  python cleanup.py                        # steps 1 & 2 only
  python cleanup.py --delete-old           # steps 1, 2 & 3
"""

import shutil
import sys
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
DRY_RUN     = "--dry-run"     in sys.argv
DELETE_OLD  = "--delete-old"  in sys.argv
CLEAN_SEEDS = "--clean-seeds" in sys.argv   # delete seed 2-5 from checkpoint/predicted _rerun_fixed

BASE        = Path("/work/shannon/fine-tune_plm")
DATA        = BASE / "data"
DATA_RERUN  = BASE / "data_rerun_fixed"
KEEP_IN_DATA = {"checkpoints", "dataset", "enzyme", "inference_example"}
SPLIT_FILES  = ["train.csv", "test.csv"] + [f"train_{i}.csv" for i in range(1, 6)]
# ────────────────────────────────────────────────────────────────────────────


def log(msg: str) -> None:
    print(("[DRY RUN] " if DRY_RUN else "") + msg)


def move(src: Path, dst: Path) -> None:
    log(f"  MOVE  {src.relative_to(BASE)}  →  {dst.relative_to(BASE)}")
    if not DRY_RUN:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))


def copy(src: Path, dst: Path) -> None:
    log(f"  COPY  {src.relative_to(BASE)}  →  {dst.relative_to(BASE)}")
    if not DRY_RUN:
        shutil.copy2(str(src), str(dst))


def rmtree(path: Path) -> None:
    log(f"  DELETE  {path.relative_to(BASE)}/")
    if not DRY_RUN:
        shutil.rmtree(str(path))




# ── Step 1: Organise data_rerun_fixed/ ──────────────────────────────────────
print("=" * 70)
print("Step 1 — Organise data_rerun_fixed/ datasets")
print("=" * 70)

datasets = sorted(p.name for p in DATA_RERUN.iterdir() if p.is_dir())
spurs_missing = []

for dataset in datasets:
    rerun_dir = DATA_RERUN / dataset
    seed1_dir = rerun_dir / "seed_1"
    src_spurs = DATA / dataset / "spurs_prediction.tsv"
    dst_spurs = rerun_dir / "spurs_prediction.tsv"

    print(f"\n  [{dataset}]")

    # Copy spurs_prediction.tsv from data/
    if src_spurs.exists():
        if not dst_spurs.exists():
            copy(src_spurs, dst_spurs)
        else:
            log("  spurs_prediction.tsv already present — skip")
    else:
        log("  WARNING: spurs_prediction.tsv not found in data/ — skipping")
        spurs_missing.append(dataset)

    # Move existing flat split files into seed_1/
    to_move = [f for f in SPLIT_FILES if (rerun_dir / f).exists()]
    if to_move:
        if not DRY_RUN:
            seed1_dir.mkdir(exist_ok=True)
        else:
            log(f"  mkdir seed_1/")
        for fname in to_move:
            move(rerun_dir / fname, seed1_dir / fname)
    else:
        log("  No flat split files found (seed_1/ may already be populated)")

if spurs_missing:
    print(f"\n  ⚠  {len(spurs_missing)} dataset(s) have no spurs_prediction.tsv in data/:")
    for d in spurs_missing:
        print(f"     {d}")

# ── Step 2: Delete dataset folders in data/ ──────────────────────────────────
print("\n" + "=" * 70)
print("Step 2 — Remove dataset folders from data/  (keeping special subfolders)")
print("=" * 70)

for item in sorted(DATA.iterdir()):
    if item.name in KEEP_IN_DATA:
        log(f"  KEEP  data/{item.name}/")
    elif item.is_dir():
        rmtree(item)
    else:
        log(f"  SKIP  data/{item.name}  (not a directory)")

# ── Step 3: Delete checkpoint/ and predicted/ ────────────────────────────────
print("\n" + "=" * 70)
print("Step 3 — Delete checkpoint/ and predicted/")
print("=" * 70)

if not DELETE_OLD:
    print("  Skipping — pass --delete-old to delete checkpoint/ and predicted/")
else:
    for folder_name in ["checkpoint", "predicted"]:
        folder = BASE / folder_name
        keep   = BASE / f"{folder_name}_rerun_fixed"
        if not folder.exists():
            log(f"  {folder_name}/ not found — skip")
            continue
        if not keep.exists():
            log(f"  WARNING: {folder_name}_rerun_fixed/ missing — skipping {folder_name}/ deletion for safety")
            continue
        rmtree(folder)

# ── Step 4: Remove seed 2-5 from checkpoint_rerun_fixed/ and predicted_rerun_fixed/ ──
print("\n" + "=" * 70)
print("Step 4 — Clean seed 2-5 from checkpoint_rerun_fixed/ and predicted_rerun_fixed/")
print("=" * 70)

SEEDS_TO_CLEAN = {2, 3, 4, 5}

if not CLEAN_SEEDS:
    print("  Skipping — pass --clean-seeds to remove seed 2-5 results")
else:
    # checkpoint_rerun_fixed/{dataset}/shot{N}/seed{M}/
    ckpt_root = BASE / "checkpoint_rerun_fixed"
    if ckpt_root.exists():
        for dataset_dir in sorted(ckpt_root.iterdir()):
            if not dataset_dir.is_dir():
                continue
            for shot_dir in sorted(dataset_dir.iterdir()):
                if not shot_dir.is_dir():
                    continue
                for seed_dir in sorted(shot_dir.iterdir()):
                    if not seed_dir.is_dir():
                        continue
                    # folder is named e.g. "seed2" or "seed1"
                    name = seed_dir.name
                    if name.startswith("seed"):
                        try:
                            s = int(name[4:])
                        except ValueError:
                            continue
                        if s in SEEDS_TO_CLEAN:
                            rmtree(seed_dir)
    else:
        log("  checkpoint_rerun_fixed/ not found — skip")

    # predicted_rerun_fixed/{dataset}/shot{N}_seed{M}_...
    pred_root = BASE / "predicted_rerun_fixed"
    if pred_root.exists():
        for dataset_dir in sorted(pred_root.iterdir()):
            if not dataset_dir.is_dir():
                continue
            for run_dir in sorted(dataset_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                for s in SEEDS_TO_CLEAN:
                    if f"_seed{s}_" in run_dir.name:
                        rmtree(run_dir)
                        break
    else:
        log("  predicted_rerun_fixed/ not found — skip")

print("\n" + "=" * 70)
print("Done." if not DRY_RUN else "Dry run complete — rerun without --dry-run to apply.")
print("=" * 70)
