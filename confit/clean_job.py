from pathlib import Path
import os
import shutil
import argparse


def is_correct_old_format(name):
    """Return True if this old-format combination is bug-free and should be kept.
    Buggy combos (should be deleted):
      - trainmodea_only  (Issue 5)
      - modenone         (Issue 3)
      - modesingle + combinedlogits  (Issue 1)
    """
    if "trainmodea_only" in name:
        return False
    if "modenone" in name:
        return False
    if "modesingle" in name and "combinedlogits" in name:
        return False
    return True


def clean_predicted(predicted_dir: Path, dry_run: bool = False):
    for folder in predicted_dir.iterdir():
        if not folder.is_dir():
            continue
        dataset_name = folder.name
        renamed = deleted = 0
        for entry in folder.iterdir():
            first_term = entry.name.split("_")[0]
            if first_term in ["shot64", "shot96"]:
                continue  # already correct format, keep
            # Old format (e.g. seed1_mode...)
            if is_correct_old_format(entry.name):
                new_path = folder / ("shot96_" + entry.name)
                if not new_path.exists():
                    print(f"  RENAME: {entry.name}  →  {new_path.name}")
                    if not dry_run:
                        entry.rename(new_path)
                    renamed += 1
                else:
                    print(f"  DELETE (replaced): {entry.name}")
                    if not dry_run:
                        shutil.rmtree(entry) if entry.is_dir() else os.remove(entry)
                    deleted += 1
            else:
                print(f"  DELETE (buggy):    {entry.name}")
                if not dry_run:
                    shutil.rmtree(entry) if entry.is_dir() else os.remove(entry)
                deleted += 1
        print(f"{dataset_name}: renamed {renamed}, deleted {deleted}")
    print("cleaning in predicted/ done.")


def clean_checkpoint(checkpoint_dir: Path, dry_run: bool = False):
    for folder in checkpoint_dir.iterdir():
        if not folder.is_dir():
            continue
        dataset_name = folder.name
        renamed = deleted = 0
        for subdir in folder.iterdir():
            if not subdir.is_dir():
                continue
            if subdir.name in ["shot64", "shot96"]:
                continue  # already correct format, keep
            # Old format seed folder (e.g. "seed1") — go inside and process each combo
            seed_name = subdir.name
            for combo_dir in subdir.iterdir():
                if not combo_dir.is_dir():
                    continue
                if is_correct_old_format(combo_dir.name):
                    target = folder / "shot96" / seed_name / combo_dir.name
                    if not target.exists():
                        print(f"  MOVE: {seed_name}/{combo_dir.name}  →  shot96/{seed_name}/{combo_dir.name}")
                        if not dry_run:
                            target.parent.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(combo_dir), str(target))
                        renamed += 1
                    else:
                        print(f"  DELETE (replaced): {seed_name}/{combo_dir.name}")
                        if not dry_run:
                            shutil.rmtree(combo_dir)
                        deleted += 1
                else:
                    print(f"  DELETE (buggy):    {seed_name}/{combo_dir.name}")
                    if not dry_run:
                        shutil.rmtree(combo_dir)
                    deleted += 1
            # Remove old seed folder if now empty
            if not dry_run and subdir.exists() and not any(subdir.iterdir()):
                subdir.rmdir()
        print(f"{dataset_name}: renamed {renamed}, deleted {deleted}")
    print("cleaning in checkpoint/ done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would be done without making any changes")
    args = parser.parse_args()

    if args.dry_run:
        print("=== DRY RUN — no files will be changed ===\n")

    base = Path("/work/shannon/fine-tune_plm")
    clean_predicted(base / "predicted", dry_run=args.dry_run)
    clean_checkpoint(base / "checkpoint", dry_run=args.dry_run)
