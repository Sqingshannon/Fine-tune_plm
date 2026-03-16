from pathlib import Path
import os
import shutil

def clean_predicted(predicted_dir: Path):
    for folder in predicted_dir.iterdir():
        if folder.is_dir():
            dataset_name = folder.name
            temp_count = 0
            for file in folder.iterdir():
                first_term = file.name.split("_")[0]  # e.g. "shot64" from "shot64_seed1_..."
                if first_term not in ["shot64", "shot96"]:
                    temp_count += 1
                    os.remove(file)
            print(f"Cleaned {temp_count} files in {dataset_name}")
    print("cleaning in predicted/ done.")

def clean_checkpoint(checkpoint_dir: Path):
    for folder in checkpoint_dir.iterdir():
        if folder.is_dir():
            dataset_name = folder.name
            temp_count = 0
            for subdir in folder.iterdir():
                if subdir.is_dir() and subdir.name not in ["shot64", "shot96"]:
                    temp_count += 1
                    shutil.rmtree(subdir)
            print(f"Cleaned {temp_count} subdirs in {dataset_name}")
    print("cleaning in checkpoint/ done.")

if __name__ == "__main__":
    base = Path("/work/shannon/fine-tune_plm")
    clean_predicted(base / "predicted")
    clean_checkpoint(base / "checkpoint")
