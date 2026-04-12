from Bio import SeqIO
from pathlib import Path
import pandas as pd
import subprocess
import sys
import time
import itertools
import argparse

BASE = Path("/work/shannon/fine-tune_plm")

# def predicted_folder(dataset, shot, seed, a_type, a_init, combined_way, train_mode):
#     name = f"shot{shot}_seed{seed}_mode{a_type}_ainit{a_init}_combined{combined_way}_trainmode{train_mode}"
#     return BASE / "predicted" / dataset / name

# def checkpoint_folder(dataset, shot, seed, a_type, a_init, combined_way, train_mode):
#     name = f"mode{a_type}_ainit{a_init}_combined{combined_way}_trainmode{train_mode}"
#     return BASE / "checkpoint" / dataset / f"shot{shot}" / f"seed{seed}" / name

def predicted_folder(dataset, shot, seed, a_type, a_init, combined_way, train_mode, run_suffix):
    name = f"shot{shot}_seed{seed}_mode{a_type}_ainit{a_init}_combined{combined_way}_trainmode{train_mode}"
    return BASE / f"predicted_{run_suffix}" / dataset / name

def checkpoint_folder(dataset, shot, seed, a_type, a_init, combined_way, train_mode, run_suffix):
    name = f"mode{a_type}_ainit{a_init}_combined{combined_way}_trainmode{train_mode}"
    return BASE / f"checkpoint_{run_suffix}" / dataset / f"shot{shot}" / f"seed{seed}" / name

# def is_combo_done(dataset, shot, seed, combo):
#     a_type, a_init, combined_way, train_mode = combo
#     pred = predicted_folder(dataset, shot, seed, a_type, a_init, combined_way, train_mode)
#     ckpt = checkpoint_folder(dataset, shot, seed, a_type, a_init, combined_way, train_mode)
#     pred_done = (pred / "pred.csv").exists()
#     ckpt_done = ckpt.exists() and any(ckpt.iterdir())
#     return pred_done and ckpt_done

def is_combo_done(dataset, shot, seed, combo, run_suffix):
    a_type, a_init, combined_way, train_mode = combo
    pred = predicted_folder(dataset, shot, seed, a_type, a_init, combined_way, train_mode, run_suffix)
    ckpt = checkpoint_folder(dataset, shot, seed, a_type, a_init, combined_way, train_mode, run_suffix)
    pred_done = (pred / "pred.csv").exists()
    ckpt_done = ckpt.exists() and any(ckpt.iterdir())
    return pred_done and ckpt_done

def check_all_status(df, combinations, shot, seed=1):
    missing = []
    for _, row in df.iterrows():
        dataset = row['dms_id']
        for combo in combinations:
            if not is_combo_done(dataset, shot, seed, combo, args.run_suffix):
                missing.append((dataset, combo))
    print(f"\n{'='*80}")
    print(f"Status check: {len(df)*len(combinations)} total | "
          f"{len(df)*len(combinations)-len(missing)} done | {len(missing)} missing")
    for dataset, combo in missing:
        a_type, a_init, combined_way, train_mode = combo
        print(f"  MISSING: {dataset} | {a_type} {a_init} {combined_way} {train_mode}")
    print(f"{'='*80}\n")
    return missing
# def check_all_status(df, combinations, shot, seed=1, run_suffix="rerun_fixed"):
#     missing = []
#     for _, row in df.iterrows():
#         dataset = row['dms_id']
#         for combo in combinations:
#             if not is_combo_done(dataset, shot, seed, combo, run_suffix):
#                 missing.append((dataset, combo))
#     print(f"\n{'='*80}")
#     print(f"Status check: {len(df)*len(combinations)} total | "
#           f"{len(df)*len(combinations)-len(missing)} done | {len(missing)} missing")
#     for dataset, combo in missing:
#         a_type, a_init, combined_way, train_mode = combo
#         print(f"  MISSING: {dataset} | {a_type} {a_init} {combined_way} {train_mode}")
#     print(f"{'='*80}\n")
#     return missing

def get_undone_datasets(df, combinations, shot, seed=1):
    """Return subset of df where at least one combo is not done."""
    undone = []
    for _, row in df.iterrows():
        dataset = row['dms_id']
        if any(not is_combo_done(dataset, shot, seed, combo, args.run_suffix) for combo in combinations):
            undone.append(row)
    return pd.DataFrame(undone).reset_index(drop=True)

def get_failed_datasets(tail_part, log_dir="logs"):
    failed = set()
    log_dir = Path(log_dir)
    for q in [1, 2, 3, 4]:
        log_file = log_dir / f"q{q}_{tail_part}.log"
        if not log_file.exists():
            continue
        current_dataset = None
        with open(log_file) as f:
            for line in f:
                line = line.strip()
                if "DATASET" in line and "→" in line and "(len=" in line:
                    # Format: DATASET X/Y → dms_id (len=Z)
                    current_dataset = line.split("→")[1].split("(len=")[0].strip()
                elif line.startswith("FAILED") and current_dataset:
                    failed.add(current_dataset)
    return failed


parser = argparse.ArgumentParser()
parser.add_argument("--quarter", type=int, choices=[1, 2, 3, 4], required=True,
                    help="Which quarter of datasets to run (1-4)")
parser.add_argument("--mode", type=str, choices=["main", "rerun_shot64", "rerun_shot96", "rerun_all_failed", "check_status"],
                    default="main",
                    help="main: current run; rerun_shot64: re-run a_only shot=64; rerun_shot96: re-run affected shot=96 combos; rerun_all_failed: re-run all missing combos for shot=96; check_status: print missing combinations per dataset")
parser.add_argument("--check_shot", type=int, default=96,
                    help="Which shot to check when using check_status mode (default: 96)")

parser.add_argument('--run_suffix', type=str, default='rerun_fixed',
                    help='suffix for data/predicted/checkpoint folders (default: rerun_fixed)')
parser.add_argument("--max_datasets", type=int, default=None,
                        help="If set, only run the first N datasets in this quarter (great for quick testing)")
args = parser.parse_args()

base_dir = Path("/work/yunan/PsiFit/data/proteingym")
datasets = [d.name for d in base_dir.iterdir() if d.is_dir()]

data = []

for dataset in datasets:
    fasta_path = base_dir / dataset / "wildtype.fasta"
    if fasta_path.exists():
        try:
            record = next(SeqIO.parse(fasta_path, "fasta"))
            seq = str(record.seq)
            length = len(seq)
            data.append({'dms_id': dataset, 'seq_length': length})
        except Exception as e:
            print(f"Error processing {fasta_path}: {e}")
            
df = pd.DataFrame(data)
df = df[~df['dms_id'].str.contains("Tsuboyama")]
df.sort_values(by='seq_length', ascending=True, inplace=True)

# num_datasets = 1
# short_df = df.head(num_datasets).reset_index(drop=True)
short_df = df[df['seq_length'] <= 1022].copy().reset_index(drop=True)
num_datasets = len(short_df)

print("=" * 80)
print(f"Found {len(short_df)} datasets with seq_length < 1022")
print(short_df[['dms_id', 'seq_length']].to_string(index=False))
print("=" * 80)

n = len(short_df)
q = args.quarter - 1
starts = [0, n//4, n//2, 3*n//4]
ends   = [n//4, n//2, 3*n//4, n]
short_df = short_df.iloc[starts[q]:ends[q]].reset_index(drop=True)

# ==================== NEW: Sample / Test mode ====================
if args.max_datasets is not None and args.max_datasets > 0:
    short_df = short_df.head(args.max_datasets).reset_index(drop=True)
    print(f"→ SAMPLE MODE: limiting to first {len(short_df)} datasets only (--max_datasets={args.max_datasets})")
else:
    print(f"Quarter {args.quarter}: datasets {starts[q]+1}–{ends[q]} ({len(short_df)} total)")

num_datasets = len(short_df)
print("=" * 80)
# print(f"Quarter {args.quarter}: datasets {starts[q]+1}–{ends[q]} ({num_datasets} total)")

skip_done = False  # set to True in rerun_all_failed to skip already-completed combos

all_combos = list(itertools.product(
    ["single", "position-specific", "context-specific", "none"],
    [-1.0, 0.1], ["scores", "logits"], ["full", "a_only"]
))
filtered_combinations = []
for combo in all_combos:
    a_type, a_init, combined_way, train_mode = combo
    if a_type == "none":
        if combined_way == "scores" and train_mode == "full":
            if a_init == 0.1:
                filtered_combinations.append(combo)
    else:
        filtered_combinations.append(combo)

if args.mode.split("_")[0] == "main":
    if args.mode.split("_")[1] == "shot64":
        shot = 64
    elif args.mode.split("_")[1] == "shot96":
        shot = 96

elif args.mode == "rerun_shot64":
    # Issue 5: a_only with shot=64 had wrong checkpoint selection in Stage 1
    shot = 64
    filtered_combinations = [
        ("context-specific", -1.0, "scores", "a_only"),
    ]

elif args.mode == "rerun_shot96":
    # Issue 1: single+logits+full used wrong tensor
    # Issue 3: none+scores+full added ddg with a=1
    # Issue 5: all a_only combinations had wrong Stage 1 checkpoint selection
    shot = 96
    full_affected = [
        ("single",  0.1,  "logits", "full"),
        ("single",  -1.0, "logits", "full"),
        ("none",    0.1,  "scores", "full"),
        ("none",    -1.0, "scores", "full"),
    ]
    a_only_affected = list(itertools.product(
        ["single", "position-specific", "context-specific"],
        [0.1, -1.0],
        ["logits", "scores"],
        ["a_only"],
    ))
    filtered_combinations = full_affected + a_only_affected
    
# elif args.mode == "rerun_allnone":
#     shot = 96
#     filtered_combinations = [
#         ("none", 0.1,  "scores", "full")
#     ]
    
elif args.mode == "check_status":
    shot = args.check_shot

    for shot in [64, 96]:
        print(f"\n{'='*80}")
        print(f"Checking shot={shot}, quarter={args.quarter} | {len(short_df)} datasets x {len(filtered_combinations)} combos")
        print("=" * 80)
        total_done = total_missing = 0
        for _, row in short_df.iterrows():
            dataset = row['dms_id']
            missing = [(a, i, c, t) for (a, i, c, t) in filtered_combinations
                       if not is_combo_done(dataset, shot, 1, (a, i, c, t))]
            done_count = len(filtered_combinations) - len(missing)
            total_done += done_count
            total_missing += len(missing)
            if missing:
                print(f"\n  {dataset}  [{done_count}/{len(filtered_combinations)} done]")
                for a_type, a_init, combined_way, train_mode in missing:
                    print(f"    MISSING: {a_type} | {a_init} | {combined_way} | {train_mode}")
            else:
                print(f"  {dataset}  [ALL DONE]")
        print(f"\n{'='*80}")
        print(f"SUMMARY shot={shot}: {total_done} done, {total_missing} missing out of {len(short_df)*len(filtered_combinations)} total")
        print("=" * 80)
    sys.exit(0)

elif args.mode == "rerun_allnone_failed":
    shot = 96
    filtered_combinations = [
        ("none", 0.1,  "scores", "full")]
    failed_datasets = get_failed_datasets(tail_part="allnone")
    print(f"Found {len(failed_datasets)} failed datasets from logs: {failed_datasets}")
    short_df = short_df[short_df['dms_id'].isin(failed_datasets)].reset_index(drop=True)
    num_datasets = len(short_df)
    print(f"After filtering to failed datasets, {num_datasets} remain in the quarter: {short_df['dms_id'].tolist()}")

elif args.mode == "rerun_all_failed":
    # shot = args.check_shot  # use --check_shot to specify 64 or 96, default 96
    shot = 64
    skip_done = True
    all_combos = list(itertools.product(
        ["single", "position-specific", "context-specific", "none"],
        [-1.0, 0.1], ["scores", "logits"], ["full", "a_only"]
    ))
    filtered_combinations = []
    for combo in all_combos:
        a_type, a_init, combined_way, train_mode = combo
        if a_type == "none":
            if combined_way == "scores" and train_mode == "full":
                if a_init == 0.1:
                    filtered_combinations.append(combo)
        else:
            filtered_combinations.append(combo)
    num_datasets = len(short_df)
    print(f"Mode rerun_all_failed: shot={shot}, quarter={args.quarter}, {num_datasets} datasets")
    print("Will skip combinations already present in checkpoint/predicted folders.")

print(f"Total valid combinations after filtering: {len(filtered_combinations)}\n")

total_runs = len(short_df) * len(filtered_combinations)
global_run = 0

for idx, row in short_df.iterrows():
    dms_id = row['dms_id']
    length = row['seq_length']
    
    print(f"\n{'='*100}")
    print(f"DATASET {idx+1}/{len(short_df)} → {dms_id} (len={length})")
    print(f"Running {len(filtered_combinations)} combinations")
    print(f"{'='*100}\n")
    
    for combo in filtered_combinations:
        a_type, a_init, combined_way, train_mode = combo
        global_run += 1

        if skip_done and is_combo_done(dms_id, shot, 1, combo):
            print(f"  [{global_run}/{total_runs}] → SKIP (already done): "
                  f"a_type={a_type} | a_init={a_init} | combined_way={combined_way} | train_mode={train_mode}")
            continue

        print(f"  [{global_run}/{total_runs}] → a_type={a_type} | a_init={a_init} | "
              f"combined_way={combined_way} | train_mode={train_mode}")
        
        cmd = [
            "accelerate", "launch",
            "--config_file", "config/parallel_config.yaml",
            "--main_process_port", str(29500 + args.quarter),
            "confit/train.py",
            "--config", "config/training_config.yaml",
            "--dataset",       dms_id,
            "--a_type",        a_type,
            "--a_init",        str(a_init),
            "--combined_way",  combined_way,
            "--train_mode",    train_mode, 
            "--sample_seed", "0",
            "--model_seed", "1",
            "--shot", str(shot),
            "--run_suffix", args.run_suffix,
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=False)
            print("SUCCESS\n")
        except subprocess.CalledProcessError as e:
            print(f"FAILED (code {e.returncode})\n")
        except FileNotFoundError:
            print("FAILED (train.py not found)\n")
            sys.exit(1)
            
        try:
            import torch, gc
            torch.cuda.empty_cache()
            gc.collect()
        except:
            pass
        
        time.sleep(3)

print("\n" + "="*80)
print(f"FINISHED! Ran {global_run} trainings on the {num_datasets} smallest datasets.")
print("="*80)
