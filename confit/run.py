from Bio import SeqIO
from pathlib import Path
import pandas as pd
import subprocess
import sys
import time
import itertools
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--quarter", type=int, choices=[1, 2, 3, 4], required=True,
                    help="Which quarter of datasets to run (1-4)")
parser.add_argument("--mode", type=str, choices=["main", "rerun_shot64", "rerun_shot96", "rerun_allnone"],
                    default="main",
                    help="main: current run; rerun_shot64: re-run a_only shot=64; rerun_shot96: re-run affected shot=96 combos; rerun_allnone: re-run none+scores+full for shot=96")
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
num_datasets = len(short_df)
print(f"Quarter {args.quarter}: datasets {starts[q]+1}–{ends[q]} ({num_datasets} total)")

if args.mode.split("_")[0] == "main":
    if args.mode.split("_")[1] == "shot64":
        shot = 64
    elif args.mode.split("_")[1] == "shot96":
        shot = 96
        
    a_types       = ["single", "position-specific", "context-specific", "none"]
    a_inits       = [-1.0, 0.1]
    combined_ways = ["scores", "logits"]
    train_modes   = ["full", "a_only"]
    combinations  = list(itertools.product(a_types, a_inits, combined_ways, train_modes))
    filtered_combinations = []
    for combo in combinations:
        a_type, a_init, combined_way, train_mode = combo
        if a_type == "none":
            if combined_way == "scores" and train_mode == "full":
                filtered_combinations.append(combo)
        else:
            filtered_combinations.append(combo)

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
    
elif args.mode == "rerun_allnone":
    shot = 96
    filtered_combinations = [
        ("none", 0.1,  "scores", "full")
    ]
    

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