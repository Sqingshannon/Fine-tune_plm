from Bio import SeqIO
from pathlib import Path
import pandas as pd
import subprocess
import sys
import time
import itertools

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

num_datasets = 5
short_df = df.head(num_datasets).reset_index(drop=True)
# short_df = df[df['seq_length'] <= 1022].copy().reset_index(drop=True)

print("=" * 80)
print(f"Found {len(short_df)} datasets with seq_length < 1022")
print(short_df[['dms_id', 'seq_length']].to_string(index=False))
print("=" * 80)

a_types       = ["single", "position-specific", "context-specific"]
a_inits       = [0.1, -1.0]
combined_ways = ["logits", "scores"]

combinations = list(itertools.product(a_types, a_inits, combined_ways))

total_runs = len(short_df) * len(combinations)
global_run = 0

for idx, row in short_df.iterrows():
    dms_id = row['dms_id']
    length = row['seq_length']
    
    print(f"\n{'='*100}")
    print(f"DATASET {idx+1}/{len(short_df)} → {dms_id} (len={length})")
    print(f"Running {len(combinations)} combinations")
    print(f"{'='*100}\n")
    
    for combo in combinations:
        a_type, a_init, combined_way = combo
        global_run += 1
        
        print(f"  [{global_run}/{total_runs}] → a_type={a_type} | a_init={a_init} | combined_way={combined_way}")
        
        cmd = [
            "accelerate", "launch",
            "--config_file", "config/parallel_config.yaml",
            "confit/train.py",
            "--config", "config/training_config.yaml",
            "--dataset",       dms_id,
            "--a_type",        a_type,
            "--a_init",        str(a_init),
            "--combined_way",  combined_way,
            "--sample_seed", "0",
            "--model_seed", "1",
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