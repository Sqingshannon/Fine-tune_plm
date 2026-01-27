import shutil
from pathlib import Path
import pandas as pd
import os
from Bio import SeqIO

def data_restruct(dms_id, input_dir=Path("/work/yunan/PsiFit/data/proteingym"), output_base=Path("./data")):
    predicted_dir = Path("/data/predicted") / dms_id
    if predicted_dir.exists():
        shutil.rmtree(predicted_dir)
    else:
        print(f"{predicted_dir} in predicted does not exist, no need to remove.")
    
    try:
        output_dms_dir = output_base / dms_id
        output_dms_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"{output_dms_dir} already exists. Skipping.")
        return
    
    wt_seq = str(next(SeqIO.parse(input_dir / dms_id / "wildtype.fasta", "fasta")).seq)
    df = pd.read_csv(input_dir / dms_id / "proteingym_dms.tsv", sep='\t')
    
    shutil.copy(input_dir / dms_id / "wildtype.fasta", output_dms_dir / "wt.fasta")
    
    if 'mutated_sequence' not in df.columns:
        def apply_mutations(mutant):
            seq_list = list(wt_seq)
            mutations = mutant.split(':')
            for mut in mutations:
                if len(mut) < 3:
                    raise ValueError(f"Invalid mutant format: {mut}")
                wild_aa, pos_str, mut_aa = mut[0], mut[1:-1], mut[-1]
                pos = int(pos_str)
                if seq_list[pos - 1] != wild_aa:
                    raise ValueError(f"Mismatch at position {pos}: expected {wild_aa}, found {seq_list[pos - 1]}")
                seq_list[pos - 1] = mut_aa
            return ''.join(seq_list)
        df['mutated_sequence'] = df['mutant'].apply(apply_mutations)
        
    df = df.rename(columns={'mutated_sequence': 'seq', 'DMS_score': 'log_fitness'})
    df = df.reset_index()
    
    def extract_positions(mutant):
        positions = [int(mut[1:-1])-1 for mut in mutant.split(':')]
        if len(positions) == 1:
            return positions[0]
        else:
            return ','.join(map(str, positions))
    df['mutated_position'] = df['mutant'].apply(extract_positions)
    
    df['PID'] = df.index.astype(str)
    
    relevant_cols = ['seq', 'log_fitness', 'PID', 'mutated_position', 'mutant']
    df = df[relevant_cols]
    
    df.to_csv(output_dms_dir / "test.csv", index=True)
    df.to_csv(output_dms_dir / "data.csv", index=True)
    
    print(f"Prepared data for {dms_id} in {output_dms_dir}")
