import pandas as pd
import numpy as np
from pathlib import Path
from Bio import SeqIO
import torch
from torch.utils.data import DataLoader, Dataset

class DMSDataset(Dataset):
    def __init__(self, df, tokenizer, wt_seq, max_length=1024):
        self.df = df
        self.tokenizer = tokenizer
        self.wt_seq = wt_seq
        self.data = self._build_data(df)
    
    def _build_data(self, df):
        seq_enc = self.tokenizer(df['mutated_sequence'].tolist(),
            return_tensors="pt", padding=False, truncation=False)
        seqs, masks = seq_enc['input_ids'], seq_enc['attention_mask']
        wt_seq_str = [self.wt_seq] * len(df)
        wt_seq_enc = self.tokenizer(wt_seq_str, return_tensors="pt", padding=False, truncation=False)
        wt_seqs, wt_masks = wt_seq_enc['input_ids'], wt_seq_enc['attention_mask']
        
        # create mutated_pos tensor: indicating the mutated position (0-based index)
        mutated_pos = torch.tensor(
            [int(mutation[1:-1]) - 1 for mutation in df['mutant']],
            dtype=torch.long
        )
        label = torch.tensor(df['DMS_score'].values, dtype=torch.float32)

        data = {'seq': seqs,
            'attn_mask': masks,
            'wt': wt_seqs,
            'wt_attn_mask': wt_masks,
            'mutated_pos': mutated_pos,
            'label': label,
        }
        return data

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            'seq': self.data['seq'][idx],
            'attn_mask': self.data['attn_mask'][idx],
            'wt': self.data['wt'][idx],
            'wt_attn_mask': self.data['wt_attn_mask'][idx],
            'mutated_pos': self.data['mutated_pos'][idx],
            'label': self.data['label'][idx],
        }


class ProteinGymTask(object):
    def __init__(self, root='../../data/proteingym', dms_id=None, seed=42):
        self.task_dir = Path(root) / dms_id
        self.dms = pd.read_table(self.task_dir / 'proteingym_dms.tsv')
        self.wt_seq = str(SeqIO.read(self.task_dir / 'wildtype.fasta', 'fasta').seq)
        self.ddg = pd.read_table(self.task_dir / 'spurs_prediction.tsv', index_col=0)
        self.seed = seed
    
        self.dms['mutated_sequence'] = self._build_mutated_sequence(self.dms)

    def _build_mutated_sequence(self, dms):
        # the `mutant` column in dms has the format like 'D28E:T32K';
        # return a list of mutated sequences
        mutated_seqs = []
        for mutant in dms['mutant']:
            seq = list(str(self.wt_seq))
            for mutation in mutant.split(':'):
                wt_aa = mutation[0]
                pos = int(mutation[1:-1]) - 1  # convert to 0-based index
                mut_aa = mutation[-1]
                assert seq[pos] == wt_aa, f"Expected {wt_aa} at position {pos+1}, found {seq[pos]}"
                seq[pos] = mut_aa
            mutated_seqs.append(''.join(seq))
        return mutated_seqs
    
    def split(self, k, test_frac=0.2, valid_frac=0.2, seed=None):
        """
        :param k: number of shots for training set
        :param test_frac: fraction of data to use as test set. The rest is used as training pool.
        :param valid_frac: fraction of the k shots to use as validation set.
        """
        seed = seed if seed is not None else self.seed
        test_df = self.dms.sample(frac=test_frac, random_state=seed)
        train_data_pool = self.dms.drop(test_df.index)
        if k >= len(train_data_pool):
            print(f"Requested k={k} exceeds available training data size {len(train_data_pool)}. Returning all training data.")
            train_df = train_data_pool
        train_df = train_data_pool.sample(n=k, random_state=seed)
        valid_df = train_df.sample(frac=valid_frac, random_state=seed)
        train_df = train_df.drop(valid_df.index)
        return train_df.reset_index(drop=True), valid_df.reset_index(drop=True), test_df.reset_index(drop=True)
    
    def get_dataloader(self, train_data, valid_data, test_data, tokenizer, batch_size=16):
        train_dataset = DMSDataset(train_data, tokenizer, self.wt_seq)
        valid_dataset = DMSDataset(valid_data, tokenizer, self.wt_seq)
        test_dataset = DMSDataset(test_data, tokenizer, self.wt_seq)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    task = ProteinGymTask(dms_id='IF1_ECOLI_Kelsic_2016')
    train_set, valid_set, test_set = task.split(k=96, test_frac=0.2, valid_frac=0.2, seed=42)
    print(f"\nNumber of training samples: {len(train_set)}")
    print(f"Number of validation samples: {len(valid_set)}")
    print(f"Number of test samples: {len(test_set)}")
    train_loader, valid_loader, test_loader = task.get_dataloader(
        train_set, valid_set, test_set, tokenizer, batch_size=2)
    for batch in train_loader:
        print(batch)
        break
