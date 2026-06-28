import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForMaskedLM

import sys
sys.path.append('../data')
from constants import AA20


class PsiFitModel(nn.Module):
    def __init__(self, esm, tokenizer):
        super(PsiFitModel, self).__init__()
        self.esm = esm
        self.tokenizer = tokenizer
        # vocab = self.tokenizer.get_vocab()
        # self.aa_idx = torch.tensor([vocab[aa] for aa in AA20])  # indices of 20 AAs in ESM vocab
        # self.esmidx2aaidx = {vocab[aa]: i for i, aa in enumerate(AA20)}
        # self.aaidx2esmidx = {i: vocab[aa] for i, aa in enumerate(AA20)}

    
    def forward(self, seq, mask, wt, pos):
        """
        Compute the delta log-likelihood (DLL) score for a batch of sequences
        at the specified mutated positions, vectorized.

        Args:
            seq: (batch_size, seq_len) LongTensor of mutated sequences (token ids)
            mask: (batch_size, seq_len) attention mask for the sequences
            wt: (batch_size, seq_len) LongTensor of wild-type sequences (token ids)
            pos: (batch_size,) LongTensor of mutated positions (0-based index), assuming single mutation per sequence
            mask_token_id: optional int token id to use for masking; if None, uses `self.tokenizer.mask_token_id`

        Returns:
            scores: (batch_size,) tensor of DLL scores (seq_logprob - wt_logprob)
            logits: model logits returned by the ESM model (batch_size, seq_len, vocab_size)
        """
        device = seq.device
        b_size = seq.size(0)

        # clone and set the mask token at the mutated positions
        # positions (0-based index) are shifted by +1 because of CLS token at the start
        mask_seq = seq.clone()
        batch_idx = torch.arange(b_size, device=device)
        positions = pos.to(device).long() + 1

        # bounds check
        if (positions < 0).any() or (positions >= seq.size(1)).any():
            raise IndexError("Some mutation positions are out of sequence bounds")
        mask_val = torch.tensor(self.tokenizer.mask_token_id, device=device, dtype=mask_seq.dtype)
        mask_seq[batch_idx, positions] = mask_val

        # forward through the esm masked-language-model
        outputs = self.esm(input_ids=mask_seq, attention_mask=mask)
        logits = outputs.logits  # (batch, seq_len, vocab_size)
        log_probs = torch.log_softmax(logits, dim=-1)

        # gather token-level log-probs at the mutated positions for seq and wt
        seq_token_ids = seq[batch_idx, positions]    # (batch,)
        wt_token_ids  = wt[batch_idx, positions]     # (batch,)
        seq_logprobs = log_probs[batch_idx, positions, seq_token_ids]  # (batch,)
        wt_logprobs  = log_probs[batch_idx, positions, wt_token_ids]   # (batch,)
        scores = seq_logprobs - wt_logprobs

        return scores, logits

if __name__ == "__main__":
    from dataset import ProteinGymTask, DMSDataset
    from scipy.stats import spearmanr
    from tqdm import tqdm
    from torch.utils.data import DataLoader

    esm_model_name="facebook/esm2_t33_650M_UR50D"
    esm_model = AutoModelForMaskedLM.from_pretrained(esm_model_name)
    esm_tokenizer = AutoTokenizer.from_pretrained(esm_model_name)

    task = ProteinGymTask(dms_id='SUMO1_HUMAN_Weile_2017')
    test_set = DMSDataset(task.dms, esm_tokenizer, task.wt_seq)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PsiFitModel(esm_model, esm_tokenizer).to(device)
    model.eval()
    y_pred, y_true = None, None
    for batch in tqdm(test_loader):
        seq, mask = batch["seq"].to(device), batch["attn_mask"].to(device)
        wq_seq, wt_mask = batch["wt"].to(device), batch["wt_attn_mask"].to(device)
        pos, label = batch["mutated_pos"].to(device), batch["label"].to(device)
        scores, logits = model(seq, mask, wq_seq, pos)
        y_pred = scores.detach().cpu() if y_pred is None else torch.cat((y_pred, scores.detach().cpu()), dim=0)
        y_true = label.detach().cpu() if y_true is None else torch.cat((y_true, label.detach().cpu()), dim=0)
    print(spearmanr(y_true.numpy(), y_pred.numpy())[0])   
