import torch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy import stats

import torch.nn.functional as F


def spearman(y_pred, y_true):
    if np.var(y_pred) < 1e-6 or np.var(y_true) < 1e-6:
        return 0.0
    return spearmanr(y_pred, y_true)[0]


def compute_stat(sr):
    sr = np.asarray(sr)
    mean = np.mean(sr)
    std = np.std(sr)
    return mean, std


def compute_score(model, seq, mask, wt, pos, tokenizer, A, spurs_ddg, aa_token_ids):
    '''
    compute mutational proxy using masked marginal probability
    :param seq:mutant seq
    :param mask:attention mask for input seq
    :param wt: wild type sequence
    :param pos:mutant position
    :return:
        score: mutational proxy score
        logits: output logits for masked sequence
    '''
    device = seq.device
    bs = seq.shape[0]
    
    # pos starts from 0, as mutated_position in data.csv
    pos_list = [p[0].item() if p.numel() == 1 else p[0].item() for p in pos]
    pos = torch.tensor(pos_list, dtype=torch.long, device=device)
    
    mask_seq = seq.clone()
    mask_seq[torch.arange(bs, device=device), pos + 1] = tokenizer.mask_token_id
    
    out = model(mask_seq, mask, output_hidden_states=True)
    logits = out.logits
    
    if A is not None and A.combined_way == "logits":
        seq_len = mask_seq.shape[1] - 2
        aligned_ddg = spurs_ddg.unsqueeze(0).expand(bs, -1, -1).to(device)
        aligned_logits = logits[:, 1:seq_len +1, aa_token_ids]
        
        if A.mode == "single":
            scaled_ddg = A.A * aligned_logits
        elif A.mode == "position-specific":
            a = A.A.unsqueeze(0).unsqueeze(2).expand(bs, -1, -1).to(device)
            scaled_ddg = a * aligned_ddg
        elif A.mode == "context-specific":
            flat_esm = aligned_logits.reshape(-1, 20)
            flat_ddg = aligned_ddg.reshape(-1, 20)
            a = A(flat_esm, flat_ddg)
            a = a.reshape(bs, seq_len, 1)
            scaled_ddg = a * aligned_ddg
            
        logits[:, 1:seq_len +1, aa_token_ids] += scaled_ddg
        
    log_probs = torch.log_softmax(logits, dim=-1)
    
    batch_idx = torch.arange(bs, device=device)
    p = pos + 1
    
    mut_token = seq[batch_idx, p]
    wt_token = wt[batch_idx, p]
    
    logp_mut = log_probs[batch_idx, p, mut_token]
    logp_wt = log_probs[batch_idx, p, wt_token]
    scores = logp_mut - logp_wt
    
    if A is not None and A.combined_way == "scores":
        aa_token_ids = aa_token_ids.to(device)        
        mut_idx = (aa_token_ids == mut_token.unsqueeze(1)).nonzero(as_tuple=True)[1]
        
        ddg_value = spurs_ddg[pos, mut_idx]
        
        if A.mode == "single":
            a = A.A.expand(bs).to(device)
        elif A.mode == "position-specific":
            a = A(mut_pos=pos).to(device)
        elif A.mode == "context-specific":
            esm_i = logits[batch_idx, p]
            esm_i = esm_i[:, aa_token_ids]
            ddg_i = spurs_ddg[pos]
            a = A(esm_i, ddg_i).squeeze(-1)
        else:
            a = torch.ones(bs, device=device)
            
        scores += a * ddg_value
            
    return scores, logits



def BT_loss(scores, golden_score):
    loss = torch.tensor(0.)
    loss = loss.cuda()
    for i in range(len(scores)):
        for j in range(i, len(scores)):
            if golden_score[i] > golden_score[j]:
                loss += torch.log(1+torch.exp(scores[j]-scores[i]))
            else:
                loss += torch.log(1+torch.exp(scores[i]-scores[j]))
    return loss


def KLloss(logits, logits_reg, seq, att_mask):

    creterion_reg = torch.nn.KLDivLoss(reduction='mean')
    batch_size = int(seq.shape[0])

    loss = torch.tensor(0.)
    loss = loss.cuda()
    probs = torch.softmax(logits, dim=-1)
    probs_reg = torch.softmax(logits_reg, dim=-1)
    for i in range(batch_size):

        probs_i = probs[i]
        probs_reg_i = probs_reg[i]


        seq_len = torch.sum(att_mask[i])

        reg = probs_reg_i[torch.arange(0, seq_len), seq[i, :seq_len]]
        pred = probs_i[torch.arange(0, seq_len), seq[i, :seq_len]]

        loss += creterion_reg(reg.log(), pred)
    return loss