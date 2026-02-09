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

    mask_seq = seq.clone()
    m_id = tokenizer.mask_token_id

    batch_size = int(seq.shape[0])
    for i in range(batch_size):
        mut_pos = pos[i]
        mask_seq[i, mut_pos+1] = m_id

    out = model(mask_seq, mask, output_hidden_states=True)
    logits = out.logits
    log_probs = torch.log_softmax(logits, dim=-1)
    scores = torch.zeros(batch_size)
    scores = scores.to(device)
    
    token_to_aa_idx = {aa_token_ids[j].item(): j for j in range(len(aa_token_ids))}
    
    if A.combined_way == "logits":
        # print("We do combine by logits")
        seq_len = mask_seq.shape[1] - 2
        aligned_ddg = spurs_ddg.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        aligned_logits = logits[:, 1:seq_len + 1, aa_token_ids]
        
        if A.mode == 'single':
            scaled_ddg = A.A * aligned_ddg
        elif A.mode == 'position-specific':
            scaled_ddg = A.A.unsqueeze(0).unsqueeze(-1) * aligned_ddg
        elif A.mode == 'context-specific':
            flat_esm = aligned_ddg.reshape(-1, 20)
            flat_ddg = aligned_ddg.reshape(-1, 20)
            a = A(flat_esm, flat_ddg)
            a = a.reshape(batch_size, seq_len, 1)
            scaled_ddg = a * aligned_ddg
        
        adjusted_logits = aligned_logits + scaled_ddg
        logits[:, 1:seq_len + 1, aa_token_ids] = adjusted_logits
    elif A.combined_way == "scores":
        # print("We do combine by scores")
        logits = logits
    else:
        raise ValueError(f"Invalid combined_way 1st: {A.combined_way}")
    

    for i in range(batch_size):

        mut_pos = pos[i]
        score_i = log_probs[i]
        wt_i = wt[i]
        seq_i = seq[i]
        scores[i] = torch.sum(score_i[mut_pos+1, seq_i[mut_pos+1]])-torch.sum(score_i[mut_pos+1, wt_i[mut_pos+1]])
        
        if A.combined_way == "scores":
            # print("second check we do combine by scores")
            if A.mode == 'context-specific':
                esm_i = score_i[mut_pos+1, aa_token_ids]
                ddg_i = spurs_ddg[mut_pos, :]
                a_i = A(esm_i, ddg_i)
            
            mut_token = seq_i[mut_pos+1].item()
            aa_idx = token_to_aa_idx.get(mut_token)
            if aa_idx is not None:
                spurs_score_i = spurs_ddg[mut_pos, aa_idx].item()
                
                if A.mode == 'single':
                    scores[i] += A.A * spurs_score_i
                elif A.mode == 'position-specific':
                    scores[i] += A.A[mut_pos] * spurs_score_i
                elif A.mode == 'context-specific':
                    scores[i] += (a_i * spurs_score_i).item()
        elif A.combined_way == "logits":
            # print("second check we do combine by logits")
            pass
        else:
            raise ValueError(f"Invalid combined_way 2nd: {A.combined_way}")
            
            
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