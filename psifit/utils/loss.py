import torch
import torch.nn.functional as F

def BTLoss_naive(scores, golden_score):
    loss = torch.tensor(0.)
    loss = loss.cuda()
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            if golden_score[i] > golden_score[j]:
                loss += torch.log(1 + torch.exp(scores[j] - scores[i]))
            else:
                loss += torch.log(1 + torch.exp(scores[i] - scores[j]))
    return loss
import torch


def BTLoss(y_pred, y_true):
    """
    Vectorized Bradley-Terry pairwise loss. 
    Running time almost linearly increases with n, while the naive implementation increases quadratically.
    ~100x, 600x, 1500x, 1700x speedup compared to naive implementation for n=100, 1000, 2000, 3000 respectively.

    Args:
        y_pred:  (n,) tensor of predicted scores (float, on correct device)
        y_true: (n,) tensor of ground-truth scores/ranking values

    Returns:
        scalar tensor: sum of pairwise logistic losses over i < j:
            loss_ij = log(1 + exp(y_pred_j - y_pred_i)) if y_true_i > y_true_j
                    = log(1 + exp(y_pred_i - y_pred_j)) otherwise
    """
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    n = y_pred.size(0)
    device = y_pred.device
    dtype = y_pred.dtype

    if n <= 1:
        return torch.tensor(0., device=device, dtype=dtype)

    # pairwise differences
    p_diff = y_pred.unsqueeze(1) - y_pred.unsqueeze(0)  # shape (n, n), y_pred_i - y_pred_j
    t_diff = y_true.unsqueeze(1) - y_true.unsqueeze(0)  # shape (n, n)

    # label: +1 when y_true_i > y_true_j else -1 (treat equal as else branch)
    labels = torch.where(t_diff > 0,
                         torch.tensor(1., device=device, dtype=dtype),
                         torch.tensor(-1., device=device, dtype=dtype))

    # compute pairwise argument to softplus:
    # loss_ij = softplus(- labels * (y_pred_i - y_pred_j))
    pairwise = - labels * p_diff
    # only sum each unordered pair once (i < j)
    mask = torch.triu(torch.ones((n, n), dtype=torch.bool, device=device), diagonal=1)
    losses = F.softplus(pairwise)    # numerically stable log(1+exp(x))
    loss = losses[mask].mean()
    return loss

def KLLoss_naive(logits, logits_reg):
    creterion_reg = torch.nn.KLDivLoss(reduction='mean')
    batch_size = logits.size(0)

    loss = torch.tensor(0.)
    loss = loss.cuda()
    probs = torch.softmax(logits, dim=-1)
    probs_reg = torch.softmax(logits_reg, dim=-1)
    for i in range(batch_size):
        loss += creterion_reg(probs_reg[i].log(), probs[i])
    return loss

def KLLoss(logits, logits_ref, reduction='mean'):
    """
    KL divergence regularization between model logits and reference logits.

    Computes KL(P || P_ref) per sequence position (vocabulary distribution),
    averages the per-position KL over sequence positions for each sequence,
    then aggregates over the batch.

    Formally:
        kl^j_i = KL( P_{j,i} || Q_{j,i} )                            # per-position KL
        kl^j   = mean_i( kl^j_i )                                    # average over positions
        output = sum_j kl^j          (if reduction == 'sum')
               = mean_j kl^j         (if reduction == 'mean')

    Notes:
    - `logits` and `logits_ref` are un-normalized scores (shapes described below).
    - We compute log-probabilities with `log_softmax` and use `F.kl_div` with
      `log_target=True` for numerical stability (avoids extra exp/log).
    - The function is fully vectorized (no Python loops).

    Args:
        logits (Tensor): shape (batch_size, seq_len, vocab_size).  Model logits.
        logits_ref (Tensor): same shape as `logits`.  Reference model logits.
        reduction (str): 'sum' (default) to sum per-sequence KLs over batch,
                         'mean' to average per-sequence KLs over batch.

    Returns:
        Tensor: scalar tensor (KL aggregated according to `reduction`).
    """
    # Compute log-probabilities for model and reference (stable and vectorized).
    # Shapes: (B, L, V)
    log_p = torch.log_softmax(logits, dim=-1)      # log P (model)
    log_q = torch.log_softmax(logits_ref, dim=-1)  # log Q (reference)

    # Use PyTorch's kl_div in a numerically stable way:
    # F.kl_div(input=log_q, target=log_p, log_target=True) computes
    # per-elem target * (log_target - input) = p * (log p - log q).
    per_elem = F.kl_div(log_q, log_p, reduction='none', log_target=True)  # (B, L, V)

    # Sum over vocabulary to get KL per position: (B, L)
    kl_per_pos = per_elem.sum(dim=-1)
    # Average over sequence positions to get per-sequence KL: (B,)
    kl_per_seq = kl_per_pos.mean(dim=1)

    if reduction == 'sum':
        return kl_per_seq.sum()
    elif reduction == 'mean':
        return kl_per_seq.mean()
    else:
        raise ValueError("reduction must be 'sum' or 'mean'")
    

if __name__ == "__main__":
    """
    Test: BTLoss
    Randomize y_true and y_pred for testing and benchmark the running time of the vectorized loss vs naive loss.
    n=100 : vectorized: 0.0062s vs naive:  0.930s  (100x speedup)
    n=1000: vectorized: 0.0196s vs naive: 12.637s  (600x speedup)
    n=2000: vectorized: 0.0378s vs naive: 47.745s  (1500x speedup)
    n=3000: vectorized: 0.0643s vs naive: 104.626s (1700x speedup)
    """
    import torch
    import time
    n = 100
    y_true = torch.rand(n)
    y_pred = torch.rand(n)
    start = time.time()
    loss_vec = BTLoss(y_pred, y_true)
    end = time.time()
    print(f"Vectorized BT loss: {loss_vec.item():.4f}, time:{end - start:.4f} seconds")
    start = time.time()
    loss_naive = BTLoss_naive(y_pred, y_true)
    end = time.time()
    print(f"Naive BT loss: {loss_naive.item():.4f}, time:{end - start:.4f} seconds")
    assert torch.isclose(loss_vec, loss_naive), "Vectorized and naive losses do not match!"

    """
    Randomized logits and logits_ref for testing KLLoss and KLLoss_naive
    """
    batch_size = 1
    seq_len = 3
    vocab_size = 6
    logits = torch.randn(batch_size, seq_len, vocab_size)
    logits_ref = torch.randn(batch_size, seq_len, vocab_size)
    loss_vec = KLLoss(logits, logits_ref, reduction='sum')
    print(f"KLLoss: {loss_vec.item():.4f}")
    loss_naive = KLLoss_naive(logits, logits_ref)
    print(f"KLLoss naive: {loss_naive.item():.4f}")