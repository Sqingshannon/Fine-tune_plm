import copy
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
from scipy.stats import spearmanr
from pathlib import Path

import sys
sys.path.append('../data')
sys.path.append('../model')
sys.path.append('../utils')
from constants import AA20
from dataset import ProteinGymTask, DMSDataset
from psifit import PsiFitModel
from loss import BTLoss, KLLoss, BTLoss_naive, KLLoss_naive
from earlystopper import EarlyStopper
from logger import Logger

def forward_pass(model, batch, device):
    seq, mask = batch["seq"].to(device), batch["attn_mask"].to(device)
    wq_seq = batch["wt"].to(device)
    pos, label = batch["mutated_pos"].to(device), batch["label"].to(device)
    scores, logits = model(seq, mask, wq_seq, pos)
    return scores, logits, label

def train(model, ref_logits, train_loader, optimizer, device, lambda_reg=1):
    model.train()
    total_loss_BT, total_loss_reg, total_loss = 0, 0, 0
    for batch in train_loader:
        scores, logits, label = forward_pass(model, batch, device)
        l_BT = BTLoss(scores, label)
        logits_ref = ref_logits.expand(logits.size(0), -1, -1)  # broadcast cached wt logits to batch
        l_reg = KLLoss(logits, logits_ref)
        loss = l_BT + lambda_reg * l_reg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss_BT += l_BT.item()
        total_loss_reg += l_reg.item()
        total_loss += loss.item()
    avg_loss_BT = total_loss_BT / len(train_loader)
    avg_loss_reg = total_loss_reg / len(train_loader)
    avg_loss = total_loss / len(train_loader)
    return avg_loss_BT, avg_loss_reg, avg_loss

def evaluate(model, test_loader, device):
    model.eval()
    y_pred, y_true = None, None
    with torch.inference_mode():
        for batch in tqdm(test_loader, leave=False):
            scores, logits, label = forward_pass(model, batch, device)
            y_pred = scores.detach().cpu() if y_pred is None else torch.cat((y_pred, scores.detach().cpu()), dim=0)
            y_true = label.detach().cpu() if y_true is None else torch.cat((y_true, label.detach().cpu()), dim=0)
    corr = spearmanr(y_true.numpy(), y_pred.numpy())[0]
    return corr


def main():
    parser = argparse.ArgumentParser(description="Train PsiFit model on a ProteinGym task")
    parser.add_argument('-k', '--shot', type=int, default=96, help='k-shot training split')
    parser.add_argument('--dataset', type=str, default='IF1_ECOLI_Kelsic_2016', help='dms id / dataset')
    parser.add_argument('--epochs', type=int, default=30, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--lambda_reg', type=int, default=10, help='lambda of KL regularization term')
    parser.add_argument('--exp_root', type=str, default='../../output/experiment/proteingym')
    parser.add_argument('--exp_tag', type=str, help='experiment tag')
    parser.add_argument('--log', action='store_true', default=False, help='save log file')
    args = parser.parse_args()

    logger = Logger(logfile=Path(f"{args.exp_root}/{args.exp_tag}/{args.dataset}/exp.log") if args.log else None)

    esm_model_name="facebook/esm2_t33_650M_UR50D"
    esm_model = AutoModelForMaskedLM.from_pretrained(esm_model_name)
    esm_tokenizer = AutoTokenizer.from_pretrained(esm_model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    task = ProteinGymTask(dms_id=args.dataset)
    train_df, valid_df, test_df = task.split(k=args.shot, test_frac=0.2, valid_frac=0.2, seed=42)
    train_loader, valid_loader, test_loader = task.get_dataloader(
        train_df, valid_df, test_df, esm_tokenizer, batch_size=args.batch_size)
    
    # Precompute reference logits for the wild-type sequence once, then free the ref model
    ref_model = AutoModelForMaskedLM.from_pretrained(esm_model_name)
    for pm in ref_model.parameters():
        pm.requires_grad = False
    ref_model.eval()

    wt_tokens = esm_tokenizer([task.wt_seq], return_tensors="pt", padding=False, truncation=False)
    wt_input_ids = wt_tokens['input_ids'].to(device)
    wt_attn_mask = wt_tokens['attention_mask'].to(device)
    ref_model.to(device)
    with torch.inference_mode():
        ref_out = ref_model(input_ids=wt_input_ids, attention_mask=wt_attn_mask)
    ref_logits = ref_out.logits.detach().to(device)
    del ref_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = PsiFitModel(esm_model, esm_tokenizer)
    optimizer = torch.optim.Adam(model.parameters(),)
    
    model.to(device)
    stopper = EarlyStopper(patience=args.patience, eval_freq=1, higher_better=True)
    best_model_state_dict = None
    for epoch in range(args.epochs):
        avg_loss_BT, avg_loss_reg, avg_loss = train(
            model, ref_logits, train_loader, optimizer, device, lambda_reg=args.lambda_reg)
        val_corr = evaluate(model, valid_loader, device)
        is_best = stopper.update(val_corr)
        if is_best:
            best_model_state_dict = copy.deepcopy(model.state_dict())
        logger.info(f"Epoch {epoch+1} | BT loss: {avg_loss_BT:.6f} | Reg loss: {avg_loss_reg:.6f} | Total loss: {avg_loss:.6f}"\
        + f" | val corr: {val_corr:.6f}")
        if stopper.early_stop:
            logger.info(f'Eearly stop at epoch {epoch}')
            break
    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
        del best_model_state_dict
        torch.cuda.empty_cache()
    test_corr = evaluate(model, test_loader, device)
    logger.info(f"{args.dataset} | Test Spearman correlation: {test_corr:.6f}")

if __name__ == "__main__":
    main()