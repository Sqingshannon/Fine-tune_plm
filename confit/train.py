import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model
from peft.utils.other import fsdp_auto_wrap_policy
from transformers import EsmForMaskedLM, EsmTokenizer, EsmConfig
import os
import argparse
from pathlib import Path
import accelerate
from accelerate import Accelerator

from data_utils import Mutation_Set, split_train, sample_data
from stat_utils import spearman, compute_score, BT_loss, KLloss
import gc
import warnings
import time
import yaml
warnings.filterwarnings("ignore")

from data_check import data_restruct
import random

class PsiFit(nn.Module):
    def __init__(self, esm_model, spurs_ddg, aa_token_ids):
        super(PsiFit, self).__init__()
        self.esm = esm_model
        device = next(self.esm.parameters()).device
        self.A = nn.Parameter(torch.tensor(0.1))
        
        self.spurs_ddg = spurs_ddg.to(device)
        self.aa_token_ids = aa_token_ids
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        seq_len = input_ids.shape[1] - 2
        aligned_ddg = self.spurs_ddg.unsqueeze(0).to(logits.device)
        scaled_ddg = self.A * aligned_ddg
        
        aligned_logits = logits[:, 1:seq_len+1, self.aa_token_ids]
        adjusted_logits = aligned_logits + scaled_ddg
        
        logits[:, 1:seq_len+1, self.aa_token_ids] = adjusted_logits
        outputs.logits = logits
        return outputs
    
class CoeffNN(nn.Module):
    def __init__(self):
        super(CoeffNN, self).__init__()
        self.lin1 = nn.Linear(20, 20)
        self.lin2 = nn.Linear(20, 20)
        self.lin3 = nn.Linear(20, 1)
        self.relu = nn.ReLU()
        
    def forward(self, esm_i, ddg_i):
        x_esm = self.relu(self.lin1(esm_i))
        x_ddg = self.relu(self.lin2(ddg_i))
        x = x_esm + x_ddg
        a_i = self.lin3(x)
        return a_i 
    
class AModule(nn.Module):
    def __init__(self, mode, spurs_ddg_shape, a_init, hidden_size=20, combined_way=None):
        super(AModule, self).__init__()
        self.mode = mode
        self.combined_way = combined_way
        # print("a init:", a_init)
        if self.mode == 'single':
            self.A = nn.Parameter(torch.tensor(a_init))
        elif self.mode == 'position-specific':
            self.A = nn.Parameter(torch.full((spurs_ddg_shape[0],), a_init))  
        elif self.mode == 'context-specific':
            self.lin1 = nn.Linear(20, hidden_size)
            self.lin2 = nn.Linear(hidden_size, hidden_size)
            self.lin3 = nn.Linear(hidden_size, 1)
            self.relu = nn.ReLU()
        elif self.mode == None:
            self.A = None
        else:
            raise ValueError(f'Invalid mode: {mode}')
        
    def forward(self, esm_i=None, ddg_i=None, mut_pos=None):
        if self.mode == "single":
            return self.A
        elif self.mode == "position-specific":
            if mut_pos is None:
                raise ValueError("mut_pos must be provided for position-specific mode")
            return self.A[mut_pos]
        elif self.mode == "context-specific":
            if esm_i is None or ddg_i is None:
                raise ValueError("esm_i and ddg_i must be provided for context-specific mode")
            x_esm = self.relu(self.lin1(esm_i))
            x_ddg = self.relu(self.lin2(ddg_i))
            x = x_esm + x_ddg
            a_i = self.lin3(x)
            return a_i
        elif self.mode == "none":
            return None
        else:
            raise ValueError(f'Invalid mode: {self.mode}')

def train(model, model_reg, trainloder, optimizer, tokenizer, lambda_reg, A, spurs_ddg, aa_token_ids, accelerator):

    model.train()

    total_loss = 0.

    for step, data in enumerate(trainloder):
        seq, mask = data[0], data[1]
        wt, wt_mask = data[2], data[3]
        pos = data[4]
        golden_score = data[5]
        mutation = data[7]
        score, logits = compute_score(model, seq, mask, wt, pos, tokenizer, A, spurs_ddg, aa_token_ids)
        score = score.cuda()

        l_BT = BT_loss(score, golden_score)

        out_reg = model_reg(wt, wt_mask)
        logits_reg = out_reg.logits
        l_reg = KLloss(logits, logits_reg, seq, mask)

        loss = l_BT + lambda_reg*l_reg

        optimizer.zero_grad()
        loss.backward()
        
        # accelerator.print(f'Step {step}: l_BT={l_BT.item():.4f}, l_reg={l_reg.item():.4f}, A.grad={A.grad.item() if A.grad is not None else "None"}')
        
        optimizer.step()
        total_loss += loss.item()
        
    # accelerator.print(f'End of epoch: A={A.item()}')
    return total_loss


def evaluate(model, testloader, tokenizer, accelerator, A, spurs_ddg, aa_token_ids, istest=False):
    model.eval()
    seq_list = []
    score_list = []
    gscore_list = []
    mutation_list = []
    with torch.no_grad():
        for step, data in enumerate(testloader):
            seq, mask = data[0], data[1]
            wt, wt_mask = data[2], data[3]
            pos = data[4]
            golden_score = data[5]
            pid = data[6]
            mutation = data[7]
            if istest:
                pid = pid.cuda()
                pid = accelerator.gather(pid)
                for s in pid:
                    seq_list.append(s.cpu())

            score, logits = compute_score(model, seq, mask, wt, pos, tokenizer, A, spurs_ddg, aa_token_ids)

            score = score.cuda()
            score = accelerator.gather(score)
            golden_score = accelerator.gather(golden_score)
            mutation = accelerator.gather(data[7])
            score = np.asarray(score.cpu())
            golden_score = np.asarray(golden_score.cpu())
            mutation = np.asarray(mutation.cpu())
            score_list.extend(score)
            gscore_list.extend(golden_score)
            mutation_list.extend(mutation)
        
            
            # print(data[7])
            # print("mutation_list", mutation_list)
            # if step == 5:
            #     exit(1)
    score_list = np.asarray(score_list)
    gscore_list = np.asarray(gscore_list)
    sr = spearman(score_list, gscore_list)

    if istest:
        seq_list = np.asarray(seq_list)

        return sr, score_list, gscore_list, seq_list, mutation_list
    else:
        return sr


def main():
    parser = argparse.ArgumentParser(description='ConFit train, set hyperparameters')
    parser.add_argument('--config', type=str, required=True,
                        help='the config file name')
    parser.add_argument('--dataset', type=str, help='the dataset name')
    parser.add_argument('--sample_seed', type=int, default=0, help='the sample seed for dataset')
    parser.add_argument('--model_seed', type=int, default=1, help='the random seed for the pretrained model initiate')
    parser.add_argument('--a_type', type=str, help='the type of A, none, single or position-specific or context-specific')
    parser.add_argument('--a_init', type=float, help='the initial value of A, only used when A_type is single or position-specific')
    parser.add_argument('--combined_way', type=str, help='the way to combine esm and ddg for context-specific A, score or feature')
    parser.add_argument('--train_mode', type=str, help='full = joint ConFit training (default), a_only = train ONLY A module')
    
    args, _ = parser.parse_known_args()
    dataset = args.dataset
    a_type = args.a_type
    a_init = args.a_init
    combined_way = args.combined_way
    train_mode = args.train_mode
    
    np.random.seed(args.sample_seed)
    random.seed(args.sample_seed)
    torch.manual_seed(args.model_seed)
    torch.cuda.manual_seed_all(args.model_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    batch_size = int(int(config['batch_size'])/int(config['gpu_number']))
    
    accelerator = Accelerator()
    # accelerator.set_seed(args.model_seed)
    
    if accelerator.is_main_process:
        data_restruct(dms_id=dataset, seed=args.model_seed, a_type=a_type, a_init=a_init, combined_way=combined_way, train_mode=train_mode)
    accelerator.wait_for_everyone()

    #read in config
    with open(f'{args.config}', 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

    ### creat model
    if config['model'] == 'ESM-1v':
        basemodel = EsmForMaskedLM.from_pretrained(f'facebook/esm1v_t33_650M_UR90S_{args.model_seed}')
        model_reg = EsmForMaskedLM.from_pretrained(f'facebook/esm1v_t33_650M_UR90S_{args.model_seed}')
        tokenizer = EsmTokenizer.from_pretrained(f'facebook/esm1v_t33_650M_UR90S_{args.model_seed}')

    elif config['model'] == 'ESM-2':
        basemodel = EsmForMaskedLM.from_pretrained('facebook/esm2_t48_15B_UR50D')
        model_reg = EsmForMaskedLM.from_pretrained('facebook/esm2_t48_15B_UR50D')
        tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t48_15B_UR50D')

    elif config['model'] == 'ESM-1b':
        basemodel = EsmForMaskedLM.from_pretrained('facebook/esm1b_t33_650M_UR50S')
        model_reg = EsmForMaskedLM.from_pretrained('facebook/esm1b_t33_650M_UR50S')
        tokenizer = EsmTokenizer.from_pretrained('facebook/esm1b_t33_650M_UR50S')

    aa_tokens = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    aa_token_ids = tokenizer.convert_tokens_to_ids(aa_tokens)
    aa_token_ids = torch.tensor(aa_token_ids)
    spurs_ddg = pd.read_csv(f'data/{dataset}/spurs_prediction.tsv', sep='\t', index_col=0)
    spurs_ddg = torch.tensor(spurs_ddg.values, dtype=torch.float32).to(accelerator.device)

    for pm in model_reg.parameters():
        pm.requires_grad = False
    model_reg.eval()    #regularization model
    
    A = AModule(mode=a_type, spurs_ddg_shape=spurs_ddg.shape, a_init=a_init, combined_way=combined_way).to(accelerator.device)


    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=int(config['lora_r']),
        lora_alpha=int(config['lora_alpha']),
        lora_dropout=float(config['lora_dropout']),
        target_modules=["query", "value"]
    )

    model = get_peft_model(basemodel, peft_config)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(A.parameters()),
        lr=float(config['ini_lr'])
    )
      
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2*int(config['max_epochs']), eta_min=float(config['min_lr']))
    if os.environ.get("ACCELERATE_USE_FSDP", None) is not None:
        accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(model)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    model_reg = accelerator.prepare(model_reg)

    accelerator.print(f'===================dataset:{dataset}, preparing data=============')

    # sample data
    if accelerator.is_main_process:
        sample_data(dataset, args.sample_seed, int(config['shot']))
        split_train(dataset)

    with accelerator.main_process_first():
        train_csv = pd.DataFrame(None)
        test_csv = pd.read_csv(f'data/{dataset}/test.csv')
        val_csv = None
        for i in range(1, 6):
            temp_csv = pd.read_csv(f'data/{dataset}/train_{i}.csv')
            if i == args.model_seed:
                val_csv = temp_csv
            else:
                train_csv = pd.concat([train_csv, temp_csv], axis=0)

    #creat dataset and dataloader
    trainset = Mutation_Set(data=train_csv, fname=dataset, tokenizer=tokenizer)
    testset = Mutation_Set(data=test_csv, fname=dataset,  tokenizer=tokenizer)
    valset = Mutation_Set(data=val_csv, fname=dataset,  tokenizer=tokenizer)
    with accelerator.main_process_first():
        trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=trainset.collate_fn, shuffle=True)
        testloader = DataLoader(testset, batch_size=2, collate_fn=testset.collate_fn, shuffle=False) #shuffle=False
        valloader = DataLoader(valset, batch_size=2, collate_fn=testset.collate_fn)

    trainloader = accelerator.prepare(trainloader)
    testloader = accelerator.prepare(testloader)
    valloader = accelerator.prepare(valloader)
    accelerator.print('==============data preparing done!================')
    # accelerator.print("Current allocated memory:", torch.cuda.memory_allocated())
    # accelerator.print("cached:", torch.cuda.memory_reserved())
    save_dir = Path('checkpoint', f'{dataset}',
                                     f'seed{args.model_seed}',
                                     f'mode{a_type}_ainit{a_init}_combined{combined_way}_trainmode{args.train_mode}')
    
    if args.train_mode == "full":
        accelerator.print("========start full LoRA + A training!============")
        
    
    best_sr = -np.inf
    endure = 0
    best_epoch = 0

    for epoch in range(int(config['max_epochs'])):
        loss = train(model, model_reg, trainloader, optimizer, tokenizer, float(config['lambda_reg']), A, spurs_ddg, aa_token_ids, accelerator)
        accelerator.print(f'========epoch{epoch}; training loss :{loss}=================')
        sr = evaluate(model, valloader, tokenizer, accelerator, A=A, spurs_ddg=spurs_ddg, aa_token_ids=aa_token_ids)
        accelerator.print(f'========epoch{epoch}; val spearman correlation :{sr}=================')
        scheduler.step()
        if best_sr > sr:
            endure += 1
        else:
            endure = 0
            best_sr = sr
            best_epoch = epoch

            if not os.path.isdir(f'checkpoint/{dataset}'):
                if accelerator.is_main_process:
                    os.makedirs(f'checkpoint/{dataset}')
            save_path = os.path.join('checkpoint', f'{dataset}',
                                     f'seed{args.model_seed}',
                                     f'mode{a_type}_ainit{a_init}_combined{combined_way}_trainmode{args.train_mode}')
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(save_path)
            
            accelerator.save(A.state_dict(), os.path.join(save_path, 'A.pth'))
            # if a_type == 'single' or a_type == 'position-specific':
            #     accelerator.save(A, os.path.join(save_path, 'A.pth'))
            # elif a_type == 'context-specific':
            #     accelerator.save(A.state_dict(), os.path.join(save_path, 'A.pth'))
            
        if sr == 1.0:
            accelerator.print(f'========early stop at epoch{epoch}!============')
            break
        if endure > int(config['endure_time']):
            accelerator.print(f'========early stop at epoch{epoch}!============')
            break

    # inference on the test sest
    accelerator.print('=======training done!, test the performance!========')
    save_path = Path(os.path.join('checkpoint', f'{dataset}', f'seed{args.model_seed}'), f'mode{a_type}_ainit{a_init}_combined{combined_way}_trainmode{args.train_mode}')
    del basemodel
    del model
    accelerator.free_memory()

    if config['model'] == 'ESM-1v':
        basemodel = EsmForMaskedLM.from_pretrained(f'facebook/esm1v_t33_650M_UR90S_{args.model_seed}')
        tokenizer = EsmTokenizer.from_pretrained(f'facebook/esm1v_t33_650M_UR90S_{args.model_seed}')

    if config['model'] == 'ESM-2':
        basemodel = EsmForMaskedLM.from_pretrained('facebook/esm2_t48_15B_UR50D')
        tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t48_15B_UR50D')

    if config['model'] == 'ESM-1b':
        basemodel = EsmForMaskedLM.from_pretrained('facebook/esm1b_t33_650M_UR50S')
        tokenizer = EsmTokenizer.from_pretrained('facebook/esm1b_t33_650M_UR50S')

    aa_token_ids = tokenizer.convert_tokens_to_ids(aa_tokens)
    aa_token_ids = torch.tensor(aa_token_ids)
    
    
    A = AModule(mode=a_type, spurs_ddg_shape=spurs_ddg.shape, a_init=a_init, combined_way=combined_way).to(accelerator.device)
    A.load_state_dict(torch.load(os.path.join(save_path, 'A.pth'), map_location=accelerator.device))
    A.requires_grad_(False)
    # if a_type == 'single':
    #     A.data = torch.load(os.path.join(save_path, 'A.pth'), map_location=accelerator.device)
    #     A.requires_grad_(False)
    # elif a_type == 'position-specific':
    #     A = CoeffNN().to(accelerator.device)
    #     A.load_state_dict(torch.load(os.path.join(save_path, 'A.pth'), map_location=accelerator.device))
    #     A.requires_grad_(False)
    
    model = PeftModel.from_pretrained(basemodel, save_path)
    model = accelerator.prepare(model)
    sr, score, gscore, pid, mutation_list = evaluate(model, testloader, tokenizer, accelerator, istest=True, A=A, spurs_ddg=spurs_ddg, aa_token_ids=aa_token_ids)
    # print("mutation_list:", mutation_list)
    # print("len of pid:", len(pid), "len of mutation_list:", len(mutation_list), "len of score:", len(score), "len of gscore:", len(gscore))
    pred_csv = pd.DataFrame({f'{args.model_seed}': score, 'mutation': mutation_list, "y_true": gscore})
    pred_save_path = Path(f'predicted/{dataset}/seed{args.model_seed}_mode{a_type}_ainit{a_init}_combined{combined_way}_trainmode{args.train_mode}')
    # pred_save_path = f'predicted/{dataset}'
    if accelerator.is_main_process:
        if not os.path.isdir(pred_save_path):
            os.makedirs(pred_save_path)
        if os.path.exists(pred_save_path / 'pred.csv'):
            pred = pd.read_csv(pred_save_path / 'pred.csv', index_col=0)
            pred = pd.merge(pred, pred_csv, on='PID')
        else:
            pred = pred_csv
        pred.to_csv(pred_save_path / 'pred.csv')
    accelerator.print(f'=============the test spearman correlation for early stop: {sr}==================')


if __name__ == "__main__":
    main()





