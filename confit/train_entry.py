"""Training entry point — thin CLI wrapper around ConFitTrainer.

This script is what ``accelerate launch`` calls.  All business logic lives in
the ``confit`` package; this file only handles argument parsing, wiring up the
components, and calling ``trainer.fit()``.
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from peft import PeftModel

warnings.filterwarnings("ignore")

from confit.config.loader import ConfigLoader
from confit.data.dataset import MutationDataset
from confit.data.preprocessing import DataPreprocessor
from confit.data.splitter import DataSplitter
from confit.models.factory import ESMModelFactory
from confit.models.registry import ModelRegistry, ModelVariant
from confit.models.scaling import AModule
from confit.scoring.masked_marginal import MaskedMarginalScorer
from confit.training.evaluator import ConFitEvaluator
from confit.training.trainer import ConFitTrainer, TrainMode
from confit.utils.seeding import seed_everything
from torch.utils.data import DataLoader


_AA_TOKENS = [
    "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
    "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
]


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the training entry point."""
    p = argparse.ArgumentParser(description="ConFit train")
    p.add_argument("--config",       type=str, required=True)
    p.add_argument("--dataset",      type=str)
    p.add_argument("--sample_seed",  type=int, default=0)
    p.add_argument("--model_seed",   type=int, default=1)
    p.add_argument("--a_type",       type=str)
    p.add_argument("--a_init",       type=float)
    p.add_argument("--combined_way", type=str)
    p.add_argument("--train_mode",   type=str)
    p.add_argument("--shot",         type=int, default=None)
    p.add_argument("--run_suffix",   type=str, default="rerun_fixed")
    return p.parse_args()


def main() -> None:
    """Main training entry point."""
    args = _parse_args()

    os.environ["ACCELERATE_USE_FSDP"] = "0"
    seed_everything(sample_seed=args.sample_seed, model_seed=args.model_seed)

    cfg = ConfigLoader.load(args.config)
    shot = args.shot if args.shot is not None else cfg.shot
    accelerator = Accelerator()

    # --- Data loading from the pre-built data/ directory ---
    # data_root = Path(f"data_{args.run_suffix}")  # old: generate/copy splits per run
    # if accelerator.is_main_process:
    #     preprocessor = DataPreprocessor(output_base=data_root)
    #     preprocessor.prepare(args.dataset)
    # accelerator.wait_for_everyone()
    # splitter = DataSplitter(data_root=data_root)
    # if accelerator.is_main_process:
    #     splitter.ensure_splits_exist(args.dataset, seed=args.sample_seed, shot=shot)
    # accelerator.wait_for_everyone()
    data_root = Path("data")  # use the pre-built data/ directory directly

    # --- Build model bundle ---
    variant = ModelRegistry.from_string(cfg.model)
    factory = ESMModelFactory()
    bundle = factory.build(variant, model_seed=args.model_seed)
    tokenizer = bundle.tokenizer

    aa_token_ids = torch.tensor(tokenizer.convert_tokens_to_ids(_AA_TOKENS))

    # Load SPURS DDG from data/ (same directory as all other data)
    spurs_ddg = pd.read_csv(
        f"data/{args.dataset}/spurs_prediction.tsv", sep="\t", index_col=0
    )
    spurs_ddg_tensor = torch.tensor(
        spurs_ddg.values, dtype=torch.float32
    ).to(accelerator.device)

    # Prepare frozen reference model
    model_reg = accelerator.prepare(bundle.reg_model)

    # Build A module and PEFT model
    A = AModule(
        mode=args.a_type,
        spurs_ddg_shape=spurs_ddg_tensor.shape,
        a_init=args.a_init,
        combined_way=args.combined_way,
    ).to(accelerator.device)

    model = ConFitTrainer.build_peft_model(bundle.backbone, cfg, accelerator)

    # --- Build data loaders ---
    accelerator.print(f"=======dataset:{args.dataset}, preparing data==========")
    per_device_bs = cfg.per_device_batch_size

    with accelerator.main_process_first():
        test_csv = pd.read_csv(data_root / args.dataset / "test.csv")
        train_csv = pd.DataFrame()
        val_csv: Optional[pd.DataFrame] = None
        for i in range(1, 6):
            temp = pd.read_csv(data_root / args.dataset / f"train_{i}.csv")
            if i == args.model_seed:
                val_csv = temp
            else:
                train_csv = pd.concat([train_csv, temp], axis=0)

    dataset_kwargs = dict(fname=args.dataset, tokenizer=tokenizer,
                          data_root=str(data_root))
    trainset = MutationDataset(data=train_csv, **dataset_kwargs)
    testset  = MutationDataset(data=test_csv,  **dataset_kwargs)
    valset   = MutationDataset(data=val_csv,   **dataset_kwargs)

    with accelerator.main_process_first():
        trainloader = DataLoader(
            trainset, batch_size=per_device_bs,
            collate_fn=trainset.collate_fn, shuffle=True,
        )
        testloader = DataLoader(
            testset, batch_size=2,
            collate_fn=testset.collate_fn, shuffle=False,
        )
        valloader = DataLoader(
            valset, batch_size=2,
            collate_fn=valset.collate_fn,
        )

    trainloader = accelerator.prepare(trainloader)
    testloader  = accelerator.prepare(testloader)
    valloader   = accelerator.prepare(valloader)
    accelerator.print("==============data preparing done!================")

    # --- Build scorer / evaluator / trainer ---
    scorer = MaskedMarginalScorer(
        tokenizer=tokenizer,
        a_module=A,
        spurs_ddg=spurs_ddg_tensor,
        aa_token_ids=aa_token_ids,
    )
    evaluator = ConFitEvaluator(scorer=scorer, accelerator=accelerator, tokenizer=tokenizer)

    save_dir = Path(
        f"checkpoint_{args.run_suffix}",
        args.dataset,
        f"shot{shot}",
        f"seed{args.model_seed}",
        f"mode{args.a_type}_ainit{args.a_init}_combined{args.combined_way}"
        f"_trainmode{args.train_mode}",
    )

    trainer = ConFitTrainer(
        config=cfg,
        model=model,
        model_reg=model_reg,
        a_module=A,
        scorer=scorer,
        evaluator=evaluator,
        accelerator=accelerator,
        train_mode=TrainMode(args.train_mode),
        tokenizer=tokenizer,
        basemodel=bundle.backbone,
    )

    trainer.fit(trainloader, valloader, save_dir)
    accelerator.print("=======training done!, testing performance!========")

    # --- Reload best checkpoint for test evaluation ---
    gc_module = __import__("gc")
    del model
    accelerator.free_memory()
    gc_module.collect()

    bundle2 = factory.build(variant, model_seed=args.model_seed)
    aa_token_ids2 = torch.tensor(bundle2.tokenizer.convert_tokens_to_ids(_AA_TOKENS))
    A2 = AModule(
        mode=args.a_type,
        spurs_ddg_shape=spurs_ddg_tensor.shape,
        a_init=args.a_init,
        combined_way=args.combined_way,
    ).to(accelerator.device)
    if A2.mode.value != "none":
        A2.load_state_dict(
            torch.load(save_dir / "A.pth", map_location=accelerator.device)
        )
        A2.requires_grad_(False)

    model2 = PeftModel.from_pretrained(bundle2.backbone, save_dir)
    model2 = accelerator.prepare(model2)

    scorer2 = MaskedMarginalScorer(
        tokenizer=bundle2.tokenizer,
        a_module=A2,
        spurs_ddg=spurs_ddg_tensor,
        aa_token_ids=aa_token_ids2,
    )
    evaluator2 = ConFitEvaluator(
        scorer=scorer2, accelerator=accelerator, tokenizer=bundle2.tokenizer
    )
    result = evaluator2.evaluate(model2, testloader, is_test=True)
    sr = result.spearman_correlation

    pred_csv = pd.DataFrame({
        f"{args.model_seed}": result.scores,
        "mutation": result.mutation_ids,
        "y_true": result.ground_truth,
    })

    pred_save_path = Path(
        f"predicted_{args.run_suffix}",
        args.dataset,
        f"shot{shot}_seed{args.model_seed}_mode{args.a_type}"
        f"_ainit{args.a_init}_combined{args.combined_way}"
        f"_trainmode{args.train_mode}",
    )
    if accelerator.is_main_process:
        pred_save_path.mkdir(parents=True, exist_ok=True)
        pred_csv.to_csv(pred_save_path / "pred.csv", index=False)

    accelerator.print(
        f"=============test Spearman (early-stop checkpoint): {sr}=================="
    )


if __name__ == "__main__":
    main()
