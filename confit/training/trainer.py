"""ConFit trainer — full LoRA + A joint training and two-stage A-only training.

Replaces the ``train()`` function and the monolithic ``main()`` training loops
from the original ``train.py``.  Training logic is split into small, named
methods: ``_run_epoch``, ``_run_val``, ``_save_checkpoint``, and ``_load_stage1``.

Two training modes are supported via the :class:`TrainMode` enum:

* :attr:`TrainMode.FULL` — jointly trains LoRA weights and the A module.
* :attr:`TrainMode.A_ONLY` — Stage 1 trains LoRA only (A frozen),
  Stage 2 freezes LoRA and trains A only.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch
from peft import PeftModel, LoraConfig, get_peft_model
from torch.utils.data import DataLoader

from confit.config.schema import TrainingConfig
from confit.losses.bradley_terry import BradleyTerryLoss
from confit.losses.kl_regularization import KLRegularizationLoss
from confit.models.scaling import AModule, ScalingMode
from confit.scoring.masked_marginal import MaskedMarginalScorer
from confit.training.base import BaseTrainer
from confit.training.evaluator import ConFitEvaluator, EvaluationResult


class TrainMode(str, Enum):
    """Supported ConFit training modes.

    Attributes:
        FULL: Joint LoRA + A training.
        A_ONLY: Two-stage: LoRA-first, then A-only.
    """

    FULL = "full"
    A_ONLY = "a_only"


class ConFitTrainer(BaseTrainer):
    """Trains an ESM model with LoRA and an optional SPURS scaling module.

    Args:
        config: Validated :class:`~confit.config.schema.TrainingConfig`.
        model: PEFT-wrapped ESM backbone (prepared by Accelerator).
        model_reg: Frozen reference ESM model (prepared by Accelerator).
        a_module: :class:`~confit.models.scaling.AModule` instance.
        scorer: :class:`~confit.scoring.masked_marginal.MaskedMarginalScorer`.
        evaluator: :class:`~confit.training.evaluator.ConFitEvaluator`.
        accelerator: :class:`~accelerate.Accelerator` instance.
        train_mode: :class:`TrainMode` controlling the training strategy.
        tokenizer: ESM tokenizer.
        basemodel: The unwrapped ESM backbone (needed for Stage 2 reload).

    Example:
        >>> trainer = ConFitTrainer(config, model, model_reg, A, scorer, evaluator,
        ...                         accelerator, TrainMode.FULL, tokenizer, basemodel)
        >>> best_sr = trainer.fit(train_loader, val_loader, save_dir)
    """

    _LORA_TARGET_MODULES = ["query", "value"]
    _STAGE2_EPOCHS = 20
    _STAGE2_T0 = 40

    def __init__(
        self,
        config: TrainingConfig,
        model: Any,
        model_reg: Any,
        a_module: AModule,
        scorer: MaskedMarginalScorer,
        evaluator: ConFitEvaluator,
        accelerator: Any,
        train_mode: TrainMode,
        tokenizer: Any,
        basemodel: Any,
    ) -> None:
        self.config = config
        self.model = model
        self.model_reg = model_reg
        self.a_module = a_module
        self.scorer = scorer
        self.evaluator = evaluator
        self.accelerator = accelerator
        self.train_mode = train_mode
        self.tokenizer = tokenizer
        self.basemodel = basemodel
        self._bt_loss = BradleyTerryLoss()
        self._kl_loss = KLRegularizationLoss()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_dir: Path,
    ) -> float:
        """Train the model according to :attr:`train_mode`.

        Args:
            train_loader: DataLoader over the labelled training set.
            val_loader: DataLoader over the validation fold.
            save_dir: Directory for writing checkpoints.

        Returns:
            Best validation Spearman ρ across all epochs.
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        if self.train_mode == TrainMode.FULL:
            return self._fit_full(train_loader, val_loader, save_dir)
        if self.train_mode == TrainMode.A_ONLY:
            return self._fit_a_only(train_loader, val_loader, save_dir)
        raise ValueError(f"Unsupported TrainMode: {self.train_mode!r}")

    # ------------------------------------------------------------------
    # Training modes
    # ------------------------------------------------------------------

    def _fit_full(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_dir: Path,
    ) -> float:
        """Jointly train LoRA weights and the A module."""
        self.accelerator.print("========start full LoRA + A training!============")
        self._set_a_grad(enabled=self.a_module.mode != ScalingMode.NONE)

        optimizer, scheduler = self._build_optimizer_scheduler(
            params=list(self.model.parameters()) + list(self.a_module.parameters()),
            t0=2 * self.config.max_epochs,
        )
        optimizer, scheduler = self.accelerator.prepare(optimizer, scheduler)
        return self._epoch_loop(
            train_loader, val_loader, optimizer, scheduler, save_dir, stage=None
        )

    def _fit_a_only(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_dir: Path,
    ) -> float:
        """Two-stage training: LoRA first, then A-only."""
        stage1_dir = save_dir / "stage1_confit"

        # ---------- Stage 1: LoRA only ----------
        self.accelerator.print("========STAGE 1: Training only model (LoRA)============")
        self._set_a_grad(enabled=False)

        opt1, sched1 = self._build_optimizer_scheduler(
            params=list(self.model.parameters()),
            t0=2 * self.config.max_epochs,
        )
        opt1, sched1 = self.accelerator.prepare(opt1, sched1)
        self._epoch_loop(train_loader, val_loader, opt1, sched1, stage1_dir, stage=1)

        # ---------- Stage 2: A only ----------
        self.accelerator.print("========STAGE 2: Training only A============")
        self.model = self._load_stage1(stage1_dir)
        self._freeze_model()
        self._set_a_grad(enabled=True)

        opt2, sched2 = self._build_optimizer_scheduler(
            params=list(self.a_module.parameters()),
            t0=self._STAGE2_T0,
        )
        opt2, sched2 = self.accelerator.prepare(opt2, sched2)
        return self._epoch_loop(
            train_loader, val_loader, opt2, sched2, save_dir,
            stage=2, max_epochs=self._STAGE2_EPOCHS
        )

    # ------------------------------------------------------------------
    # Epoch loop
    # ------------------------------------------------------------------

    def _epoch_loop(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Any,
        scheduler: Any,
        save_dir: Path,
        stage: Optional[int],
        max_epochs: Optional[int] = None,
    ) -> float:
        """Run the training/validation loop for up to *max_epochs* epochs.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            optimizer: Prepared AdamW optimizer.
            scheduler: Prepared cosine annealing scheduler.
            save_dir: Checkpoint directory.
            stage: Logging label (``None``, ``1``, or ``2``).
            max_epochs: Override for max epochs (defaults to config value).

        Returns:
            Best validation Spearman ρ.
        """
        max_epochs = max_epochs if max_epochs is not None else self.config.max_epochs
        prefix = f"Stage {stage} " if stage is not None else ""
        best_sr = -np.inf
        endure = 0

        for epoch in range(max_epochs):
            loss = self._run_epoch(train_loader, optimizer)
            self.accelerator.print(
                f"========{prefix}epoch{epoch}; training loss:{loss:.4f}================="
            )

            result: EvaluationResult = self.evaluator.evaluate(self.model, val_loader)
            sr = result.spearman_correlation
            self.accelerator.print(
                f"========{prefix}epoch{epoch}; val SR:{sr:.4f}================="
            )
            scheduler.step()

            if sr > best_sr:
                best_sr = sr
                endure = 0
                self._save_checkpoint(save_dir)
            else:
                endure += 1

            if sr == 1.0 or endure > self.config.endure_time:
                self.accelerator.print(
                    f"========{prefix}early stop at epoch{epoch}!============"
                )
                break

        return best_sr

    # ------------------------------------------------------------------
    # Per-step training
    # ------------------------------------------------------------------

    def _run_epoch(self, loader: DataLoader, optimizer: Any) -> float:
        """Run one training epoch and return the total loss.

        Args:
            loader: Training DataLoader.
            optimizer: Prepared optimizer.

        Returns:
            Total accumulated loss over all steps.
        """
        self.model.train()
        total_loss = 0.0

        for batch in loader:
            seq, mask, wt, wt_mask, pos, golden_score, _pid, _mutation = batch

            scores, logits = self.scorer.score(self.model, seq, mask, wt, pos)
            scores = scores.to(self.accelerator.device)

            l_bt = self._bt_loss(scores, golden_score)
            out_reg = self.model_reg(wt, wt_mask)
            l_kl = self._kl_loss(logits, out_reg.logits, seq, mask)
            loss = l_bt + self.config.lambda_reg * l_kl

            optimizer.zero_grad()
            self.accelerator.backward(loss)
            optimizer.step()
            total_loss += loss.item()

        return total_loss

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _save_checkpoint(self, save_dir: Path) -> None:
        """Save model LoRA weights and A module to *save_dir*.

        Args:
            save_dir: Directory to write ``adapter_model`` and ``A.pth``.
        """
        self.accelerator.wait_for_everyone()
        unwrapped = self.accelerator.unwrap_model(self.model)
        if self.accelerator.is_main_process:
            unwrapped.save_pretrained(save_dir)
            if self.a_module.mode != ScalingMode.NONE:
                self.accelerator.save(
                    self.a_module.state_dict(), save_dir / "A.pth"
                )
        self.accelerator.wait_for_everyone()

    def _load_stage1(self, stage1_dir: Path) -> Any:
        """Reload Stage-1 LoRA checkpoint and prepare with Accelerator.

        Args:
            stage1_dir: Directory written by Stage 1 :meth:`_save_checkpoint`.

        Returns:
            A new prepared model instance with Stage-1 weights.
        """
        model = PeftModel.from_pretrained(self.basemodel, stage1_dir)
        return self.accelerator.prepare(model)

    # ------------------------------------------------------------------
    # Parameter management
    # ------------------------------------------------------------------

    def _set_a_grad(self, enabled: bool) -> None:
        """Enable or disable gradient computation for A module parameters.

        Args:
            enabled: ``True`` to train, ``False`` to freeze.
        """
        for param in self.a_module.parameters():
            param.requires_grad = enabled

    def _freeze_model(self) -> None:
        """Freeze all parameters of the main ESM model."""
        for param in self.model.parameters():
            param.requires_grad = False

    def _build_optimizer_scheduler(
        self,
        params: list,
        t0: int,
    ) -> Tuple[Any, Any]:
        """Build AdamW + CosineAnnealingWarmRestarts scheduler.

        Args:
            params: List of parameter tensors to optimise.
            t0: ``T_0`` argument for cosine annealing restart period.

        Returns:
            Tuple ``(optimizer, scheduler)`` — not yet prepared by Accelerator.
        """
        optimizer = torch.optim.AdamW(params, lr=self.config.ini_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=t0, eta_min=self.config.min_lr
        )
        return optimizer, scheduler

    # ------------------------------------------------------------------
    # Factory class method
    # ------------------------------------------------------------------

    @classmethod
    def build_peft_model(
        cls,
        basemodel: Any,
        config: TrainingConfig,
        accelerator: Any,
    ) -> Any:
        """Wrap *basemodel* with LoRA and prepare with Accelerator.

        Args:
            basemodel: Unwrapped ESM backbone.
            config: Training config supplying LoRA hyperparameters.
            accelerator: Accelerator instance.

        Returns:
            Accelerator-prepared PEFT model.
        """
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=cls._LORA_TARGET_MODULES,
        )
        model = get_peft_model(basemodel, peft_config)
        return accelerator.prepare(model)