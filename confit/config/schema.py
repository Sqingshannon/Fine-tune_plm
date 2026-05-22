"""Pydantic v2 schema for ConFit training configuration.

Replaces the raw dict loaded from YAML everywhere in the original codebase.
All fields are validated and type-coerced at load time, so downstream code
never needs to call ``int(config['batch_size'])`` manually.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, ConfigDict, field_validator


class TrainingConfig(BaseModel):
    """Validated configuration for a ConFit training run.

    Attributes:
        model: ESM model variant name (e.g. ``'ESM-1v'``).
        batch_size: Total batch size across all GPUs.
        gpu_number: Number of GPUs used (used to derive per-device batch size).
        lora_r: LoRA rank.
        lora_alpha: LoRA scaling factor.
        lora_dropout: Dropout probability applied inside LoRA layers.
        ini_lr: Initial learning rate for AdamW.
        min_lr: Minimum learning rate for cosine annealing.
        max_epochs: Maximum number of training epochs per stage.
        lambda_reg: Weight of the KL regularisation term.
        shot: Default number of labelled training examples (k-shot).
        endure_time: Early-stopping patience (epochs without improvement).
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    model: str = Field(..., description="ESM model variant, e.g. 'ESM-1v'.")
    batch_size: int = Field(..., gt=0, description="Total batch size across all GPUs.")
    gpu_number: int = Field(..., gt=0, description="Number of GPUs.")
    lora_r: int = Field(..., gt=0, description="LoRA rank.")
    lora_alpha: int = Field(..., gt=0, description="LoRA alpha scaling.")
    lora_dropout: float = Field(..., ge=0.0, le=1.0, description="LoRA dropout.")
    ini_lr: float = Field(..., gt=0.0, description="Initial learning rate.")
    min_lr: float = Field(..., gt=0.0, description="Minimum learning rate.")
    max_epochs: int = Field(..., gt=0, description="Maximum training epochs.")
    lambda_reg: float = Field(..., ge=0.0, description="KL regularisation weight.")
    shot: int = Field(..., gt=0, description="Default k-shot training size.")
    endure_time: int = Field(..., gt=0, description="Early-stopping patience.")

    @field_validator("min_lr")
    @classmethod
    def min_lr_below_ini(cls, v: float, info: object) -> float:
        """Ensure min_lr does not exceed ini_lr."""
        data = info.data if hasattr(info, "data") else {}
        ini = data.get("ini_lr")
        if ini is not None and v > ini:
            raise ValueError(
                f"min_lr ({v}) must be <= ini_lr ({ini})."
            )
        return v

    @property
    def per_device_batch_size(self) -> int:
        """Per-GPU batch size derived from total batch_size and gpu_number."""
        return self.batch_size // self.gpu_number