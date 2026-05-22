"""ConFit — Context-aware Fine-tuning for protein fitness prediction.

Public API surface. Import everything you need from this top-level package.

Example:
    >>> from confit import TrainingConfig, ConFitTrainer, ExperimentRunner
"""

from confit.config.schema import TrainingConfig
from confit.config.loader import ConfigLoader
from confit.data.dataset import MutationDataset
from confit.data.preprocessing import DataPreprocessor
from confit.data.splitter import DataSplitter
from confit.models.registry import ModelVariant, ModelRegistry
from confit.models.factory import ESMModelFactory
from confit.models.scaling import (
    ScalingMode,
    AModule,
    BaseScalingStrategy,
    NoScalingStrategy,
    SingleScalingStrategy,
    PositionSpecificStrategy,
    ContextSpecificStrategy,
)
from confit.losses.bradley_terry import BradleyTerryLoss
from confit.losses.kl_regularization import KLRegularizationLoss
from confit.scoring.masked_marginal import MaskedMarginalScorer
from confit.metrics.correlation import spearman, compute_stat
from confit.training.trainer import ConFitTrainer, TrainMode
from confit.training.evaluator import ConFitEvaluator, EvaluationResult
from confit.runners.experiment import ExperimentRunner
from confit.runners.inference import InferenceAggregator
from confit.utils.seeding import seed_everything
from confit.utils.cleanup import ArtifactCleaner

__all__ = [
    # Config
    "TrainingConfig",
    "ConfigLoader",
    # Data
    "MutationDataset",
    "DataPreprocessor",
    "DataSplitter",
    # Models
    "ModelVariant",
    "ModelRegistry",
    "ESMModelFactory",
    "ScalingMode",
    "AModule",
    "BaseScalingStrategy",
    "NoScalingStrategy",
    "SingleScalingStrategy",
    "PositionSpecificStrategy",
    "ContextSpecificStrategy",
    # Losses
    "BradleyTerryLoss",
    "KLRegularizationLoss",
    # Scoring
    "MaskedMarginalScorer",
    # Metrics
    "spearman",
    "compute_stat",
    # Training
    "ConFitTrainer",
    "TrainMode",
    "ConFitEvaluator",
    "EvaluationResult",
    # Runners
    "ExperimentRunner",
    "InferenceAggregator",
    # Utils
    "seed_everything",
    "ArtifactCleaner",
]