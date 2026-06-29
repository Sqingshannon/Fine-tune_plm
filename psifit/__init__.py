"""ConFit — Context-aware Fine-tuning for protein fitness prediction.

Public API surface. Import everything you need from this top-level package.

Example:
    >>> from psifit import TrainingConfig, ConFitTrainer, ExperimentRunner
"""

from psifit.config.schema import TrainingConfig
from psifit.config.loader import ConfigLoader
from psifit.data.dataset import MutationDataset
from psifit.data.preprocessing import DataPreprocessor
from psifit.data.splitter import DataSplitter
from psifit.models.registry import ModelVariant, ModelRegistry
from psifit.models.factory import ESMModelFactory
from psifit.models.scaling import (
    ScalingMode,
    AModule,
    BaseScalingStrategy,
    NoScalingStrategy,
    SingleScalingStrategy,
    PositionSpecificStrategy,
    ContextSpecificStrategy,
)
from psifit.losses.bradley_terry import BradleyTerryLoss
from psifit.losses.kl_regularization import KLRegularizationLoss
from psifit.scoring.masked_marginal import MaskedMarginalScorer
from psifit.metrics.correlation import spearman, compute_stat
from psifit.training.trainer import ConFitTrainer, TrainMode
from psifit.training.evaluator import ConFitEvaluator, EvaluationResult
from psifit.runners.experiment import ExperimentRunner
from psifit.runners.inference import InferenceAggregator
from psifit.utils.seeding import seed_everything
from psifit.utils.cleanup import ArtifactCleaner

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