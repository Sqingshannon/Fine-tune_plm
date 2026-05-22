"""Config subpackage — schema validation and YAML loading."""

from confit.config.schema import TrainingConfig
from confit.config.loader import ConfigLoader

__all__ = ["TrainingConfig", "ConfigLoader"]