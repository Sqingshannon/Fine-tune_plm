"""Losses subpackage — abstract base and concrete loss implementations."""

from confit.losses.base import BaseLoss
from confit.losses.bradley_terry import BradleyTerryLoss
from confit.losses.kl_regularization import KLRegularizationLoss

__all__ = ["BaseLoss", "BradleyTerryLoss", "KLRegularizationLoss"]