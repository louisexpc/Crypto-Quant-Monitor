"""Loss builders for classification and regression."""

from .cls import build_classification_loss  # noqa: F401
from .reg import build_regression_loss  # noqa: F401

__all__ = ["build_classification_loss", "build_regression_loss"]
