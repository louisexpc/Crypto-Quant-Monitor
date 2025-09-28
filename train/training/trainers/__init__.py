"""Training entry points for single-fold runs."""

from train.training.trainers.utils import get_trainer  # noqa: F401
from train.training.trainers.classification import train_one_fold as train_classification_fold  # noqa: F401
from train.training.trainers.regression import train_one_fold as train_regression_fold  # noqa: F401
from train.training.trainers.xgb import _train_one_fold_xgb  # noqa: F401

__all__ = [
    "get_trainer",
    "train_classification_fold",
    "train_regression_fold",
    "_train_one_fold_xgb",
]
