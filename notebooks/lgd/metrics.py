"""Evaluation metrics, bootstrap confidence intervals and result tables."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from .config import N_BOOTSTRAP, SEED


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    """Label metrics (weighted, as in the thesis) plus probability metrics, which matter for LGD."""
    proba = np.asarray(proba).ravel()
    y_pred = (proba >= threshold).astype(int)
    auc = roc_auc_score(y_true, proba)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted"),
        "F1 Score": f1_score(y_true, y_pred, average="weighted"),
        "ROC-AUC": auc,
        "Gini": 2 * auc - 1,
        "Brier": brier_score_loss(y_true, proba),
        "Log-loss": log_loss(y_true, np.clip(proba, 1e-7, 1 - 1e-7)),
    }


def regression_metrics(y_true, y_pred) -> dict:
    y_pred = np.asarray(y_pred).ravel()
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R^2": r2_score(y_true, y_pred),
    }


def bootstrap_ci(y_true, y_pred, metric, n_bootstrap: int = N_BOOTSTRAP, level: float = 0.95,
                 seed: int = SEED) -> tuple:
    """Percentile bootstrap confidence interval of `metric(y_true, y_pred)` on resampled test rows."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred).ravel()
    rng = np.random.default_rng(seed)
    scores = [metric(y_true[idx], y_pred[idx])
              for idx in rng.integers(0, len(y_true), size=(n_bootstrap, len(y_true)))]
    alpha = (1 - level) / 2
    return tuple(np.quantile(scores, [alpha, 1 - alpha]))


def format_ci(value: float, ci: tuple, digits: int = 4) -> str:
    return f"{value:.{digits}f} [{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"


def results_table(results: dict) -> pd.DataFrame:
    """{model_name: {metric: value}} -> DataFrame with one row per model."""
    return pd.DataFrame(results).T
