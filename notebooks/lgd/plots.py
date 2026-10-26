"""Plots. Every figure is saved to outputs/figures before it is shown."""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from .config import FIGURE_DIR


def save_and_show(fig, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=300)
    plt.show()


def _grid(n_panels: int, n_cols: int = 4, panel_size: float = 4.0):
    n_rows = math.ceil(n_panels / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_size * n_cols, panel_size * n_rows), squeeze=False)
    for ax in axes.flat[n_panels:]:
        ax.set_visible(False)
    return fig, axes.flat


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def plot_confusion_matrices(y_true, probabilities: dict, filename: str, threshold: float = 0.5) -> None:
    fig, axes = _grid(len(probabilities))
    for ax, (name, proba) in zip(axes, probabilities.items()):
        y_pred = (np.asarray(proba) >= threshold).astype(int)
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap=plt.cm.Blues, ax=ax, colorbar=False)
        ax.set_title(name)
    save_and_show(fig, filename)


def plot_roc_curves(y_true, probabilities: dict, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, proba in probabilities.items():
        RocCurveDisplay.from_predictions(y_true, proba, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_title("ROC Curves (test set)")
    save_and_show(fig, filename)


def plot_calibration(y_true, probabilities: dict, filename: str, n_bins: int = 10) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, proba in probabilities.items():
        CalibrationDisplay.from_predictions(y_true, proba, n_bins=n_bins, strategy="quantile", name=name, ax=ax)
    ax.set_title("Calibration of P(cure) (test set)")
    save_and_show(fig, filename)


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------
def plot_residuals(y_true, predictions: dict, filename: str) -> None:
    fig, axes = _grid(len(predictions))
    for ax, (name, y_pred) in zip(axes, predictions.items()):
        y_pred = np.asarray(y_pred).ravel()
        sns.scatterplot(x=y_pred, y=np.asarray(y_true) - y_pred, ax=ax, s=6, alpha=0.4, linewidth=0)
        ax.axhline(0, color="red", linestyle="--")
        ax.set(title=name, xlabel="Predicted", ylabel="Residual")
    save_and_show(fig, filename)


def plot_predicted_vs_actual(y_true, predictions: dict, filename: str) -> None:
    fig, axes = _grid(len(predictions))
    lims = [np.min(y_true), np.max(y_true)]
    for ax, (name, y_pred) in zip(axes, predictions.items()):
        ax.scatter(y_true, np.asarray(y_pred).ravel(), s=6, alpha=0.4)
        ax.plot(lims, lims, "r--")  # 45-degree line
        ax.set(title=name, xlabel="Actual", ylabel="Predicted")
    save_and_show(fig, filename)


# ---------------------------------------------------------------------------
# Training, tuning and interpretation
# ---------------------------------------------------------------------------
def plot_history(history: dict, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["loss"], label="Training loss")
    ax.plot(history["val_loss"], label="Validation loss")
    if "best_epoch" in history:
        ax.axvline(history["best_epoch"] - 1, color="grey", linestyle=":", label="Best epoch")
    ax.set(title=title, xlabel="Epoch", ylabel="Loss")
    ax.legend()
    ax.grid(True)
    save_and_show(fig, filename)


def plot_optuna_history(study, title: str, ylabel: str, filename: str) -> None:
    trials = study.trials_dataframe()
    trials = trials[trials["state"] == "COMPLETE"]
    best_so_far = trials["value"].cummax() if study.direction.name == "MAXIMIZE" else trials["value"].cummin()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(trials["number"], trials["value"], "o", alpha=0.6, label="Trial")
    ax.plot(trials["number"], best_so_far, "-", color="red", label="Best so far")
    ax.set(title=title, xlabel="Trial", ylabel=ylabel)
    ax.legend()
    ax.grid(True)
    save_and_show(fig, filename)


def plot_coefficients(coefficients: pd.Series, title: str, filename: str, top: int = 20) -> None:
    top_coefficients = coefficients.reindex(coefficients.abs().sort_values(ascending=False).index[:top])[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(top_coefficients) + 1))
    colors = np.where(top_coefficients > 0, "tab:blue", "tab:red")
    ax.barh(top_coefficients.index, top_coefficients.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(title=title, xlabel="Coefficient (features scaled to [0, 1])")
    save_and_show(fig, filename)


def plot_shap_summary(model, X: pd.DataFrame, title: str, filename: str, max_rows: int = 2000) -> None:
    """SHAP beeswarm plot for a tree model (XGBoost / LightGBM / Random Forest)."""
    import shap

    sample = X.sample(min(max_rows, len(X)), random_state=0)
    shap_values = shap.TreeExplainer(model).shap_values(sample)
    if isinstance(shap_values, list):  # some classifiers return one array per class
        shap_values = shap_values[1]
    elif np.ndim(shap_values) == 3:
        shap_values = shap_values[..., 1]
    shap.summary_plot(shap_values, sample, max_display=15, show=False)
    fig = plt.gcf()
    fig.suptitle(title)
    save_and_show(fig, filename)
