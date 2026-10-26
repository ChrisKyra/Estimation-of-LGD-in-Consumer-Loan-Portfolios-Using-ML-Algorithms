"""Hyperparameter search with Optuna (replaces the grid searches of the thesis)."""

import time

import numpy as np
import optuna
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split

from .config import CV_FOLDS, SEED, TUNING_SAMPLE

optuna.logging.set_verbosity(optuna.logging.WARNING)


def tuning_rows(y, classification: bool, max_rows: int = TUNING_SAMPLE) -> np.ndarray:
    """Positions of at most `max_rows` rows (stratified for classification) used for the hyperparameter search."""
    positions = np.arange(len(y))
    if len(y) <= max_rows:
        return positions
    sample, _ = train_test_split(positions, train_size=max_rows, random_state=SEED,
                                 stratify=np.asarray(y) if classification else None)
    return np.sort(sample)


def tune_with_cv(build_estimator, X, y, scoring: str, n_trials: int, classification: bool, study_name: str):
    """Maximise the mean K-fold CV score on the training set (on a sample of it if it is large, see `tuning_rows`).

    `build_estimator(trial)` suggests hyperparameters and returns an unfitted estimator.
    Returns the study, the best estimator refitted on the whole training set, and that refit time (s).
    """
    folds = (StratifiedKFold if classification else KFold)(CV_FOLDS, shuffle=True, random_state=SEED)
    rows = tuning_rows(y, classification)
    X_tune, y_tune = X.iloc[rows], y.iloc[rows]

    def objective(trial):
        return cross_val_score(build_estimator(trial), X_tune, y_tune, cv=folds, scoring=scoring).mean()

    study = optuna.create_study(direction="maximize", study_name=study_name,
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_estimator = clone(build_estimator(optuna.trial.FixedTrial(study.best_params)))
    start = time.perf_counter()
    best_estimator.fit(X, y)
    return study, best_estimator, time.perf_counter() - start


def tune_network(train_trial, n_trials: int, study_name: str):
    """Minimise the validation loss returned by `train_trial(trial)`; weak trials are pruned early."""
    study = optuna.create_study(direction="minimize", study_name=study_name,
                                sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10))
    study.optimize(train_trial, n_trials=n_trials, show_progress_bar=True)
    return study


def fit_xgboost_with_early_stopping(estimator, X, y, classification: bool, max_rounds: int = 5000,
                                    early_stopping_rounds: int = 50):
    """Find the number of boosting rounds by K-fold CV with early stopping, then refit on all rows.

    The tuned learning rate is kept; only `n_estimators` is replaced by the CV-optimal number of rounds.
    Returns the fitted estimator, the number of rounds and the refit time (s).
    """
    import xgboost as xgb

    params = {key: value for key, value in estimator.get_xgb_params().items() if value is not None}
    params["eval_metric"] = "logloss" if classification else "rmse"
    history = xgb.cv(params, xgb.DMatrix(X, label=y), num_boost_round=max_rounds, nfold=CV_FOLDS,
                     stratified=classification, early_stopping_rounds=early_stopping_rounds, seed=SEED, shuffle=True)
    best_rounds = len(history)
    start = time.perf_counter()
    fitted = clone(estimator).set_params(n_estimators=best_rounds).fit(X, y)
    return fitted, best_rounds, time.perf_counter() - start
