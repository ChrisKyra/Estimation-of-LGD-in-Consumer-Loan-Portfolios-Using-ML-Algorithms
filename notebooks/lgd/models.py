"""scikit-learn style estimators used by the recovery notebook."""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_predict

from .config import CV_FOLDS, SEED

BOUNDARY = 1e-4   # recovery rates within this distance of 0 or 1 count as exactly 0 or 1


class ZeroOneInflatedRegressor(BaseEstimator, RegressorMixin):
    """Two-part model for recovery rates with point masses at 0 and 1.

    1. `classifier` (multi-class) predicts P(y = 0), P(y = 1) and P(0 < y < 1).
    2. `regressor` predicts E[y | 0 < y < 1], trained on the partial recoveries only.
    Prediction: E[y] = P(y = 1) + P(0 < y < 1) · E[y | 0 < y < 1].
    """

    def __init__(self, classifier, regressor):
        self.classifier = classifier
        self.regressor = regressor

    @staticmethod
    def outcome_class(y) -> np.ndarray:
        """0 = no recovery, 1 = full recovery, 2 = partial recovery."""
        y = np.asarray(y)
        return np.where(y <= BOUNDARY, 0, np.where(y >= 1 - BOUNDARY, 1, 2))

    def fit(self, X, y):
        classes = self.outcome_class(y)
        partial = classes == 2
        # Some datasets have no recoveries of exactly 0 or 1: the classifier only sees the classes that occur.
        self.classes_present_ = np.unique(classes)
        if len(self.classes_present_) > 1:
            encoded = np.searchsorted(self.classes_present_, classes)
            self.classifier_ = clone(self.classifier).fit(X, encoded)
        self.regressor_ = clone(self.regressor).fit(X[partial], np.asarray(y)[partial])
        return self

    def predict(self, X):
        probabilities = np.zeros((len(X), 3))
        if len(self.classes_present_) > 1:
            probabilities[:, self.classes_present_] = self.classifier_.predict_proba(X)
        else:
            probabilities[:, self.classes_present_[0]] = 1.0
        return probabilities[:, 1] + probabilities[:, 2] * np.clip(self.regressor_.predict(X), 0, 1)


class StackedRegressor(BaseEstimator, RegressorMixin):
    """Stacking: the out-of-fold prediction of `base` becomes an extra feature for `meta`.

    For the training rows the base prediction comes from `CV_FOLDS`-fold cross-validation (models that did not see
    those rows); for new rows it comes from `base` refitted on all training rows.
    """

    def __init__(self, base, meta):
        self.base = base
        self.meta = meta

    def fit(self, X, y):
        out_of_fold = cross_val_predict(clone(self.base), X, y, cv=KFold(CV_FOLDS, shuffle=True, random_state=SEED))
        self.base_ = clone(self.base).fit(X, y)
        self.meta_ = self.meta.fit(np.column_stack([np.asarray(X), out_of_fold]), y)
        return self

    def predict(self, X):
        return self.meta_.predict(np.column_stack([np.asarray(X), self.base_.predict(X)]))
