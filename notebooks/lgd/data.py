"""Feature engineering, train/validation/test split and loading of the modelling dataset."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import MODEL_DATA_FILE, NON_FEATURE_COLUMNS, POST_DEFAULT_FEATURES, SEED, TARGETS

DATE_COLUMNS = ["DATE_OF_LOAN", "DATE_FROM", "DATE_WO"]  # origination, default, write-off
CORRELATION_THRESHOLD = 0.80
CORRELATED_FEATURES_TO_DROP = ["out_prncp", "total_rec_prncp", "collection_recovery_fee",
                               "total_rev_hi_lim", "grade", "sub_grade"]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Month of each event and days between events; the raw dates are dropped."""
    df = df.copy()
    loan, default, write_off = (pd.to_datetime(df[col]) for col in DATE_COLUMNS)

    df["loan_month"] = loan.dt.month
    df["default_month"] = default.dt.month
    df["wo_month"] = write_off.dt.month

    df["loan_to_default_days"] = (default - loan).dt.days
    df["loan_to_wo_days"] = (write_off - loan).dt.days
    df["default_to_wo_days"] = (write_off - default).dt.days

    return df.drop(columns=DATE_COLUMNS)


def one_hot(df: pd.DataFrame, column: str, reference: str) -> pd.DataFrame:
    """One-hot encode `column`, dropping the `reference` category to avoid the dummy-variable trap."""
    dummies = pd.get_dummies(df[column], prefix=column, prefix_sep="_", dtype=int)
    dummies = dummies.drop(columns=f"{column}_{reference}")
    return pd.concat([df.drop(columns=column), dummies], axis=1)


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    * home_ownership, marital_status_code, city: one-hot with the most frequent category as reference.
    * country: dropped, it is fully determined by city.
    * postal_code: dropped, city was derived from it.
    * grade, sub_grade: label codes, only kept for the correlation analysis (they are dropped there).
    """
    df = df.copy()
    for col in ["grade", "sub_grade"]:
        df[col] = df[col].astype("category").cat.codes
    for col in ["home_ownership", "marital_status_code", "city"]:
        df = one_hot(df, col, reference=df[col].mode()[0])
    return df.drop(columns=["country", "postal_code"])


def assign_split(df: pd.DataFrame, seed: int = SEED, stratify: str = "cure") -> pd.Series:
    """70% train / 10% validation / 20% test, stratified on the `stratify` column."""
    labels = df[stratify].to_numpy()
    train_pos, temp_pos = train_test_split(np.arange(len(df)), test_size=0.3, stratify=labels, random_state=seed)
    test_pos, valid_pos = train_test_split(temp_pos, test_size=1 / 3, stratify=labels[temp_pos], random_state=seed)
    split = np.full(len(df), "train", dtype=object)
    split[valid_pos] = "valid"
    split[test_pos] = "test"
    return pd.Series(split, index=df.index, name="split")


def high_correlation_pairs(corr: pd.DataFrame, threshold: float = CORRELATION_THRESHOLD) -> pd.DataFrame:
    upper_triangle = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = upper_triangle.stack()
    pairs = pairs[pairs.abs() > threshold].rename("correlation")
    return (pairs.rename_axis(["feature_1", "feature_2"]).reset_index()
            .sort_values("correlation", key=abs, ascending=False, ignore_index=True))


# ---------------------------------------------------------------------------
# Modelling dataset
# ---------------------------------------------------------------------------
@dataclass
class Split:
    X: pd.DataFrame
    y: pd.Series

    def __len__(self):
        return len(self.y)


def load_model_dataset() -> pd.DataFrame:
    if not MODEL_DATA_FILE.exists():
        raise FileNotFoundError(f"{MODEL_DATA_FILE} not found - run notebook 01 first.")
    return pd.read_csv(MODEL_DATA_FILE, index_col="loan_id")


def feature_columns(df: pd.DataFrame, feature_set: str = "at_default") -> list:
    """`thesis`: every feature used in the thesis. `at_default`: only what is known at the moment of default."""
    columns = [col for col in df.columns if col not in TARGETS + NON_FEATURE_COLUMNS]
    if feature_set == "at_default":
        return [col for col in columns if col not in POST_DEFAULT_FEATURES]
    if feature_set == "thesis":
        return columns
    raise ValueError(f"unknown feature set {feature_set!r}")


def make_splits(df: pd.DataFrame, target: str, feature_set: str = "at_default", rows=None) -> dict:
    """{'train' | 'valid' | 'test': Split}. `rows` is an optional boolean mask, e.g. non-cured loans only."""
    if rows is not None:
        df = df[rows]
    features = feature_columns(df, feature_set)
    return {name: Split(part[features], part[target])
            for name, part in df.groupby("split", sort=False)}
