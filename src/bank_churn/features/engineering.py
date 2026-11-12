"""Feature engineering, identical to notebooks/modelling.ipynb.

Steps
-----
1. drop the identifiers (RowNumber, CustomerId, Surname)
2. add ZeroBalance = 1 when Balance == 0
3. one-hot encode Geography and Gender
4. order the columns, target last

`to_feature_matrix` then selects the exact training column order, so a single customer
record carries the same 14 columns the model was fitted on.
"""

from __future__ import annotations

import pandas as pd

from ..config.settings import (
    CATEGORICAL_COLUMNS,
    FEATURE_ORDER,
    ID_COLUMNS,
    TARGET,
)


def add_zero_balance(frame: pd.DataFrame) -> pd.DataFrame:
    """Add `ZeroBalance` (1 when the account balance is exactly 0)."""
    out = frame.copy()
    out["ZeroBalance"] = (out["Balance"] == 0).astype("int8")
    return out


def encode_categoricals(frame: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode Geography and Gender, keeping every level."""
    present = [col for col in CATEGORICAL_COLUMNS if col in frame.columns]
    if not present:
        return frame
    return pd.get_dummies(frame, columns=present, prefix=["Geo", "Gender"], dtype="int8")


def build_model_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw rows -> model-ready frame (identifiers dropped, target last when present)."""
    frame = raw.drop(columns=[c for c in ID_COLUMNS if c in raw.columns]).copy()
    frame = add_zero_balance(frame)
    frame = encode_categoricals(frame)

    if TARGET in frame.columns:
        frame = frame[[c for c in frame.columns if c != TARGET] + [TARGET]]
    return frame


def to_feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the training columns in order; missing one-hot levels become 0."""
    missing = [col for col in FEATURE_ORDER if col not in frame.columns]
    matrix = frame.reindex(columns=FEATURE_ORDER, fill_value=0)
    if missing:
        # expected for a single record: e.g. a German customer has no Geo_France column
        pass
    return matrix.astype("float64")
