"""Read and write the churn datasets.

`load_raw` reads the downloaded file. `load_processed` reads the model-ready file that
`features.engineering.build_model_frame` produces and study 3 will re-use.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config.settings import (
    PROCESSED_DATA_PATH,
    PROCESSED_DIR,
    RAW_DATA_PATH,
    TARGET,
)


def load_raw(path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw Kaggle-style churn file (10,000 rows, 14 columns)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"raw dataset not found: {path}")
    return pd.read_csv(path)


def load_processed(path: Path | str = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load the processed, model-ready dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"processed dataset not found: {path} - run the notebook or "
            "build_processed.py first"
        )
    return pd.read_csv(path)


def processed_exists(path: Path | str = PROCESSED_DATA_PATH) -> bool:
    """True when the processed dataset is on disk."""
    return Path(path).exists()


def save_processed(frame: pd.DataFrame, path: Path | str = PROCESSED_DATA_PATH) -> Path:
    """Write the processed dataset to disk and return the path."""
    path = Path(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def churn_summary(frame: pd.DataFrame | None = None) -> dict:
    """Row count, churn count and churn rate for the sidebar / health checks."""
    frame = load_processed() if frame is None else frame
    if TARGET not in frame.columns:
        return {"rows": int(len(frame)), "churned": None, "churn_rate": None}
    return {
        "rows": int(len(frame)),
        "churned": int(frame[TARGET].sum()),
        "churn_rate": float(frame[TARGET].mean()),
    }


__all__ = [
    "PROCESSED_DIR", "churn_summary", "load_processed", "load_raw",
    "processed_exists", "save_processed",
]
