"""Feature engineering shared by the notebooks, the API and the UI."""

from .engineering import (
    add_zero_balance,
    build_model_frame,
    encode_categoricals,
    to_feature_matrix,
)

__all__ = ["add_zero_balance", "build_model_frame", "encode_categoricals", "to_feature_matrix"]
