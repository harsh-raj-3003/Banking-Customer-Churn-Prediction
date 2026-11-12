"""FastAPI service for churn scoring.

The application object lives in ``bank_churn.api.main`` and is started with::

    PYTHONPATH=src python -m uvicorn bank_churn.api.main:app

Nothing is imported here on purpose: the UI reads ``bank_churn.api.schemas`` for validation, and
an eager ``from .main import app`` would load xgboost and scikit-learn for every page load.
"""

__all__: list[str] = []
