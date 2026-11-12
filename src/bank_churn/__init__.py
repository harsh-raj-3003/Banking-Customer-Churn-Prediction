"""Bank customer churn: shared code for the data pipeline, API and UI.

Sub-packages
------------
config   : paths and constants shared by every script
data     : reading and writing the raw / processed datasets
features : feature engineering, identical to the notebooks
models   : model loading and scoring (empty until study 3 finishes)
api      : FastAPI service
ui       : Streamlit frontend
"""

__version__ = "0.1.0"
