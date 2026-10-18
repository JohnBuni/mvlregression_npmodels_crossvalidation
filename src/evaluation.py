
"""Reusable model evaluation utilities."""

import pandas as pd

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
)


class ModelEvaluator:
    """Evaluate regression models on training and validation data."""

    def __init__(self, models: dict):
        self.models = models

    def evaluate_regression(self, X, y, dataset_name: str) -> pd.DataFrame:
        """Evaluate each model using R², MAE, and MAPE."""
        results = []

        for degree, model in self.models.items():
            predictions = model.predict(X)

            results.append({
                "Degree": degree,
                "Dataset": dataset_name,
                "R²": r2_score(y, predictions),
                "MAE": mean_absolute_error(y, predictions),
                "MAPE (%)": mean_absolute_percentage_error(y, predictions) * 100,
            })

        return pd.DataFrame(results)
