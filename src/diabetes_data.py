
"""Data loading and inspection utilities for Practical Lab 2."""

import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


class DiabetesData:
    """Manage the diabetes progression dataset."""

    def __init__(self) -> None:
        self._features = None
        self._target = None
        self._dataframe = None

    def load_data(self) -> pd.DataFrame:
        """Load the dataset and return an independent copy."""
        dataset = load_diabetes(as_frame=True)

        self._features = dataset.data.copy()
        self._target = dataset.target.copy()
        self._dataframe = dataset.frame.copy()

        return self._dataframe.copy()

    def get_shape(self) -> tuple[int, int]:
        """Return the number of observations and columns."""
        self._require_loaded()
        return self._dataframe.shape

    def get_feature_names(self) -> list[str]:
        """Return the ten predictor names."""
        self._require_loaded()
        return self._features.columns.tolist()

    def get_summary(self) -> pd.DataFrame:
        """Return descriptive statistics for all columns."""
        self._require_loaded()
        return self._dataframe.describe().T

    def get_missing_values(self) -> pd.Series:
        """Count missing values in each column."""
        self._require_loaded()
        return self._dataframe.isna().sum()

    def get_target_correlations(self) -> pd.Series:
        """Return predictor correlations with the target, ranked by strength."""
        self._require_loaded()

        correlations = self._dataframe.corr(numeric_only=True)["target"]
        correlations = correlations.drop("target")

        return correlations.reindex(
            correlations.abs().sort_values(ascending=False).index
        )
        
    def split_data(self, random_state: int = 42) -> dict:
        """Split features and target into 75% train, 10% validation, 15% test."""
        self._require_loaded()

        # Reserve 15% for final testing.
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            self._features,
            self._target,
            test_size=0.15,
            random_state=random_state
        )

        # Validation is 10% of the original dataset.
        validation_fraction = 0.10 / 0.85

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=validation_fraction,
            random_state=random_state
        )

        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test
        }
    
    def get_duplicate_count(self) -> int:
        """Return the number of fully duplicated observations."""
        self._require_loaded()
        return int(self._dataframe.duplicated().sum())

    def _require_loaded(self) -> None:
        """Ensure data is available before accessing it."""
        if self._dataframe is None:
            raise RuntimeError("Call load_data() first.")
