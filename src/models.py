
"""Object-oriented machine learning models for Practical Lab 2."""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class PolynomialRegressionModel(BaseEstimator, RegressorMixin):
    """Fit polynomial regression using a scikit-learn pipeline."""

    def __init__(self, degree: int = 1):
        self.degree = degree
        self.pipeline = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit polynomial features and a linear regression model."""
        if not isinstance(self.degree, int) or not 0 <= self.degree <= 5:
            raise ValueError("Degree must be an integer between 0 and 5.")

        
        self.pipeline = Pipeline([
            (
                "polynomial",
                PolynomialFeatures(
                    degree=self.degree,
                    include_bias=(self.degree == 0)
                )
            ),
            ("regression", LinearRegression())
        ])

        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict target values for new observations."""
        if self.pipeline is None:
            raise RuntimeError("Fit the model before predicting.")

        return self.pipeline.predict(X)

    def get_coefficients(self) -> np.ndarray:
        """Return fitted regression coefficients."""
        if self.pipeline is None:
            raise RuntimeError("Fit the model before accessing coefficients.")

        return self.pipeline.named_steps["regression"].coef_

    def get_intercept(self) -> float:
        """Return the fitted intercept."""
        if self.pipeline is None:
            raise RuntimeError("Fit the model before accessing the intercept.")

        return float(self.pipeline.named_steps["regression"].intercept_)

    def get_feature_names(self, input_features: list[str]) -> list[str]:
        """Return the names of generated polynomial features."""
        if self.pipeline is None:
            raise RuntimeError("Fit the model before accessing feature names.")

        polynomial = self.pipeline.named_steps["polynomial"]

        return polynomial.get_feature_names_out(input_features).tolist()


class MultivariatePolynomialRegressionModel:
    """Polynomial regression using multiple input features."""

    def __init__(self, degree: int = 2):
        if degree < 2:
            raise ValueError("Multivariate polynomial degree must be at least 2.")

        self.degree = degree
        self.pipeline = Pipeline([
            (
                "polynomial",
                PolynomialFeatures(
                    degree=self.degree,
                    include_bias=False
                )
            ),
            ("regression", LinearRegression())
        ])

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train the multivariate polynomial regression model."""
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict disease progression using multiple predictors."""
        return self.pipeline.predict(X)

    def get_feature_names(self, input_features: list[str]) -> list[str]:
        """Return generated polynomial feature names."""
        polynomial = self.pipeline.named_steps["polynomial"]
        return polynomial.get_feature_names_out(input_features).tolist()


class DecisionTreeRegressionModel:
    """Decision tree regression with configurable maximum depth."""

    def __init__(self, max_depth: int = 3, random_state: int = 42):
        self.max_depth = max_depth
        self.random_state = random_state

        self.model = DecisionTreeRegressor(
            max_depth=self.max_depth,
            random_state=self.random_state
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train the decision tree regression model."""
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict diabetes progression using the trained tree."""
        return self.model.predict(X)

    def get_depth(self) -> int:
        """Return the depth of the trained decision tree."""
        return self.model.get_depth()

    def get_leaf_count(self) -> int:
        """Return the number of leaf nodes."""
        return self.model.get_n_leaves()


class KNNRegressionModel:
    """k-Nearest Neighbors regression with feature scaling."""

    def __init__(self, n_neighbors: int = 3):
        self.n_neighbors = n_neighbors

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(
                n_neighbors=self.n_neighbors
            ))
        ])

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train the kNN regression model."""
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict disease progression using nearby observations."""
        return self.pipeline.predict(X)


class LogisticClassificationModel:
    """Binary logistic regression with configurable regularization."""

    def __init__(self, C: float = 1.0, random_state: int = 42):
        self.C = C
        self.random_state = random_state

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("logistic", LogisticRegression(
                C=self.C,
                max_iter=2000,
                random_state=self.random_state
            ))
        ])

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train the binary classification model."""
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the binary disease-progression category."""
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted probabilities for both classes."""
        return self.pipeline.predict_proba(X)

