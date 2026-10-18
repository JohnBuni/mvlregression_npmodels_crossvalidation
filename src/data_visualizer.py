
"""Reusable exploratory data visualization methods."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class DataVisualizer:
    """Create visualizations for the diabetes dataset."""

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self._df = dataframe.copy()

    def plot_histograms(self) -> None:
        """Plot distributions of all predictors and the target."""
        self._df.hist(
            bins=20,
            figsize=(14, 10),
            edgecolor="black"
        )

        plt.suptitle("Diabetes Dataset: Variable Distributions")
        plt.tight_layout()
        plt.show()

    def plot_scatter(self, feature: str) -> None:
        """Plot a predictor against disease progression."""
        if feature not in self._df.columns or feature == "target":
            raise ValueError(f"Invalid predictor: {feature}")

        plt.figure(figsize=(8, 5))

        sns.scatterplot(
            data=self._df,
            x=feature,
            y="target",
            alpha=0.65
        )

        plt.title(f"{feature.upper()} vs. Disease Progression")
        plt.xlabel(f"{feature.upper()} (standardized)")
        plt.ylabel("Disease Progression")
        plt.tight_layout()
        plt.show()

    def plot_correlation_matrix(self) -> None:
        """Display correlations between all numeric variables."""
        correlations = self._df.corr(numeric_only=True)

        plt.figure(figsize=(12, 9))

        sns.heatmap(
            correlations,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0
        )

        plt.title("Diabetes Dataset: Correlation Matrix")
        plt.tight_layout()
        plt.show()

    
    def plot_polynomial_regression(
        self,
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        degree: int
    ) -> None:
        """Plot observed BMI values and the fitted polynomial curve."""
        import numpy as np

        plt.figure(figsize=(10, 6))

        plt.scatter(
            X_train["bmi"], y_train,
            alpha=0.4, label="Training data"
        )
        plt.scatter(
            X_val["bmi"], y_val,
            alpha=0.6, label="Validation data"
        )
        plt.scatter(
            X_test["bmi"], y_test,
            alpha=0.6, label="Test data"
        )

        # Generate a smooth curve across the observed BMI range.
        all_bmi = pd.concat([
            X_train["bmi"],
            X_val["bmi"],
            X_test["bmi"]
        ])

        bmi_values = np.linspace(
            all_bmi.min(),
            all_bmi.max(),
            300
        )

        bmi_grid = pd.DataFrame({"bmi": bmi_values})
        predictions = model.predict(bmi_grid)

        plt.plot(
            bmi_values,
            predictions,
            color="black",
            linewidth=2.5,
            label=f"Polynomial fit (degree {degree})"
        )

        plt.title("Polynomial Regression: BMI vs. Diabetes Progression")
        plt.xlabel("BMI (standardized)")
        plt.ylabel("Disease Progression")
        plt.legend()
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.show()
