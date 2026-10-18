# Project Documentation — Practical Lab 2

**Topic:** Multivariate Linear Regression, Non-Parametric Models, and Cross-Validation  
**Dataset:** Scikit-learn Diabetes  
**Primary implementation:** [`../PracticalLab2.ipynb`](../PracticalLab2.ipynb)

## 1. Project purpose and scope

This project investigates supervised learning methods for predicting diabetes disease progression one year after baseline. Its main task is **regression**: predicting a continuous progression score from baseline measurements. An additional **binary classification** experiment uses logistic regression to distinguish observations above and below a training-derived progression threshold.

The purpose is to compare model complexity, validation performance, generalization, and stability rather than to develop a deployable clinical screening system. The progression score is **not** a diagnosis or a calibrated probability of diabetes.

The experiments follow the lab's specified 75%/10%/15% training/validation/test protocol. Five-fold cross-validation on the training partition is included as a supplemental robustness analysis.

## 2. Data and exploratory analysis

The scikit-learn Diabetes dataset contains **442 observations**, **10 numeric baseline predictors**, and a continuous target. Predictors are `age`, `sex`, `bmi`, `bp`, and `s1`–`s6`. The supplied features are scaled; in particular, BMI values used in the univariate experiments are **standardized BMI**, not raw BMI in kg/m².

Exploratory analysis includes descriptive statistics, feature and target histograms, scatterplots, and a correlation matrix. Among the observed feature–target correlations, BMI (**0.586**), `s5` (**0.566**), and blood pressure (**0.441**) are the strongest positive associations reported in the notebook; `s3` has a negative correlation of approximately **−0.395**. These associations do not establish causality.

The data-quality checks found **no missing values and no duplicate observations**. No imputation or row removal was required. Unusual but plausible observations were retained rather than removed without evidence of invalidity.

## 3. Data partitioning and leakage prevention

A reproducible split with `random_state=42` produced:

| Partition | Observations | Approximate share | Purpose |
|---|---:|---:|---|
| Training | 330 | 75% | Fit models and learn preprocessing parameters |
| Validation | 45 | 10% | Compare candidate configurations and select models |
| Test | 67 | 15% | Final evaluation after selection |

The original test partition is not used to select polynomial degree, tree depth, neighbor count, or logistic regularization strength. The classification threshold is computed from the **training target median (142.00)** and then applied to the other partitions. kNN and logistic regression use scikit-learn pipelines so scaling is fitted on training data rather than independently on evaluation observations.

## 4. Implementation architecture

The notebook is the experiment and reporting interface, with reusable object-oriented functionality separated into `src/`:

| Module | Responsibility |
|---|---|
| `src/diabetes_data.py` | Dataset loading, summary statistics, data-quality checks, correlations, and train/validation/test splitting |
| `src/data_visualizer.py` | Histograms, scatterplots, correlation matrix, and the fitted univariate polynomial curve |
| `src/models.py` | Wrappers for polynomial regression, decision tree regression, kNN regression, and logistic classification |
| `src/evaluation.py` | Regression comparison using R², MAE, and MAPE |

The project uses NumPy, pandas, Matplotlib, Seaborn, scikit-learn, and Jupyter. See [`../requirements.txt`](../requirements.txt) for the environment dependencies.

## 5. Evaluation metrics and interpretation

For continuous prediction, the experiments report:

- **R² (coefficient of determination):** explained variance relative to a constant-mean baseline; values can be negative on held-out data.
- **MAE (mean absolute error):** average absolute deviation between predicted and observed progression scores, in target-score units.
- **MAPE (mean absolute percentage error):** relative error expressed as a percentage. It can be especially sensitive to smaller actual target values.

For the separate binary classification task, the experiments report **accuracy, precision, recall, F1-score, and confusion matrices**. These measures answer a different question and should not be directly ranked against regression R² or MAE.

## 6. Part 2 — Univariate polynomial regression

Six polynomial regression models of degrees **0 through 5** use standardized `bmi` as the sole predictor. A degree-0 model is an intercept-only baseline. The models are evaluated on the training and validation partitions, with **validation R²** serving as the principal selection criterion.

The degree-5 model obtained the highest validation R² (**0.454**), although the differences between degrees 1–5 were small. Degree 3 had the lowest validation MAE (**41.741**), and degree 4 had the lowest validation MAPE (**40.241%**). These differing rankings illustrate why model selection should state its primary metric explicitly.

**Selected degree-5 model:**

| Measure | Result |
|---|---:|
| Validation R² | 0.454 |
| Test R² | 0.197 |
| Test MAE | 54.548 |
| Test MAPE | 47.884% |
| Fitted parameters | 6 (intercept plus five coefficients) |

The fitted equation, using standardized BMI `x`, is reported to two decimal places in the notebook:

```text
ŷ = 155.08 + 1131.59x − 3946.02x² − 26110.86x³
    + 768725.87x⁴ − 2823251.97x⁵
```

For **standardized BMI = 0.05**, the fitted model predicts a progression score of approximately **202.46**, calculated using `.predict()` rather than manually substituting rounded coefficients.

For univariate polynomial degree `d`, the number of fitted parameters is **d + 1**, including the intercept: degrees 0–5 therefore have **1, 2, 3, 4, 5, and 6** parameters, respectively.

The decline from validation R² 0.454 to test R² 0.197 shows that the selected model generalizes less strongly to the test observations. BMI alone cannot explain all relevant variation, and higher-order polynomial behavior can be sensitive near the observed range boundaries.

## 7. Part 3 — Multivariate and non-parametric models

All ten predictors are used for the multivariate experiments. Two configurations are fitted per required model family.

### 7.1 Multivariate polynomial regression

| Degree | Polynomial features | Training R² | Validation R² | Validation MAE |
|---|---:|---:|---:|---:|
| 2 | 65 | 0.598 | **0.603** | **37.745** |
| 3 | 285 | 0.891 | −125.339 | 333.915 |

Degree 2 was selected on validation performance. The degree-3 model has far better training fit but catastrophically poor validation fit, consistent with severe overfitting and potentially unstable high-dimensional coefficient estimation. Its 285 expanded features are numerous relative to the 330 training observations.

### 7.2 Decision tree regression

| Maximum depth | Training R² | Validation R² | Validation MAE |
|---|---:|---:|---:|
| 3 | 0.533 | **0.413** | **42.078** |
| 8 | 0.912 | 0.017 | 57.707 |

The depth-8 tree strongly fits training observations but generalizes poorly. The shallower depth-3 tree was selected.

### 7.3 k-nearest neighbors regression

| Neighbors (`k`) | Training R² | Validation R² | Validation MAE |
|---|---:|---:|---:|
| 3 | 0.641 | 0.323 | 50.526 |
| 15 | 0.489 | **0.553** | **40.188** |

The `k=15` configuration generalizes better to the validation partition than `k=3`, despite a lower training R². The kNN pipeline includes `StandardScaler` to ensure meaningful distance calculations within each fitted training subset.

### 7.4 Logistic regression classification

Logistic regression does **not** predict the continuous progression score. Instead, the target is converted into two categories using the training median **142.00**: class 1 denotes progression above this threshold, and class 0 denotes progression at or below it. This is a dataset-relative classification target, not a clinical risk cutoff.

| Regularization `C` | Validation accuracy | Validation F1 |
|---|---:|---:|
| 0.1 | **0.844** | **0.844** |
| 10.0 | 0.822 | 0.826 |

The selected `C=0.1` classifier had validation precision **0.826** and recall **0.864**. Its validation confusion matrix was `[[19, 4], [3, 19]]` (true negatives, false positives, false negatives, true positives in conventional matrix order).

### 7.5 Model selection and held-out testing

The highest **single-split validation R²** among the selected regression configurations belonged to multivariate polynomial degree 2 (**0.603**), ahead of kNN `k=15` (**0.553**) and decision tree depth 3 (**0.413**). It was therefore the final regression model tested in Part 3.

| Final held-out evaluation | Result |
|---|---:|
| Multivariate degree-2 polynomial test R² | **0.337** |
| Multivariate degree-2 polynomial test MAE | **45.867** |
| Multivariate degree-2 polynomial test MAPE | **39.474%** |
| Logistic regression (`C=0.1`) test accuracy | **0.746** |
| Logistic regression (`C=0.1`) test precision | **0.710** |
| Logistic regression (`C=0.1`) test recall | **0.733** |
| Logistic regression (`C=0.1`) test F1 | **0.721** |

The logistic test confusion matrix was `[[28, 9], [8, 22]]`, representing **50 correct classifications out of 67**. The lower test performance relative to validation illustrates the uncertainty associated with a small holdout partition.

The multivariate polynomial model outperformed the BMI-only degree-5 model on test R² (**0.337 vs. 0.197**), but substantial prediction error remained.

## 8. Part 4 — Five-fold cross-validation

As a supplemental analysis, three previously selected regression configurations were assessed using **5-fold shuffled K-fold cross-validation**, `random_state=42`, on **the original training partition only**. The original validation and test partitions were excluded from these folds. Each scikit-learn pipeline or estimator is refitted within each fold.

| Configuration | Mean cross-validation R² | Standard deviation |
|---|---:|---:|
| **kNN (`k=15`)** | **0.432** | **0.051** |
| Decision tree (depth 3) | 0.300 | 0.053 |
| Multivariate polynomial (degree 2) | 0.257 | 0.165 |

Unlike the single validation split, cross-validation favored **kNN**, which had the highest mean R² and lowest observed fold-to-fold variability. The degree-2 polynomial model varied more substantially across folds.

**Interpretation:** The original validation-based selection remains the procedure used for the reported held-out test results. Cross-validation provides an additional diagnostic of robustness; it does not retroactively replace the model-selection protocol. Because these configurations were selected using the validation split before this supplemental analysis, the cross-validation comparison should not be described as an independent, nested estimate of the entire selection process.

## 9. Limitations and responsible interpretation

1. **Small dataset and partitions.** With 442 observations, validation and test estimates may be sensitive to random assignment. Five-fold cross-validation partially illustrates this sensitivity but does not eliminate it.
2. **Limited input variables.** The ten provided baseline measurements cannot capture every determinant of subsequent disease progression.
3. **Complexity and instability.** High-degree polynomial expansion and deep trees show substantial overfitting in these experiments. Degree-5 BMI polynomial behavior may be unreliable near or beyond the observed feature range.
4. **Metric limitations.** R² is sensitive to variation in the evaluation set; MAPE can be difficult to interpret when actual target values are small. Multiple metrics are therefore reported.
5. **Classification threshold.** The logistic experiment's median-derived label is a methodological exercise, not a medically validated diagnosis or threshold.
6. **No clinical validation.** Neither the continuous predictions nor binary classifications are established as reliable tools for clinical decision-making, and no external cohort validation or calibration analysis was performed.
7. **Model selection uncertainty.** The single validation split selected the polynomial model, while five-fold cross-validation favored kNN. The disagreement is itself evidence that rankings depend on the evaluation procedure.

## 10. Reproducibility and navigation

1. Create and activate a Python virtual environment.
2. Install dependencies using `pip install -r requirements.txt`.
3. Open [`../PracticalLab2.ipynb`](../PracticalLab2.ipynb) in VS Code or Jupyter and select the project's environment as the kernel.
4. Run the notebook from a fresh kernel, with the `src/` package available alongside it.

The notebook contains the actual training code, plots, metric tables, and complete discussion. This document is a companion technical explanation, not a replacement for the executable notebook.

## 11. Overall conclusion

On the assignment's required validation split, **multivariate polynomial regression degree 2** was the best-performing regression configuration and achieved test R² **0.337**. The supplemental five-fold analysis favored **kNN with 15 neighbors** in mean R² and stability. Logistic regression, evaluated separately as a binary classification exercise, achieved test accuracy **0.746**. Together, these experiments illustrate the tradeoffs among model flexibility, overfitting, generalization, and evaluation design.
