# Practical Lab 2 — Machine Learning Models and Cross-Validation

A machine learning project using the **scikit-learn Diabetes dataset** to explore the prediction of disease progression one year after baseline.

The project implements univariate and multivariate polynomial regression, decision trees, k-nearest neighbors, and logistic regression. It includes exploratory data analysis, model evaluation, train/validation/test splitting, and five-fold cross-validation.

For a detailed explanation of the methodology, model configurations, results, and limitations, see the **[Project Documentation](documents/PROJECT_DOCUMENTATION.md)**.

## Repository Structure

```text
.
├── PracticalLab2.ipynb           # Main notebook
├── documents/
│   └── PROJECT_DOCUMENTATION.md  # Detailed project documentation
├── src/                           # Object-oriented Python modules
├── requirements.txt               # Project dependencies
├── .gitignore
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10 or newer
- Git
- Visual Studio Code with the **Python** and **Jupyter** extensions

### 1. Clone the Repository

Open PowerShell and run:

```powershell
git clone https://github.com/JohnBuni/mvlregression_npmodels_crossvalidation.git
cd mvlregression_npmodels_crossvalidation
```

### 2. Create a Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Run the Notebook

1. Open the project folder in VS Code.
2. Open `PracticalLab2.ipynb`.
3. Click **Select Kernel** and choose the Python environment from `.venv`.
4. Select **Restart Kernel → Run All** to execute the notebook.

The Diabetes dataset is included with scikit-learn, so no separate dataset download is required.

## Documentation

The notebook contains the executable experiments, visualizations, and analysis.

For complete technical details, including dataset exploration, model development, evaluation metrics, cross-validation, results, and conclusions, refer to:

**[Detailed Project Documentation](documents/PROJECT_DOCUMENTATION.md)**

---
