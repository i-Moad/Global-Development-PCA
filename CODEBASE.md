# Codebase Guide

This document explains the project structure, the purpose of its main modules, and how the different parts work together.

## 1. Project Structure

```text
global-development-pca/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── reports/
│   └── figures/
│
├── scripts/
│   ├── plot_clusters.py
│   └── robustness_analysis.py
│
├── src/
│   └── global_pca/
│       ├── data/
│       │   ├── acquisition.py
│       │   ├── preparation.py
│       │   └── pipeline.py
│       │
│       ├── analysis.py
│       ├── pca.py
│       └── clustering.py
│
├── tests/
├── requirements.txt
└── README.md
```

## 2. Data Module

Located in `src/global_pca/data/`.

### `acquisition.py`

Handles data collection from the World Bank API.

Main responsibilities:
- Retrieve available country and economy information.
- Fetch the selected socioeconomic indicators.
- Handle API pagination and response validation.
- Organize the retrieved observations into a Pandas DataFrame.

### `preparation.py`

Transforms the raw observations into an analytical dataset.

Main responsibilities:
- Validate the input data.
- Handle missing observations.
- Check for duplicate records and inconsistent country information.
- Calculate average indicator values for each economy.
- Apply the minimum data coverage requirement.

### `pipeline.py`

Connects the acquisition and preparation stages.

It defines the selected indicators and analysis period, executes the data pipeline, and saves the resulting datasets.

## 3. Analysis Module

### `analysis.py`

Contains functions for exploratory data analysis and visualization.

It includes:
- Descriptive statistics.
- Indicator distribution plots.
- Correlation matrix calculations and visualization.
- Outlier detection using the IQR method.
- PCA cluster visualization.

The purpose of this module is to understand the dataset before applying dimensionality reduction and clustering.

## 4. PCA Implementation

### `pca.py`

This is the main learning component of the project.

Instead of relying entirely on Scikit-learn, the mathematical steps of PCA were implemented using NumPy and Pandas.

The module includes:

| Function | Purpose |
|---|---|
| `standardize_data()` | Standardizes the input variables. |
| `calculate_covariance_matrix()` | Calculates the variance-covariance matrix. |
| `calculate_eigenvalues_eigenvectors()` | Performs eigendecomposition and orders the components. |
| `calculate_explained_variance()` | Calculates explained and cumulative variance. |
| `calculate_component_loadings()` | Calculates component loadings. |
| `calculate_component_scores()` | Projects observations into the new component space. |

The implementation was compared against Scikit-learn's PCA to verify numerical consistency.

## 5. Clustering Module

### `clustering.py`

Uses Scikit-learn's K-means implementation to group economies based on their principal component scores.

It contains two main functions:

- `evaluate_cluster_counts()` – Evaluates different numbers of clusters using inertia and silhouette scores.
- `fit_kmeans()` – Trains the selected K-means model and returns the cluster assignments.

## 6. Scripts

Located in `scripts/`.

### `plot_clusters.py`

Runs the PCA and clustering workflow to generate the cluster visualization saved in `reports/figures/`.

### `robustness_analysis.py`

Compares PCA results obtained from:
- The primary dataset containing 115 economies.
- A complete-case dataset containing 94 economies.

It compares explained variance and component loadings to examine whether the main findings remain similar under a stricter data inclusion criterion.

## 7. Data and Reports

### `data/raw/`

Contains the original observations retrieved from the World Bank API.

### `data/processed/`

Contains the cleaned and aggregated dataset used for the analysis.

### `reports/figures/`

Contains generated visualizations, including the K-means clustering plot.

## 8. Tests

The `tests/` directory contains automated tests written with Pytest.

They cover the main project functionality, including:
- Data acquisition and preparation.
- Exploratory analysis.
- PCA calculations and input validation.
- Clustering and cluster evaluation.

The tests help ensure that the individual modules behave as expected.

Run the test suite with:

```powershell
$env:PYTHONPATH = "src"
python -m pytest -v
```

---

The project separates data acquisition, preprocessing, analysis, dimensionality reduction, and clustering into individual modules. This keeps the code organized, easier to test, and easier to extend.