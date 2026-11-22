from collections.abc import Sequence

import numpy as np
import pandas as pd

def standardize_data(
    data: pd.DataFrame,
    indicator_codes: Sequence[str],
) -> pd.DataFrame:
  """
  Standardize selected indicators using sample Z-scores.
  """

  indicators = list(indicator_codes)

  if data.empty:
    raise ValueError(
      "The dataset is empty."
    )

  if not indicators:
    raise ValueError(
      "At least one indicator is required."
    )

  if len(set(indicators)) != len(indicators):
    raise ValueError(
      "Indicator codes must be unique."
    )

  if len(data) < 2:
    raise ValueError(
      "At least two observations are required."
    )

  missing_columns = set(indicators) - set(data.columns)

  if missing_columns:
    raise ValueError(
      f"Missing indicator columns: {sorted(missing_columns)}"
    )

  non_numeric = [
    column
    for column in indicators
    if not pd.api.types.is_numeric_dtype(data[column])
  ]

  if non_numeric:
    raise ValueError(
      f"Indicators must be numeric: {non_numeric}"
    )

  values = data.loc[:, indicators].to_numpy(dtype=float)

  if not np.isfinite(values).all():
    raise ValueError(
      "Indicator values must be finite and contain no missing values."
    )

  means = np.mean(values, axis=0)
  standard_deviations = np.std(values, axis=0, ddof=1)

  if np.any(standard_deviations == 0):
    raise ValueError(
      "Constant indicators cannot be standardized."
    )

  standardize_values = (values - means) / standard_deviations

  return pd.DataFrame(
    standardize_values,
    index=data.index,
    columns=indicators,
  )

def calculate_covariance_matrix(
    standardized_data: pd.DataFrame,
) -> pd.DataFrame:
  """
  Calculate the sample covariance matrix.
  """

  if standardized_data.empty:
    raise ValueError(
      "The dataset cannot be empty."
    )

  if len(standardized_data) < 2:
    raise ValueError(
      "At least two observations are required."
    )

  if standardized_data.shape[1] == 0:
    raise ValueError(
      "At least one indicator is required."
    )

  for column in standardized_data.columns:
    if not pd.api.types.is_numeric_dtype(standardized_data[column]):
      raise ValueError(
        f"Indicator must be numeric: {column}"
      )

  values = standardized_data.to_numpy(dtype=float)

  if not np.isfinite(values).all():
    raise ValueError(
      "Data must contain only finit values."
    )

  n_observations = values.shape[0]

  covariance_values = (values.T @ values) / (n_observations - 1)

  return pd.DataFrame(
    covariance_values,
    index=standardized_data.columns,
    columns=standardized_data.columns,
  )

def calculate_eigenvalues_eigenvectors(
    covariance_matrix: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
  """
  Calculate and sort eigenvalues and eigenvectors.
  """

  if covariance_matrix.empty:
    raise ValueError(
      "The covariance matrix cannot be empty."
    )

  if covariance_matrix.shape[0] != covariance_matrix.shape[1]:
    raise ValueError(
      "The covariance matrix must be square."
    )

  if not covariance_matrix.index.equals(covariance_matrix.columns):
    raise ValueError(
      "Row and column labels must match."
    )

  for column in covariance_matrix.columns:
    if not pd.api.types.is_numeric_dtype(covariance_matrix[column]):
      raise ValueError(
        f"Column must be numeric: {column}"
      )

  values = covariance_matrix.to_numpy(dtype=float)

  if not np.isfinite(values).all():
    raise ValueError(
      "The covariance matrix must contain finite values."
    )

  if not np.allclose(values, values.T, atol=1e-10):
    raise ValueError(
      "Thw covariance matrix must be symmetric."
    )
  
  eigenvalues, eigenvectors = np.linalg.eigh(values)

  sorted_indices = np.argsort(eigenvalues)[::-1]

  eigenvalues = eigenvalues[sorted_indices]
  eigenvectors = eigenvectors[:, sorted_indices]

  component_names = [
    f"PC{i + 1}"
    for i in range(len(eigenvalues))
  ]

  eigenvalues_series = pd.Series(
    eigenvalues,
    index=component_names,
    name="eigenvalue",
  )

  eigenvectors_dataframe = pd.DataFrame(
    eigenvectors,
    index=covariance_matrix.index,
    columns=component_names
  )

  return eigenvalues_series, eigenvectors_dataframe

def calculate_explained_variance(
    eigenvalues: pd.Series,
) -> pd.DataFrame:
  """
  Calculate explained and cumulative explained variance.
  """

  if eigenvalues.empty:
    raise ValueError(
      "Eigenvalues cannot be empty."
    )

  if not pd.api.types.is_numeric_dtype(eigenvalues):
    raise ValueError(
      "Eigenvalues must be numeric."
    )

  values = eigenvalues.to_numpy(dtype=float)

  if not np.isfinite(values).all():
    raise ValueError(
      "Eigenvalues must be finite"
    )

  if np.any(values < -1e-10):
    raise ValueError(
      "Eigenvalues cannot be negative."
    )

  total_variance = np.sum(values)

  if total_variance <= 0:
    raise ValueError(
      "Total variance must be positive."
    )

  explained_variance_ratio = values / total_variance

  cumulative_explained_variance = np.cumsum(
    explained_variance_ratio
  )

  return pd.DataFrame(
    {
      "eigenvalue": values,
      "explained_variance_ratio": explained_variance_ratio,
      "cumulative_explained_variance": (
        cumulative_explained_variance
      ),
    },
    index=eigenvalues.index,
  )

def calculate_component_loadings(
    eigenvalues: pd.Series,
    eigenvectors: pd.DataFrame,
) -> pd.DataFrame:
  """
  Calculate correlations between indicators and principal components.
  """

  if eigenvalues.empty:
    raise ValueError(
      "Eigenvalues cannot be empty."
    )

  if eigenvectors.empty:
    raise ValueError(
      "Eigenvectors cannot be empty."
    )

  if not pd.api.types.is_numeric_dtype(eigenvalues):
    raise ValueError(
      "Eigenvalues must be numeric."
    )

  if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in eigenvectors.dtypes):
    raise ValueError(
      "Eigenvectors must be numeric."
    )

  if eigenvalues.index.has_duplicates:
    raise ValueError(
      "Eigenvalue component labels must be unique."
    )

  if eigenvectors.index.has_duplicates:
    raise ValueError(
      "Eigenvector indicator labels must be unique."
    )

  if eigenvectors.columns.has_duplicates:
    raise ValueError(
      "Eigenvector component labels must be unique."
    )

  if not eigenvectors.columns.equals(eigenvalues.index):
    raise ValueError(
      "Eigenvalue and eigenvector componentlabel must match."
    )

  values = eigenvalues.to_numpy(dtype=float)
  vectors = eigenvectors.to_numpy(dtype=float)

  if not np.isfinite(values).all() or not np.isfinite(vectors).all():
    raise ValueError(
      "Eigenvalues and eigenvectors must be finite."
    )

  if np.any(values < -1e-10):
    raise ValueError(
      "Eigenvalues cannot be negative."
    )

  loadings = eigenvectors.mul(np.sqrt(np.maximum(values, 0)), axis="columns")

  return loadings

def calculate_component_scores(
    standardized_data: pd.DataFrame,
    eigenvectors: pd.DataFrame,
) -> pd.DataFrame:
  """Calculate principal component scores for each observation."""

  if standardized_data.empty:
    raise ValueError(
      "Standardized data cannot be empty."
    )

  if eigenvectors.empty:
    raise ValueError(
      "Eigenvectors cannot be empty."
    )

  if not standardized_data.columns.equals(eigenvectors.index):
    raise ValueError(
      "Data columns must match eigenvector indicator labels."
    )

  if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in standardized_data.dtypes):
    raise ValueError(
      "Standardized data must be numeric."
    )

  if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in eigenvectors.dtypes):
    raise ValueError(
      "Eigenvectors must be numeric."
    )

  data_values = standardized_data.to_numpy(dtype=float)
  vector_values = eigenvectors.to_numpy(dtype=float)

  if not np.isfinite(data_values).all():
    raise ValueError(
      "Standardized data must be finite."
    )

  if not np.isfinite(vector_values).all():
    raise ValueError(
      "Eigenvectors must be finite."
    )

  scores = data_values @ vector_values

  return pd.DataFrame(
    scores,
    index=standardized_data.index,
    columns=eigenvectors.columns,
  )
