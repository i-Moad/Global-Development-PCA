from collections.abc import Sequence
from math import ceil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from global_pca.data.pipeline import START_YEAR, END_YEAR

def calculate_descriptive_statistics(
    data: pd.DataFrame,
    indicator_codes: Sequence[str],
) -> pd.DataFrame:
  """
  Calculate descriptive statistics for selected indicators.
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
      "Indicator code must be unique."
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
    raise TypeError(
      f"Indicators must be numeric: {non_numeric}"
    )

  statistics = data[indicators].describe().T
  statistics = statistics.rename(columns={"50%": "median"})

  return statistics

def plot_indicator_distributions(
  data: pd.DataFrame,
  indicator_codes: Sequence[str],
  bins: int = 20
) -> Figure:
  """
  Plot histograms showing the distributions of selected indicators.
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

  if not isinstance(bins, int) or isinstance(bins, bool):
    raise TypeError(
      "bins must be an integer."
    )

  if bins < 1:
    raise ValueError(
      "bins must be positive."
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
    raise TypeError(
      f"Indicators must be numeric: {non_numeric}"
    )

  if data[indicators].isna().any().any():
    raise ValueError(
      "Indicator data contains missing values."
    )

  columns = min(3, len(indicators))
  rows = ceil(len(indicators) / columns)

  figure, axes = plt.subplots(
    nrows=rows,
    ncols=columns,
    figsize=(15, 5 * rows),
    squeeze=False,
  )

  for axis, indicator in zip(axes.flat, indicators):
    axis.hist(data[indicator], bins=bins, edgecolor="black")
    axis.set_title(indicator)
    axis.set_xlabel(f"Average indicator value ({START_YEAR}-{END_YEAR})")
    axis.set_ylabel("Number of economies")

  for axis in list(axes.flat)[len(indicators):]:
    axis.set_visible(False)

  figure.tight_layout()

  return figure

def calculate_correlation_matrix(
    data: pd.DataFrame,
    indicator_codes: Sequence[str],
) -> pd.DataFrame:
  """
  Calculate Pearson correlations between selected indicators.
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
    raise TypeError(
      f"Indicators must be numeric: {non_numeric}"
    )

  if data[indicators].isna().any().any():
    raise ValueError(
      "Indicator data contains missing values."
    )

  if (data[indicators].nunique() < 2).any():
    raise ValueError(
      "Correlation requires variation in every indicator."
    )

  return data[indicators].corr(method="pearson")

def plot_correlation_matrix(
    correlation_matrix: pd.DataFrame,
) -> Figure:
  """
  Visualize a correlation matrix as a heatmap.
  """

  if correlation_matrix.empty:
    raise ValueError(
      "The correlation matrix is empty."
    )

  if correlation_matrix.shape[0] != correlation_matrix.shape[1]:
    raise ValueError(
      "Thw correlation matrix must be square."
    )

  if correlation_matrix.isna().any().any():
    raise ValueError(
      "The correlation matrix contains missing values."
    )

  if not all(
    pd.api.types.is_numeric_dtype(correlation_matrix[column])
    for column in correlation_matrix.columns
  ):
    raise TypeError(
      "Correlation values must be numeric."
    )

  if not correlation_matrix.index.equals(correlation_matrix.columns):
    raise ValueError(
      "Row and column labels must match."
    )

  values = correlation_matrix.to_numpy()

  if not ((values >= -1).all() and (values <= 1).all()):
    raise ValueError(
      "Correlation values must be between -1 and 1."
    )

  if not correlation_matrix.equals(correlation_matrix.T):
    raise ValueError(
      "The correlation matrix must be symmetric."
    )

  figure, axis = plt.subplots(figsize=(10, 8))

  image = axis.imshow(
    values,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
  )

  labels = correlation_matrix.columns

  axis.set_xticks(range(len(labels)))
  axis.set_yticks(range(len(labels)))

  axis.set_xticklabels(labels, rotation=45, ha="right")
  axis.set_yticklabels(labels)

  for row in range(len(labels)):
    for column in range(len(labels)):
      axis.text(
        column,
        row,
        f"{values[row, column]:.2f}",
        ha="center",
        va="center",
        color="black",
      )

  figure.colorbar(image, ax=axis, label="Pearson correaltion")
  axis.set_title("Correlation Matrix of Development Indicators")
  figure.tight_layout()

  return figure

def detect_outliers(
    data: pd.DataFrame,
    indicator_codes: Sequence[str],
) -> pd.DataFrame:
  """
    Identify potential outliers using the IQR method.
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
    raise TypeError(
      f"Indicators must be numeric: {non_numeric}"
    )

  if data[indicators].isna().any().any():
    raise ValueError(
      "Indicator data contains missing values."
    )

  outliers = []

  for indicator in indicators:
    q1 = data[indicator].quantile(0.25)
    q3 = data[indicator].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    mask = (
      (data[indicator] < lower_bound)
      | (data[indicator] > upper_bound)
    )

    for _, row in data.loc[mask].iterrows():
      outliers.append({
        "country_code": row["country_code"],
        "country": row["country"],
        "indicator": indicator,
        "value": row[indicator],
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
      })

  return pd.DataFrame(
    outliers,
    columns=[
      "country_code",
      "country",
      "indicator",
      "value",
      "lower_bound",
      "upper_bound",
    ],
  )

def plot_cluster_scatter(
    scores: pd.DataFrame,
    assignments: pd.Series,
    component_columns: tuple[str, str] = ("PC1", "PC2"),
    cluster_names: dict[int, str] | None = None,
) -> tuple[Figure, Axes]:
  """
  Visualize K-Means clusters and their centroids in PCA space.
  """

  if scores.empty:
    raise ValueError(
      "Scores cannot be empty."
    )

  if len(component_columns) != 2:
    raise ValueError(
      "Exactly two components must be provided."
    )

  if len(set(component_columns)) != 2:
    raise ValueError(
      "Component columns must be unique."
    )

  missing_columns = set(component_columns) - set(scores.columns)

  if missing_columns:
    raise ValueError(
      f"Missing component columns: {sorted(missing_columns)}"
    )

  if not isinstance(assignments, pd.Series):
    raise TypeError(
      "Assignments must be a pandas Series."
    )

  if not scores.index.equals(assignments.index):
    raise ValueError(
      "Scores and assignments must have identical indexes."
    )

  if assignments.isna().any():
    raise ValueError(
      "Assignments cannot contain missing values."
    )

  selected_scores = scores.loc[:, list(component_columns)]

  if not all(
      pd.api.types.is_numeric_dtype(dtype)
      for dtype in selected_scores.dtypes
  ):
    raise TypeError(
      "Component scores must be numeric."
    )

  if not np.isfinite(selected_scores.to_numpy()).all():
    raise ValueError(
      "Component scores must contain finite values."
    )

  fig, ax = plt.subplots(figsize=(10, 7))

  for cluster in sorted(assignments.unique()):
    mask = assignments == cluster
    cluster_scores = selected_scores.loc[mask]

    name = (
      cluster_names.get(cluster, f"Cluster {cluster}")
      if cluster_names is not None
      else f"Cluster {cluster}"
    )

    ax.scatter(
      cluster_scores[component_columns[0]],
      cluster_scores[component_columns[1]],
      label=name,
      alpha=0.75,
      s=55,
    )

    centroid = cluster_scores.mean()

    ax.scatter(
      centroid[component_columns[0]],
      centroid[component_columns[1]],
      marker="X",
      s=180,
      edgecolors="black",
      linewidths=1,
      label=f"{name} centroid",
    )

  ax.set_xlabel(component_columns[0])
  ax.set_ylabel(component_columns[1])
  ax.set_title("K-Means Clusters in PCA Space")

  ax.axhline(0, linewidth=0.7, linestyle="--", alpha=0.5)
  ax.axvline(0, linewidth=0.7, linestyle="--", alpha=0.5)

  ax.legend()
  ax.grid(alpha=0.2)

  fig.tight_layout()

  return fig, ax