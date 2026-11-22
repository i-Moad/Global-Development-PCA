from collections.abc import Sequence
from math import ceil

import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.figure import Figure

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