from collections.abc import Sequence

import pandas as pd

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