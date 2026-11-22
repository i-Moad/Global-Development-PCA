from collections.abc import Sequence

import pandas as pd

REQUIRED_COLUMNS = {
  "country_code",
  "country",
  "indicator_code",
  "year",
  "value",
}

def prepare_country_dataset(
    raw_data: pd.DataFrame,
    indicator_codes: Sequence[str],
    min_years: int = 4,
) -> pd.DataFrame:
  """
  Prepare a country-level dataset for PCA.

  Each indicator is represented by its mean over the available years.
  Only countries meeting the minimum observation requirement for
  every indicator are retained.
  """

  if raw_data.empty:
    raise ValueError("The raw dataset is empty.")

  if not indicator_codes:
    raise ValueError("At least one indicator is required.")

  if len(set(indicator_codes)) != len(indicator_codes):
    raise ValueError("Indicator codes must be unique.")

  if not isinstance(min_years, int) or isinstance(min_years, bool):
    raise TypeError("min_years must be an integer.")

  if not 1 <= min_years <= 5:
    raise ValueError("min_years must be between 1 and 5.")

  missing_columns = REQUIRED_COLUMNS - set(raw_data.columns)

  if missing_columns:
    raise ValueError(
      f"Missing required columns: {sorted(missing_columns)}"
    )

  indicators = list(indicator_codes)

  # Keep only the indicators requested for the analysis.
  data = raw_data[
    raw_data["indicator_code"].isin(indicators)
  ].copy()

  available_indicators = set(data["indicator_code"].unique())
  missing_indicators = set(indicators) - available_indicators

  if missing_indicators:
    raise ValueError(
      f"No observations found for indicators: "
      f"{sorted(missing_indicators)}"
    )

  # Each country indicator year combination must be unique.
  duplicate_mask = data.duplicated(
    subset=["country_code", "indicator_code", "year"]
  )

  if duplicate_mask.any():
    raise ValueError(
      "Duplicate country-indicator-year observations detected."
    )

  # Ensure each country code has a consistent country name.
  name_counts = data.groupby("country_code")["country"].nunique()

  if (name_counts > 1).any():
    raise ValueError(
      "Inconsistent country names detected."
    )

  # Calculate the average value of each indicator per country.
  country_means = data.pivot_table(
    index=["country_code", "country"],
    columns="indicator_code",
    values="value",
    aggfunc="mean",
  ).reindex(columns=indicators)

  # Count available yearly observations for each indicator.
  year_count = data.pivot_table(
    index=["country_code", "country"],
    columns="indicator_code",
    values="value",
    aggfunc="count",
  ).reindex(columns=indicators).fillna(0)

  # Retain countries meeting the coverage requirement for every indicator.
  eligibale_countries = year_count.ge(min_years).all(axis=1)

  analytical_data = country_means.loc[
    eligibale_countries
  ].reset_index()

  analytical_data = analytical_data.sort_values(
    "country_code"
  ).reset_index(drop=True)

  if analytical_data.empty:
    raise ValueError(
      "No countries meet the minimum coverage requirement."
    )

  if analytical_data[indicators].isna().any().any():
    raise RuntimeError(
      "Unexpected missing values in the analytical dataset."
    )

  return analytical_data