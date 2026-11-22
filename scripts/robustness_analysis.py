import pandas as pd

from global_pca.data.pipeline import INDICATOR_CODES
from global_pca.pca import (
  standardize_data,
  calculate_covariance_matrix,
  calculate_eigenvalues_eigenvectors,
  calculate_explained_variance,
  calculate_component_loadings,
)

def run_pca(data: pd.DataFrame):
  standardized_data = standardize_data(
    data,
    INDICATOR_CODES,
  )

  covariance_matrix = calculate_covariance_matrix(
    standardized_data,
  )

  eigenvalues, eigenvectors = calculate_eigenvalues_eigenvectors(
    covariance_matrix,
  )
  
  pc1_values = eigenvectors["PC1"].to_numpy(dtype="float64")

  internet_position = eigenvectors.index.get_loc("IT.NET.USER.ZS")

  if pc1_values[internet_position] < 0:
    eigenvectors["PC1"] *= -1

  explained_variance = calculate_explained_variance(
    eigenvalues,
  )

  loadings = calculate_component_loadings(
    eigenvalues,
    eigenvectors,
  )

  return explained_variance, loadings

def main():
  primary_data = pd.read_csv(
    "data/processed/country_development.csv"
  )

  complete_data = pd.read_csv(
    "data/raw/world_bank_2019_2023.csv"
  )

  valid_observations = complete_data.dropna(
    subset=["value"]
  )

  # Count valid years for each economy and indicator
  year_counts = (
    valid_observations
    .groupby(["country_code", "indicator_code"])["year"]
    .nunique()
    .unstack(fill_value=0)
  )

  # Retain economies with five valid years for every indicator
  complete_country_codes = year_counts.index[
    (
      year_counts
      .reindex(columns=list(INDICATOR_CODES), fill_value=0)
      == 5
    ).all(axis=1)
  ]

  complete_data = valid_observations[
    valid_observations["country_code"].isin(
      complete_country_codes
    )
  ]

  # Calculate five-year averages
  complete_data = (
    complete_data
    .pivot_table(
      index=["country_code", "country"],
      columns="indicator_code",
      values="value",
      aggfunc="mean",
    )
    .reset_index()
  )

  complete_data.columns.name = None

  print("DATASET SIZES")
  print(f"Primary dataset: {len(primary_data)} economies")
  print(f"Complete-case dataset: {len(complete_data)} economies")

  primary_variance, primary_loadings = run_pca(primary_data)
  complete_variance, complete_loadings = run_pca(complete_data)

  print("\nEXPLAINED VARIANCE COMPARISON")

  comparison = pd.DataFrame({
    "Primary dataset": primary_variance[
      "explained_variance_ratio"
    ],
    "Complete-case dataset": complete_variance[
      "explained_variance_ratio"
    ],
  })

  print((comparison * 100).round(2).to_string())

  print("\nPC1 LOADINGS COMPARISON")

  loading_comparison = pd.DataFrame({
    "Primary dataset": primary_loadings["PC1"],
    "Complete-case dataset": complete_loadings["PC1"],
  })

  print(loading_comparison.round(3).to_string())


if __name__ == "__main__":
  main()