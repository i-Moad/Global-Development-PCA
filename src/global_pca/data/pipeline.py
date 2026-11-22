from pathlib import Path

import pandas as pd

from global_pca.data.acquisition import fetch_world_bank_data
from global_pca.data.preparation import prepare_country_dataset

INDICATOR_CODES = (
  "NY.GDP.PCAP.PP.KD",
  "SP.DYN.LE00.IN",
  "IT.NET.USER.ZS",
  "EG.ELC.ACCS.ZS",
  "SE.SEC.ENRR",
  "EN.GHG.CO2.PC.CE.AR5",
)
START_YEAR = 2019
END_YEAR = 2023
MIN_YEARS = 4

def run_data_pipeline(
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    min_years: int = MIN_YEARS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """
  Download, prepare, and save the World Bank dataset.
  """

  project_root = Path(__file__).resolve().parents[3]

  raw_path = (
    project_root
    / "data"
    / "raw"
    / f"world_bank_{start_year}_{end_year}.csv"
  )

  processed_path = (
    project_root
    / "data"
    / "processed"
    / "country_development.csv"
  )

  # Acquire the original observations.
  print("[INFO] Downloading World Bank observations...")

  raw_data = fetch_world_bank_data(
    indicator_codes=INDICATOR_CODES,
    start_year=start_year,
    end_year=end_year,
  )

  print(f"[INFO] Retrieved {len(raw_data)} observations.")

  # Prepare the country-level analytical dataset.
  print("[INFO] Preparing the analytical dataset...")

  analytical_data = prepare_country_dataset(
    raw_data=raw_data,
    indicator_codes=INDICATOR_CODES,
    min_years=min_years,
  )

  # Create output directories when necessary.
  raw_path.parent.mkdir(parents=True, exist_ok=True)
  processed_path.parent.mkdir(parents=True, exist_ok=True)

  # Save both datasets.
  raw_data.to_csv(
    raw_path,
    index=False,
    encoding="utf-8",
    float_format="%.8f",
  )

  analytical_data.to_csv(
    processed_path,
    index=False,
    encoding="utf-8",
    float_format="%.8f",
  )

  print("\nPIPELINE COMPLETED")
  print("-" * 50)
  print(f"Raw observations: {len(raw_data)}")
  print(f"Analytical dataset: {analytical_data.shape}")
  print(f"Raw data saved to: {raw_path}")
  print(f"Processed data saved to: {processed_path}")

  return raw_data, analytical_data

if __name__ == "__main__":
  run_data_pipeline()