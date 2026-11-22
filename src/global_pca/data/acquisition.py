from typing import Sequence

import pandas as pd
import requests

BASE_URL = "https://api.worldbank.org/v2"

def fetch_country_code() -> set[str]:
  """
    Retrieve ISO-2 identifiers for individual countries.
  """

  url = f"{BASE_URL}/country"

  params = {
    "format": "json",
    "per_page": 400,
  }

  response = requests.get(
    url=url,
    params=params,
    timeout=30
  )
  response.raise_for_status()

  payload = response.json()

  if not isinstance(payload, list) or len(payload) < 2:
    raise RuntimeError("Unexpected country metadata response.")

  countries = payload[1]

  return {
    country["iso2Code"]
    for country in countries
    if country["region"]["id"] != "NA"
  }

def fetch_world_bank_data(
    indicator_codes: Sequence[str],
    start_year: int,
    end_year: int,
    per_page: int = 1000,
) -> pd.DataFrame:
  """
    Retrieve World Bank indicator observations.

    Returns one row per country, indicator, and year.
    Missing observations are preserved as NaN.
  """

  if not indicator_codes:
    raise ValueError("At least one indicator code is required.")

  if start_year > end_year:
    raise ValueError("start_year must not exceed end_year.")

  if per_page < 1:
    raise ValueError("per_page must be postive.")

  valid_country_codes = fetch_country_code()
  observations = []

  for indicator_code in indicator_codes:
    page = 1
    totta_pages = 1

    while page <= totta_pages:
      url = (
        f"{BASE_URL}/country/all/"
        f"indicator/{indicator_code}"
      )

      params = {
        "format": "json",
        "date": f"{start_year}:{end_year}",
        "per_page": per_page,
        "page": page,
        "source": 2,
      }

      response = requests.get(
        url=url,
        params=params,
        timeout=30,
      )
      response.raise_for_status()

      payload = response.json()

      if not isinstance(payload, list):
        raise RuntimeError(
          f"Invalid API response for {indicator_code}: {payload}"
        )

      if len(payload) < 2:
        raise RuntimeError(
          f"World Bank API error for {indicator_code}: {payload}"
        )

      metadata, records = payload

      totta_pages = int(metadata["pages"])

      if records is None:
        records = []

      for record in records:
        country = record["country"]

        if country["id"] not in valid_country_codes:
          continue

        observations.append({
          "country_code": record["countryiso3code"],
          "country": record["country"]["value"],
          "indicator_code": record["indicator"]["id"],
          "indicator": record["indicator"]["value"],
          "year": int(record["date"]),
          "value": record["value"],
        })

      page += 1

  columns = [
    "country_code",
    "country",
    "indicator_code",
    "indicator",
    "year",
    "value",
  ]

  return pd.DataFrame(observations, columns=columns)