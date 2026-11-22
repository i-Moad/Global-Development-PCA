import pandas as pd
import pytest

from global_pca.data.preparation import prepare_country_dataset

INDICATORS = ["GDP", "LIFE", "INTERNET"]

@pytest.fixture
def sample_data():
  records = []

  values = {
    "AAA": {
      "GDP": [100, 110, 120, 130, 140],
      "LIFE": [70, 71, 72, 73, 74],
      "INTERNET": [40, 45, 50, 55, 60],
    },
    "BBB": {
      "GDP": [200, 210, 220, 230, 240],
      "LIFE": [60, 61, 62, 63, 64],
      "INTERNET": [20, 25, 30, 35, 40],
    },
  }

  for country_code, indicators in values.items():
    for indicator, yearly_values in indicators.items():
      for year, value in enumerate(yearly_values, start=2019):
        records.append({
          "country_code": country_code,
          "country": country_code,
          "indicator_code": indicator,
          "year": year,
          "value": value,
        })

  return pd.DataFrame(records)

def test_prepare_country_dataset_returns_ecpected_shape(sample_data):
  result = prepare_country_dataset(sample_data, INDICATORS)

  assert result.shape == (2, 5)

def test_prepare_country_dataset_calculates_means(sample_data):
  result = prepare_country_dataset(sample_data, INDICATORS)

  aaa = result.loc[
    result["country_code"] == "AAA"
  ].iloc[0]

  assert aaa["GDP"] == 120
  assert aaa["LIFE"] == 72
  assert aaa["INTERNET"] == 50

def test_prepare_country_dataset_excludes_insufficient_coverage(sample_data):
    sample_data = sample_data[
        ~(
            (sample_data["country_code"] == "BBB")
            & (sample_data["indicator_code"] == "INTERNET")
            & (sample_data["year"] >= 2021)
        )
    ]

    result = prepare_country_dataset(
        sample_data,
        INDICATORS,
        min_years=4,
    )

    assert result["country_code"].tolist() == ["AAA"]


def test_prepare_country_dataset_rejects_empty_data():
    with pytest.raises(ValueError, match="empty"):
        prepare_country_dataset(
            pd.DataFrame(),
            INDICATORS,
        )


def test_prepare_country_dataset_rejects_missing_indicators(sample_data):
    sample_data = sample_data[
        sample_data["indicator_code"] != "INTERNET"
    ]

    with pytest.raises(ValueError, match="No observations"):
        prepare_country_dataset(sample_data, INDICATORS)


def test_prepare_country_dataset_rejects_duplicate_observations(sample_data):
    duplicate = sample_data.iloc[[0]]
    sample_data = pd.concat(
        [sample_data, duplicate],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="Duplicate"):
        prepare_country_dataset(sample_data, INDICATORS)


def test_prepare_country_dataset_rejects_invalid_min_years(sample_data):
    with pytest.raises(ValueError, match="min_years"):
        prepare_country_dataset(
            sample_data,
            INDICATORS,
            min_years=6,
        )


def test_prepare_country_dataset_rejects_duplicate_indicator_codes(sample_data):
    with pytest.raises(ValueError, match="unique"):
        prepare_country_dataset(
            sample_data,
            ["GDP", "GDP"],
        )