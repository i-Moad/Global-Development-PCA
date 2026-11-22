import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest


from global_pca.analysis import calculate_descriptive_statistics
from global_pca.analysis import plot_indicator_distributions
from global_pca.analysis import calculate_correlation_matrix
from global_pca.analysis import plot_correlation_matrix
from global_pca.analysis import detect_outliers


def test_calculate_descriptive_statistics():
  data = pd.DataFrame({
    "GDP": [100, 200, 300],
    "Life": [60, 70, 80],
  })

  result = calculate_descriptive_statistics(
    data,
    ["GDP", "Life"],
  )

  assert result.shape == (2, 8)
  assert result.loc["GDP", "mean"] == 200
  assert result.loc["GDP", "median"] == 200
  assert result.loc["Life", "mean"] == 70


def test_reject_empty_dataset():
  with pytest.raises(ValueError):
    calculate_descriptive_statistics(
      pd.DataFrame(),
      ["GDP"],
    )


def test_reject_missing_indicator():
  data = pd.DataFrame({"GDP": [100, 200]})

  with pytest.raises(ValueError):
    calculate_descriptive_statistics(
      data,
      ["GDP", "Life"],
    )


def test_reject_non_numeric_indicator():
  data = pd.DataFrame({"GDP": ["100", "200"]})

  with pytest.raises(TypeError):
    calculate_descriptive_statistics(
      data,
      ["GDP"],
    )


def test_reject_duplicate_indicators():
  data = pd.DataFrame({"GDP": [100, 200]})

  with pytest.raises(ValueError):
    calculate_descriptive_statistics(
      data,
      ["GDP", "GDP"],
    )

def test_plot_indicator_distributions():
  data = pd.DataFrame({
    "GDP": [100, 200, 300],
    "Life": [60, 70, 80],
  })

  figure = plot_indicator_distributions(
    data,
    ["GDP", "Life"],
  )

  assert len(figure.axes) == 2
  assert figure.axes[0].get_title() == "GDP"
  assert figure.axes[1].get_title() == "Life"

  plt.close(figure)


def test_reject_invalid_histogram_bins():
  data = pd.DataFrame({"GDP": [100, 200]})

  with pytest.raises(ValueError):
    plot_indicator_distributions(data, ["GDP"], bins=0)

  with pytest.raises(TypeError):
    plot_indicator_distributions(data, ["GDP"], bins=2.5)


def test_reject_missing_values_in_histogram():
  data = pd.DataFrame({"GDP": [100, None, 300]})

  with pytest.raises(ValueError):
    plot_indicator_distributions(data, ["GDP"])

def test_calculate_correlation_matrix():
  data = pd.DataFrame({
    "GDP": [100, 200, 300, 400],
    "Life": [50, 60, 70, 80],
    "CO2": [400, 300, 200, 100],
  })

  result = calculate_correlation_matrix(
    data,
    ["GDP", "Life", "CO2"],
  )

  assert result.shape == (3, 3)
  assert result.loc["GDP", "GDP"] == pytest.approx(1.0)
  assert result.loc["GDP", "Life"] == pytest.approx(1.0)
  assert result.loc["GDP", "CO2"] == pytest.approx(-1.0)


def test_reject_constant_indicator_in_correlation():
  data = pd.DataFrame({
    "GDP": [100, 200, 300],
    "Life": [70, 70, 70],
  })

  with pytest.raises(ValueError):
    calculate_correlation_matrix(data, ["GDP", "Life"])


def test_reject_missing_values_in_correlation():
  data = pd.DataFrame({
    "GDP": [100, None, 300],
    "Life": [60, 70, 80],
  })

  with pytest.raises(ValueError):
    calculate_correlation_matrix(data, ["GDP", "Life"])

def test_plot_correlation_matrix():
  data = pd.DataFrame({
    "GDP": [100, 200, 300],
    "Life": [50, 60, 70],
  })

  correlation = calculate_correlation_matrix(
    data,
    ["GDP", "Life"],
  )

  figure = plot_correlation_matrix(correlation)

  assert len(figure.axes) == 2
  assert figure.axes[0].get_title() == (
    "Correlation Matrix of Development Indicators"
  )

  plt.close(figure)


def test_reject_non_square_correlation_matrix():
  correlation = pd.DataFrame({
    "GDP": [1.0, 0.5],
    "Life": [0.5, 1.0],
    "CO2": [0.3, 0.4],
  })

  with pytest.raises(ValueError):
    plot_correlation_matrix(correlation)

def test_detect_outliers():
  data = pd.DataFrame({
    "country_code": ["A", "B", "C", "D", "E"],
    "country": ["A", "B", "C", "D", "E"],
    "GDP": [10, 11, 12, 13, 100],
  })

  result = detect_outliers(data, ["GDP"])

  assert len(result) == 1
  assert result.iloc[0]["country_code"] == "E"
  assert result.iloc[0]["value"] == 100


def test_detect_outliers_returns_empty_dataframe():
  data = pd.DataFrame({
    "country_code": ["A", "B", "C", "D"],
    "country": ["A", "B", "C", "D"],
    "GDP": [10, 11, 12, 13],
  })

  result = detect_outliers(data, ["GDP"])

  assert result.empty
  assert "indicator" in result.columns


def test_reject_missing_values_in_outlier_detection():
  data = pd.DataFrame({
    "country_code": ["A", "B", "C"],
    "country": ["A", "B", "C"],
    "GDP": [10, None, 30],
  })

  with pytest.raises(ValueError):
    detect_outliers(data, ["GDP"])