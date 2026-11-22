import matplotlib.pyplot as plt
import pandas as pd
import pytest


from global_pca.analysis import calculate_descriptive_statistics
from global_pca.analysis import plot_indicator_distributions


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