import matplotlib

matplotlib.use("Agg")

import pytest
import pandas as pd
import matplotlib.pyplot as plt


from global_pca.analysis import calculate_descriptive_statistics
from global_pca.analysis import plot_indicator_distributions
from global_pca.analysis import calculate_correlation_matrix
from global_pca.analysis import plot_correlation_matrix
from global_pca.analysis import detect_outliers
from global_pca.analysis import plot_cluster_scatter


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


def test_plot_cluster_scatter():
  scores = pd.DataFrame(
    {
      "PC1": [1.0, 3.0, -1.0, -3.0],
      "PC2": [2.0, 4.0, -2.0, -4.0],
    },
    index=["A", "B", "C", "D"],
  )

  assignments = pd.Series(
    [0, 0, 1, 1],
    index=scores.index,
  )

  fig, ax = plot_cluster_scatter(scores, assignments)

  assert ax.get_xlabel() == "PC1"
  assert ax.get_ylabel() == "PC2"
  assert ax.get_title() == "K-Means Clusters in PCA Space"

  # Two collections per cluster: points and centroid
  assert len(ax.collections) == 4

  plt.close(fig)


def test_plot_cluster_scatter_centroids():
  scores = pd.DataFrame(
    {
      "PC1": [1.0, 3.0, -1.0, -3.0],
      "PC2": [2.0, 4.0, -2.0, -4.0],
    },
    index=["A", "B", "C", "D"],
  )

  assignments = pd.Series(
    [0, 0, 1, 1],
    index=scores.index,
  )

  fig, ax = plot_cluster_scatter(scores, assignments)

  # Cluster 0 centroid should be (2, 3)
  centroid_0 = ax.collections[1].get_offsets()[0]

  assert centroid_0[0] == pytest.approx(2.0)
  assert centroid_0[1] == pytest.approx(3.0)

  # Cluster 1 centroid should be (-2, -3)
  centroid_1 = ax.collections[3].get_offsets()[0]

  assert centroid_1[0] == pytest.approx(-2.0)
  assert centroid_1[1] == pytest.approx(-3.0)

  plt.close(fig)


def test_plot_cluster_scatter_rejects_empty_scores():
  scores = pd.DataFrame(columns=["PC1", "PC2"])
  assignments = pd.Series(dtype=int)

  with pytest.raises(ValueError, match="empty"):
    plot_cluster_scatter(scores, assignments)


def test_plot_cluster_scatter_rejects_missing_components():
  scores = pd.DataFrame({"PC1": [1.0, 2.0, 3.0]})
  assignments = pd.Series([0, 0, 1])

  with pytest.raises(ValueError, match="Missing component"):
    plot_cluster_scatter(scores, assignments)


def test_plot_cluster_scatter_rejects_mismatched_indexes():
  scores = pd.DataFrame(
    {
      "PC1": [1.0, 2.0, 3.0],
      "PC2": [4.0, 5.0, 6.0],
    },
    index=["A", "B", "C"],
  )

  assignments = pd.Series(
    [0, 0, 1],
    index=["X", "Y", "Z"],
  )

  with pytest.raises(ValueError, match="identical indexes"):
    plot_cluster_scatter(scores, assignments)


def test_plot_cluster_scatter_rejects_non_finite_scores():
  scores = pd.DataFrame(
    {
      "PC1": [1.0, float("nan"), 3.0],
      "PC2": [4.0, 5.0, 6.0],
    }
  )

  assignments = pd.Series([0, 0, 1])

  with pytest.raises(ValueError, match="finite values"):
    plot_cluster_scatter(scores, assignments)