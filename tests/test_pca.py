import numpy as np
import pandas as pd
import pytest

from global_pca.pca import standardize_data
from global_pca.pca import calculate_covariance_matrix
from global_pca.pca import calculate_eigenvalues_eigenvectors
from global_pca.pca import calculate_explained_variance
from global_pca.pca import calculate_component_loadings
from global_pca.pca import calculate_component_scores


@pytest.fixture
def sample_data():
  return pd.DataFrame({
    "GDP": [100, 200, 300, 400],
    "Life": [60, 70, 80, 90],
    "Internet": [20, 40, 60, 80],
  })


def test_standardize_data_has_zero_mean(sample_data):
  result = standardize_data(
    sample_data,
    ["GDP", "Life", "Internet"],
  )

  np.testing.assert_allclose(
    result.mean().to_numpy(),
    0,
    atol=1e-12,
  )


def test_standardize_data_has_unit_sample_std(sample_data):
  result = standardize_data(
    sample_data,
    ["GDP", "Life", "Internet"],
  )

  np.testing.assert_allclose(
    result.std(ddof=1).to_numpy(),
    1,
    atol=1e-12,
  )


def test_standardize_data_preserves_shape_and_columns(sample_data):
  result = standardize_data(
    sample_data,
    ["GDP", "Life"],
  )

  assert result.shape == (4, 2)
  assert list(result.columns) == ["GDP", "Life"]


def test_reject_empty_dataset():
  with pytest.raises(ValueError):
    standardize_data(
      pd.DataFrame(),
      ["GDP"],
    )


def test_reject_missing_indicator(sample_data):
  with pytest.raises(ValueError):
    standardize_data(
      sample_data,
      ["GDP", "Missing"],
    )


def test_reject_constant_indicator():
  data = pd.DataFrame({
    "GDP": [100, 100, 100],
  })

  with pytest.raises(ValueError):
    standardize_data(data, ["GDP"])


def test_reject_duplicate_indicators(sample_data):
  with pytest.raises(ValueError):
    standardize_data(
      sample_data,
      ["GDP", "GDP"],
    )


def test_reject_non_numeric_indicator():
  data = pd.DataFrame({
    "GDP": ["low", "medium", "high"],
  })

  with pytest.raises(ValueError):
    standardize_data(data, ["GDP"])


def test_reject_missing_values():
  data = pd.DataFrame({
    "GDP": [100, np.nan, 300],
  })

  with pytest.raises(ValueError):
    standardize_data(data, ["GDP"])


def test_calculate_covariance_matrix():
  data = pd.DataFrame({
    "A": [-1.0, 0.0, 1.0],
    "B": [-1.0, 1.0, 0.0],
  })

  result = calculate_covariance_matrix(data)

  expected = np.array([
    [1.0, 0.5],
    [0.5, 1.0],
  ])

  np.testing.assert_allclose(
    result.to_numpy(),
    expected,
    atol=1e-12,
  )


def test_covariance_matrix_is_symmetric(sample_data):
  standardized = standardize_data(
    sample_data,
    ["GDP", "Life", "Internet"],
  )

  result = calculate_covariance_matrix(standardized)

  np.testing.assert_allclose(
    result.to_numpy(),
    result.to_numpy().T,
    atol=1e-12,
  )


def test_covariance_matrix_has_unit_diagonal(sample_data):
  standardized = standardize_data(
    sample_data,
    ["GDP", "Life", "Internet"],
  )

  result = calculate_covariance_matrix(standardized)

  np.testing.assert_allclose(
    np.diag(result.to_numpy()),
    1.0,
    atol=1e-12,
  )


def test_reject_empty_covariance_data():
  with pytest.raises(ValueError):
    calculate_covariance_matrix(pd.DataFrame())


def test_reject_non_finite_covariance_data():
  data = pd.DataFrame({
    "A": [1.0, np.inf, 3.0],
  })

  with pytest.raises(ValueError):
    calculate_covariance_matrix(data)


def test_calculate_eigenvalues_eigenvectors():
  covariance = pd.DataFrame(
    [
      [1.0, 0.5],
      [0.5, 1.0],
    ],
    index=["A", "B"],
    columns=["A", "B"],
  )

  eigenvalues, eigenvectors = (
    calculate_eigenvalues_eigenvectors(covariance)
  )

  np.testing.assert_allclose(
    eigenvalues.to_numpy(),
    [1.5, 0.5],
    atol=1e-12,
  )

  assert eigenvectors.shape == (2, 2)


def test_eigenvalues_are_sorted_descending():
  covariance = pd.DataFrame(
    [
      [1.0, 0.5],
      [0.5, 1.0],
    ],
    index=["A", "B"],
    columns=["A", "B"],
  )

  eigenvalues, _ = calculate_eigenvalues_eigenvectors(
    covariance
  )

  assert np.all(np.diff(eigenvalues.to_numpy()) <= 0)


def test_eigenvectors_are_orthonormal():
  covariance = pd.DataFrame(
    [
      [1.0, 0.5],
      [0.5, 1.0],
    ],
    index=["A", "B"],
    columns=["A", "B"],
  )

  _, eigenvectors = calculate_eigenvalues_eigenvectors(
    covariance
  )

  vectors = eigenvectors.to_numpy()

  np.testing.assert_allclose(
    vectors.T @ vectors,
    np.eye(2),
    atol=1e-12,
  )


def test_eigendecomposition_reconstructs_matrix():
  covariance = pd.DataFrame(
    [
      [1.0, 0.5],
      [0.5, 1.0],
    ],
    index=["A", "B"],
    columns=["A", "B"],
  )

  eigenvalues, eigenvectors = (
    calculate_eigenvalues_eigenvectors(covariance)
  )

  vectors = eigenvectors.to_numpy()
  values = np.diag(eigenvalues.to_numpy())

  reconstructed = vectors @ values @ vectors.T

  np.testing.assert_allclose(
    reconstructed,
    covariance.to_numpy(),
    atol=1e-12,
  )


def test_reject_non_symmetric_matrix():
  covariance = pd.DataFrame(
    [
      [1.0, 0.5],
      [0.2, 1.0],
    ],
    index=["A", "B"],
    columns=["A", "B"],
  )

  with pytest.raises(ValueError):
    calculate_eigenvalues_eigenvectors(covariance)


def test_reject_non_square_matrix():
  covariance = pd.DataFrame(
    [[1.0, 0.5, 0.2], [0.5, 1.0, 0.3]],
    index=["A", "B"],
    columns=["A", "B", "C"],
  )

  with pytest.raises(ValueError):
    calculate_eigenvalues_eigenvectors(covariance)


def test_reject_non_finite_matrix():
  covariance = pd.DataFrame(
    [
      [1.0, np.inf],
      [np.inf, 1.0],
    ],
    index=["A", "B"],
    columns=["A", "B"],
  )

  with pytest.raises(ValueError):
    calculate_eigenvalues_eigenvectors(covariance)


def test_calculate_explained_variance():
  eigenvalues = pd.Series(
    [3.0, 1.0],
    index=["PC1", "PC2"],
  )

  result = calculate_explained_variance(eigenvalues)

  np.testing.assert_allclose(
    result["explained_variance_ratio"],
    [0.75, 0.25],
  )

  np.testing.assert_allclose(
    result["cumulative_explained_variance"],
    [0.75, 1.0],
  )


def test_explained_variance_sums_to_one():
  eigenvalues = pd.Series([3.0, 2.0, 1.0])

  result = calculate_explained_variance(eigenvalues)

  assert np.isclose(
    result["explained_variance_ratio"].sum(),
    1.0,
  )


def test_cumulative_explained_variance_is_monotonic():
  eigenvalues = pd.Series([3.0, 2.0, 1.0])

  result = calculate_explained_variance(eigenvalues)

  differences = np.diff(
    result["cumulative_explained_variance"]
  )

  assert np.all(differences >= 0)


def test_reject_empty_eigenvalues():
  with pytest.raises(ValueError):
    calculate_explained_variance(pd.Series(dtype=float))


def test_reject_negative_eigenvalues():
  with pytest.raises(ValueError):
    calculate_explained_variance(
      pd.Series([2.0, -1.0])
    )


def test_reject_zero_total_variance():
  with pytest.raises(ValueError):
    calculate_explained_variance(
      pd.Series([0.0, 0.0])
    )


def test_calculate_component_loadings():
    eigenvalues = pd.Series(
        [4.0, 1.0],
        index=["PC1", "PC2"],
    )

    eigenvectors = pd.DataFrame(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        index=["GDP", "Life expectancy"],
        columns=["PC1", "PC2"],
    )

    result = calculate_component_loadings(eigenvalues, eigenvectors)

    expected = np.array([
        [2.0, 0.0],
        [0.0, 1.0],
    ])

    np.testing.assert_allclose(result.to_numpy(), expected)


def test_component_loadings_preserve_labels():
  eigenvalues = pd.Series([4.0, 1.0], index=["PC1", "PC2"])

  eigenvectors = pd.DataFrame(
    [[1.0, 0.0], [0.0, 1.0]],
    index=["GDP", "Life expectancy"],
    columns=["PC1", "PC2"],
  )

  result = calculate_component_loadings(eigenvalues, eigenvectors)

  assert result.index.equals(eigenvectors.index)
  assert result.columns.equals(eigenvectors.columns)


def test_reject_mismatched_component_labels():
  eigenvalues = pd.Series([4.0, 1.0], index=["PC1", "PC2"])

  eigenvectors = pd.DataFrame(
    [[1.0, 0.0], [0.0, 1.0]],
    index=["GDP", "Life expectancy"],
    columns=["PC1", "PC3"],
  )

  with pytest.raises(ValueError):
    calculate_component_loadings(eigenvalues, eigenvectors)


def test_reject_negative_eigenvalues_in_loadings():
  eigenvalues = pd.Series([4.0, -1.0], index=["PC1", "PC2"])

  eigenvectors = pd.DataFrame(
    [[1.0, 0.0], [0.0, 1.0]],
    index=["GDP", "Life expectancy"],
    columns=["PC1", "PC2"],
  )

  with pytest.raises(ValueError):
    calculate_component_loadings(eigenvalues, eigenvectors)


def test_reject_non_finite_eigenvectors():
  eigenvalues = pd.Series([4.0, 1.0], index=["PC1", "PC2"])

  eigenvectors = pd.DataFrame(
    [[1.0, np.nan], [0.0, 1.0]],
    index=["GDP", "Life expectancy"],
    columns=["PC1", "PC2"],
  )

  with pytest.raises(ValueError):
    calculate_component_loadings(eigenvalues, eigenvectors)


def test_calculate_component_scores():
  standardized_data = pd.DataFrame(
    [
      [1.0, 2.0],
      [3.0, 4.0],
    ],
    index=["A", "B"],
    columns=["GDP", "Internet"],
  )

  eigenvectors = pd.DataFrame(
    [
      [1.0, 0.0],
      [0.0, 1.0],
    ],
    index=["GDP", "Internet"],
    columns=["PC1", "PC2"],
  )

  result = calculate_component_scores(
    standardized_data,
    eigenvectors,
  )

  expected = np.array([
    [1.0, 2.0],
    [3.0, 4.0],
  ])

  np.testing.assert_allclose(result.to_numpy(), expected)


def test_component_scores_preserve_labels():
  standardized_data = pd.DataFrame(
    [[1.0, 2.0], [3.0, 4.0]],
    index=["A", "B"],
    columns=["GDP", "Internet"],
  )

  eigenvectors = pd.DataFrame(
    [[1.0, 0.0], [0.0, 1.0]],
    index=["GDP", "Internet"],
    columns=["PC1", "PC2"],
  )

  result = calculate_component_scores(
    standardized_data,
    eigenvectors,
  )

  assert result.index.equals(standardized_data.index)
  assert result.columns.equals(eigenvectors.columns)


def test_reject_mismatched_indicator_labels():
  standardized_data = pd.DataFrame(
    [[1.0, 2.0]],
    columns=["GDP", "Internet"],
  )

  eigenvectors = pd.DataFrame(
    [[1.0, 0.0], [0.0, 1.0]],
    index=["GDP", "Education"],
    columns=["PC1", "PC2"],
  )

  with pytest.raises(ValueError):
    calculate_component_scores(
      standardized_data,
      eigenvectors,
    )


def test_reject_non_finite_standardized_data():
  standardized_data = pd.DataFrame(
    [[1.0, np.nan]],
    columns=["GDP", "Internet"],
  )

  eigenvectors = pd.DataFrame(
    [[1.0, 0.0], [0.0, 1.0]],
    index=["GDP", "Internet"],
    columns=["PC1", "PC2"],
  )

  with pytest.raises(ValueError):
    calculate_component_scores(
      standardized_data,
      eigenvectors,
    )