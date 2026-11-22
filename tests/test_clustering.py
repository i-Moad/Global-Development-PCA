
import numpy as np
import pandas as pd
import pytest

from global_pca.clustering import evaluate_cluster_counts
from global_pca.clustering import fit_kmeans


@pytest.fixture
def sample_scores():
    """Create a small dataset with three visible groups."""
    return pd.DataFrame({
        "PC1": [
            -5.0, -4.8, -5.2,
            0.0, 0.2, -0.2,
            5.0, 5.2, 4.8,
        ],
        "PC2": [
            -5.0, -4.8, -5.2,
            0.0, 0.2, -0.2,
            5.0, 5.2, 4.8,
        ],
    })


def test_evaluate_cluster_counts_returns_expected_columns(sample_scores):
    result = evaluate_cluster_counts(
        scores=sample_scores,
        component_columns=["PC1", "PC2"],
        k_values=range(2, 5),
    )

    assert list(result.columns) == [
        "n_clusters",
        "inertia",
        "silhouette_score",
    ]


def test_evaluate_cluster_counts_returns_all_candidates(sample_scores):
    result = evaluate_cluster_counts(
        scores=sample_scores,
        component_columns=["PC1", "PC2"],
        k_values=range(2, 5),
    )

    assert result["n_clusters"].tolist() == [2, 3, 4]


def test_inertia_is_non_negative(sample_scores):
    result = evaluate_cluster_counts(
        scores=sample_scores,
        component_columns=["PC1", "PC2"],
    )

    assert (result["inertia"] >= 0).all()


def test_silhouette_scores_are_valid(sample_scores):
    result = evaluate_cluster_counts(
        scores=sample_scores,
        component_columns=["PC1", "PC2"],
    )

    assert result["silhouette_score"].between(-1, 1).all()


def test_three_clusters_have_high_silhouette_score(sample_scores):
    result = evaluate_cluster_counts(
        scores=sample_scores,
        component_columns=["PC1", "PC2"],
        k_values=range(2, 5),
    )

    score = result.loc[
        result["n_clusters"] == 3,
        "silhouette_score",
    ].iloc[0]

    assert score > 0.8


def test_results_are_reproducible(sample_scores):
    first_result = evaluate_cluster_counts(
        scores=sample_scores,
        component_columns=["PC1", "PC2"],
    )

    second_result = evaluate_cluster_counts(
        scores=sample_scores,
        component_columns=["PC1", "PC2"],
    )

    pd.testing.assert_frame_equal(first_result, second_result)


def test_reject_empty_dataframe():
    with pytest.raises(ValueError, match="cannot be empty"):
        evaluate_cluster_counts(
            scores=pd.DataFrame(),
            component_columns=["PC1"],
        )


def test_reject_missing_component():
    scores = pd.DataFrame({"PC1": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match="Missing"):
        evaluate_cluster_counts(
            scores=scores,
            component_columns=["PC2"],
        )


def test_reject_non_numeric_components():
    scores = pd.DataFrame({
        "PC1": ["a", "b", "c"],
    })

    with pytest.raises(ValueError, match="numeric"):
        evaluate_cluster_counts(
            scores=scores,
            component_columns=["PC1"],
        )


def test_reject_non_finite_values():
    scores = pd.DataFrame({
        "PC1": [1.0, np.nan, 3.0],
    })

    with pytest.raises(ValueError, match="finite"):
        evaluate_cluster_counts(
            scores=scores,
            component_columns=["PC1"],
        )


def test_reject_duplicate_component_columns():
    scores = pd.DataFrame({
        "PC1": [1.0, 2.0, 3.0],
    })

    with pytest.raises(ValueError, match="unique"):
        evaluate_cluster_counts(
            scores=scores,
            component_columns=["PC1", "PC1"],
        )


def test_reject_invalid_cluster_count(sample_scores):
    with pytest.raises(ValueError, match="between"):
        evaluate_cluster_counts(
            scores=sample_scores,
            component_columns=["PC1", "PC2"],
            k_values=[2, 9],
        )


def test_reject_empty_cluster_counts(sample_scores):
    with pytest.raises(ValueError, match="[Aa]t least one"):
        evaluate_cluster_counts(
            scores=sample_scores,
            component_columns=["PC1", "PC2"],
            k_values=[],
        )


def test_reject_too_few_observations():
    scores = pd.DataFrame({
        "PC1": [1.0, 2.0],
    })

    with pytest.raises(ValueError, match="At least three"):
        evaluate_cluster_counts(
            scores=scores,
            component_columns=["PC1"],
        )



def test_fit_kmeans_returns_assignments_and_model(sample_scores):
    from global_pca.clustering import fit_kmeans

    assignments, model = fit_kmeans(
        scores=sample_scores,
        component_columns=["PC1", "PC2"],
        n_clusters=3,
    )

    assert isinstance(assignments, pd.Series)
    assert assignments.shape[0] == len(sample_scores)
    assert model.n_clusters == 3


def test_fit_kmeans_preserves_index(sample_scores):
    from global_pca.clustering import fit_kmeans

    scores = sample_scores.copy()
    scores.index = [f"ECONOMY_{i}" for i in range(len(scores))]

    assignments, _ = fit_kmeans(
        scores=scores,
        component_columns=["PC1", "PC2"],
        n_clusters=3,
    )

    assert assignments.index.equals(scores.index)


def test_fit_kmeans_assigns_valid_cluster_labels(sample_scores):
    from global_pca.clustering import fit_kmeans

    assignments, _ = fit_kmeans(
        scores=sample_scores,
        component_columns=["PC1", "PC2"],
        n_clusters=3,
    )

    assert set(assignments.unique()) == {0, 1, 2}


def test_fit_kmeans_is_reproducible(sample_scores):
    from global_pca.clustering import fit_kmeans

    first_assignments, _ = fit_kmeans(
        sample_scores,
        ["PC1", "PC2"],
        n_clusters=3,
    )

    second_assignments, _ = fit_kmeans(
        sample_scores,
        ["PC1", "PC2"],
        n_clusters=3,
    )

    pd.testing.assert_series_equal(
        first_assignments,
        second_assignments,
    )


def test_fit_kmeans_rejects_missing_component(sample_scores):
    from global_pca.clustering import fit_kmeans

    with pytest.raises(ValueError, match="Missing"):
        fit_kmeans(
            sample_scores,
            ["PC3"],
            n_clusters=3,
        )


def test_fit_kmeans_rejects_invalid_cluster_count(sample_scores):
    from global_pca.clustering import fit_kmeans

    with pytest.raises(ValueError, match="between"):
        fit_kmeans(
            sample_scores,
            ["PC1", "PC2"],
            n_clusters=9,
        )


def test_fit_kmeans_rejects_non_integer_cluster_count(sample_scores):
    from global_pca.clustering import fit_kmeans

    with pytest.raises(ValueError, match="integer"):
        fit_kmeans(
            sample_scores,
            ["PC1", "PC2"],
            n_clusters=2.5,
        )


def test_fit_kmeans_rejects_non_finite_values():
    from global_pca.clustering import fit_kmeans

    scores = pd.DataFrame({
        "PC1": [1.0, np.nan, 3.0],
        "PC2": [1.0, 2.0, 3.0],
    })

    with pytest.raises(ValueError, match="finite"):
        fit_kmeans(
            scores,
            ["PC1", "PC2"],
            n_clusters=2,
        )
