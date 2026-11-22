import pandas as pd
import pytest

from global_pca.analysis import calculate_descriptive_statistics


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