"""T14 (Member C) -- STUB. Asserts: zero-width CI for constants; coverage 0.95 +/- 0.04; sign-flip null; Holm example.
Implement per handbook D3, then delete the skip marker."""
import pytest

"""
T14: Unit tests for P11 statistical functions.

Tests use synthetic data so that correctness can be checked
without downloading any language models.
"""

import pytest
import numpy as np
from src.stats import (cluster_bootstrap_mean, cluster_bootstrap_diff)


def test_cluster_bootstrap_constant_values():
    """
    A constant dataset should have a zero-width confidence interval.

    There is no variation in the observations, so every
    bootstrap sample must have the same mean.
    """

    values = [7, 7, 7, 7]
    clusters = ["A", "A", "B", "B"]

    mean, lower, upper = cluster_bootstrap_mean(
        values,
        clusters,
        n_boot=1000
    )

    assert mean == pytest.approx(7.0)
    assert lower == pytest.approx(7.0)
    assert upper == pytest.approx(7.0)


def test_cluster_bootstrap_mean():
    """
    Verify that the observed mean is correctly calculated
    and lies within the confidence interval for this dataset.
    """

    values = [1, 3, 5, 7, 9, 11]
    clusters = ["A", "A", "B", "B", "C", "C"]

    mean, lower, upper = cluster_bootstrap_mean(
        values,
        clusters,
        n_boot=1000,
        seed=13
    )

    assert mean == pytest.approx(6.0)
    assert lower <= mean <= upper


def test_cluster_bootstrap_reproducibility():
    """
    Verify that using the same seed produces identical results.
    """

    values = [1, 3, 5, 7, 9, 11]
    clusters = ["A", "A", "B", "B", "C", "C"]

    result_1 = cluster_bootstrap_mean(
        values, clusters, n_boot=1000, seed=13
    )

    result_2 = cluster_bootstrap_mean(
        values, clusters, n_boot=1000, seed=13
    )

    assert result_1 == result_2

def test_cluster_bootstrap_diff_constant():
    """
    Two constant groups should produce a known difference
    with a zero-width confidence interval.
    """

    historical = [10, 10, 10, 10]
    historical_clusters = ["A", "A", "B", "B"]

    modern = [6, 6, 6, 6]
    modern_clusters = ["C", "C", "D", "D"]

    estimate, lower, upper, reps = cluster_bootstrap_diff(
        historical,
        historical_clusters,
        modern,
        modern_clusters,
        n_boot=1000
    )

    assert estimate == pytest.approx(4.0)
    assert lower == pytest.approx(4.0)
    assert upper == pytest.approx(4.0)

    # Every bootstrap repetition should give the same difference.
    assert all(value == pytest.approx(4.0) for value in reps)


def test_cluster_bootstrap_diff_reproducible():
    """
    Identical inputs and seeds should produce identical results.
    """

    historical = [10, 12, 14, 16, 18, 20]
    historical_clusters = ["A", "A", "B", "B", "C", "C"]

    modern = [5, 7, 9, 11]
    modern_clusters = ["D", "D", "E", "E"]

    result_1 = cluster_bootstrap_diff(
        historical,
        historical_clusters,
        modern,
        modern_clusters,
        n_boot=1000,
        seed=13
    )

    result_2 = cluster_bootstrap_diff(
        historical,
        historical_clusters,
        modern,
        modern_clusters,
        n_boot=1000,
        seed=13
    )

    assert result_1[0:3] == result_2[0:3]
    assert np.array_equal(result_1[3], result_2[3])