"""P11.1 Statistics module (Member C). STUB.

Implement per handbook D2. Tests: T14.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cluster_bootstrap_mean(x, clusters, n_boot: int = 10000, ci: float = 0.95, seed: int = 13) -> tuple[float, float, float]:
    """Mean and percentile CI, resampling whole clusters."""
    # STEP 1: Convert the inputs into NumPy arrays.
    values = np.asarray(x, dtype=float)
    cluster_ids = np.asarray(clusters)

    # STEP 2: Check that the inputs are valid.
    if (
        values.ndim != 1
        or cluster_ids.ndim != 1
        or len(values) != len(cluster_ids)
    ):
        raise ValueError(
            "x and clusters must be one-dimensional and have equal length"
        )

    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("x must contain finite numerical values")

    if pd.isna(cluster_ids).any():
        raise ValueError("Cluster IDs cannot be missing")

    if n_boot < 1 or not 0 < ci < 1:
        raise ValueError("Invalid n_boot or confidence interval")

    # STEP 3: Identify unique clusters.
    # group_index maps every chunk to its cluster.
    unique_ids, group_index = np.unique(
        cluster_ids,
        return_inverse=True
    )

    n_clusters = len(unique_ids)

    # STEP 4: Calculate the total NLL and number of chunks
    # within each cluster.
    #
    # This lets us resample whole editions efficiently without
    # repeatedly copying their individual chunks.
    cluster_sums = np.bincount(
        group_index,
        weights=values,
        minlength=n_clusters
    )

    cluster_sizes = np.bincount(
        group_index,
        minlength=n_clusters
    )

    # STEP 5: Calculate the observed mean across all chunks.
    estimate = float(values.mean())

    # STEP 6: Create a reproducible random-number generator.
    rng = np.random.default_rng(seed)

    bootstrap_means = np.empty(n_boot, dtype=float)

    # STEP 7: Perform cluster bootstrap resampling.
    for draw in range(n_boot):

        # Sample the same number of clusters as originally observed.
        # Replacement means an edition may be selected more than once.
        sampled = rng.integers(
            0,
            n_clusters,
            size=n_clusters
        )

        # Include every chunk from every sampled cluster.
        #
        # Dividing total NLL by total sampled chunks preserves
        # the original chunk-weighted mean definition.
        bootstrap_means[draw] = (
            cluster_sums[sampled].sum()
            / cluster_sizes[sampled].sum()
        )

    # STEP 8: Calculate the percentile confidence interval.
    # For a 95% CI, these are the 2.5th and 97.5th percentiles.
    tail_pct = 100 * (1.0 - ci) / 2.0

    lower, upper = np.percentile(
        bootstrap_means,
        [tail_pct, 100.0 - tail_pct]
    )

    # STEP 9: Return the observed mean and its confidence interval.
    return estimate, float(lower), float(upper)

def cluster_bootstrap_diff(x, cx, y, cy, n_boot: int = 10000, ci: float = 0.95, seed: int = 13) -> tuple:
    """Difference of means for independent groups; resample clusters within each group.
    Returns (estimate, ci_lo, ci_hi, replicates)."""
     # STEP 1: Convert the score arrays into NumPy arrays.
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    cx = np.asarray(cx)
    cy = np.asarray(cy)

    # STEP 2: Validate the inputs.
    for values, clusters in [(x, cx), (y, cy)]:

        if (
            values.ndim != 1
            or clusters.ndim != 1
            or len(values) == 0
            or len(values) != len(clusters)
        ):
            raise ValueError(
                "Each group requires equally sized, nonempty 1D arrays"
            )

        if not np.isfinite(values).all():
            raise ValueError("NLL values must be finite")

        if pd.isna(clusters).any():
            raise ValueError("Cluster IDs cannot be missing")

    if n_boot < 1 or not 0 < ci < 1:
        raise ValueError("Invalid bootstrap parameters")

    # STEP 3: Compute each group's observed mean.
    observed_diff = float(x.mean() - y.mean())

    # STEP 4: Group chunks by their cluster IDs.
    #
    # The first output gives the unique editions/documents.
    # The second identifies which cluster each chunk belongs to.
    unique_x, group_x = np.unique(cx, return_inverse=True)
    unique_y, group_y = np.unique(cy, return_inverse=True)

    # STEP 5: Calculate sums and chunk counts for each cluster.
    #
    # These allow us to resample whole editions/documents.
    sums_x = np.bincount(group_x, weights=x)
    counts_x = np.bincount(group_x)

    sums_y = np.bincount(group_y, weights=y)
    counts_y = np.bincount(group_y)

    # STEP 6: Initialise the reproducible random generator.
    rng = np.random.default_rng(seed)

    bootstrap_diffs = np.empty(n_boot, dtype=float)

    # STEP 7: Perform independent cluster resampling.
    for i in range(n_boot):

        # Sample historical newspaper editions with replacement.
        sampled_x = rng.integers(
            0, len(unique_x), size=len(unique_x)
        )

        # Independently sample modern documents with replacement.
        sampled_y = rng.integers(
            0, len(unique_y), size=len(unique_y)
        )

        # Calculate chunk-weighted means for both samples.
        mean_x = (
            sums_x[sampled_x].sum()
            / counts_x[sampled_x].sum()
        )

        mean_y = (
            sums_y[sampled_y].sum()
            / counts_y[sampled_y].sum()
        )

        # Store the difference for this bootstrap repetition.
        bootstrap_diffs[i] = mean_x - mean_y

    # STEP 8: Calculate the percentile confidence interval.
    alpha = (1.0 - ci) / 2.0

    lower, upper = np.quantile(
        bootstrap_diffs,
        [alpha, 1.0 - alpha]
    )

    # STEP 9: Return the estimate, confidence interval
    # and bootstrap replicates for subsequent statistical tests.
    return (
        observed_diff,
        float(lower),
        float(upper),
        bootstrap_diffs
    )


def paired_cluster_bootstrap(d, clusters, n_boot: int = 10000, ci: float = 0.95, seed: int = 13) -> tuple:
    """Mean of paired per-chunk differences d; resample clusters.
    Returns (estimate, ci_lo, ci_hi, replicates)."""
    raise NotImplementedError("P11.1 (Member C): implement per handbook P11.1")


def sign_flip_test(d, clusters, n_perm: int = 10000, seed: int = 13) -> float:
    """Two-sided p-value: randomly flip the sign of each cluster's mean difference."""
    raise NotImplementedError("P11.1 (Member C): implement per handbook P11.1")


def bootstrap_p(reps) -> float:
    """Two-sided bootstrap p = 2 * min(P(rep <= 0), P(rep >= 0))."""
    raise NotImplementedError("P11.1 (Member C): implement per handbook P11.1")


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    raise NotImplementedError("P11.1 (Member C): implement per handbook P11.1")


def ratio_ci(num_reps, den_reps, ci: float = 0.95) -> tuple:
    """CI for a ratio (e.g. GC) from joint replicates."""
    raise NotImplementedError("P11.1 (Member C): implement per handbook P11.1")
