"""P11.1 Statistics module (Member C). STUB.

Implement per handbook D2. Tests: T14.
"""
from __future__ import annotations

import pandas as pd  # noqa: F401


def cluster_bootstrap_mean(x, clusters, n_boot: int = 10000, ci: float = 0.95, seed: int = 13) -> tuple[float, float, float]:
    """Mean and percentile CI, resampling whole clusters."""
    raise NotImplementedError("P11.1 (Member C): implement per handbook P11.1")


def cluster_bootstrap_diff(x, cx, y, cy, n_boot: int = 10000, ci: float = 0.95, seed: int = 13) -> tuple:
    """Difference of means for independent groups; resample clusters within each group.
    Returns (estimate, ci_lo, ci_hi, replicates)."""
    raise NotImplementedError("P11.1 (Member C): implement per handbook P11.1")


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
