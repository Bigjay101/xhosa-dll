"""P11.1 extra cluster-aware inference, complementing existing src/stats.py.

This file is separate to avoid overwriting the user's tested mean/diff functions.
"""
from __future__ import annotations

import numpy as np


def paired_cluster_bootstrap(d, clusters, n_boot=10000, ci=.95, seed=13):
    """Chunk-weighted paired mean difference with edition/document resampling."""
    values = np.asarray(d, dtype=float)
    groups = np.asarray(clusters).astype(str)
    if values.ndim != 1 or len(values) == 0 or len(values) != len(groups):
        raise ValueError('Mismatched paired observations and cluster IDs')
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite paired differences')
    _, inverse = np.unique(groups, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    sizes = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.integers(len(sums), size=len(sums))
        reps[i] = sums[sample].sum() / sizes[sample].sum()
    lower, upper = np.quantile(reps, [(1-ci)/2, (1+ci)/2])
    return float(values.mean()), float(lower), float(upper), reps


def sign_flip_test(d, clusters, n_perm=10000, seed=13):
    """Two-sided cluster sign-flip test for a zero mean paired effect.

    Sign-flips are applied by cluster, preserving within-cluster dependence.
    We test the chunk-weighted mean, consistent with paired bootstrap point
    estimates and the primary estimand.
    """
    vals = np.asarray(d, dtype=float)
    ids = np.asarray(clusters).astype(str)
    if len(vals) == 0 or len(vals) != len(ids):
        raise ValueError('Invalid paired data')
    _, inv = np.unique(ids, return_inverse=True)
    sums = np.bincount(inv, weights=vals)
    obs = abs(sums.sum())
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=len(sums))
        extreme += abs(np.dot(signs, sums)) >= obs - 1e-12
    return (int(extreme)+1)/(n_perm+1)


def bootstrap_p(reps):
    """Two-sided percentile-bootstrap tail area, clipped to one."""
    data = np.asarray(reps, dtype=float)
    if data.size == 0 or not np.isfinite(data).all():
        raise ValueError('Invalid bootstrap values')
    return min(1., 2*min(float((data<=0).mean()), float((data>=0).mean())))


def holm(pvals):
    """Holm step-down multiplicity adjustment, original ordering preserved."""
    values = np.asarray(pvals, dtype=float)
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError('Invalid p-values')
    order = np.argsort(values)
    corrected = np.zeros(len(values))
    running = 0.
    for rank, idx in enumerate(order):
        running = max(running, (len(values)-rank)*values[idx])
        corrected[idx] = min(1., running)
    return corrected.tolist()


def ratio_ci(num_reps, den_reps, ci=.95):
    """Percentile CI of joint bootstrap numerator/denominator ratios."""
    num, den = np.asarray(num_reps, float), np.asarray(den_reps, float)
    if num.shape != den.shape:
        raise ValueError('Joint bootstrap draws have unequal shapes')
    valid = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > 1e-10)
    if not len(den) or valid.mean() < .95:
        raise ValueError('Unstable gap-closure denominator')
    return tuple(float(v) for v in np.quantile(num[valid]/den[valid],
                                               [(1-ci)/2, (1+ci)/2]))
