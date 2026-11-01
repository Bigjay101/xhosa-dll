"""
P9.5: Synthetic tests for contextual spelling-family validation.
"""

import numpy as np
import pandas as pd

from src.rules import family_validity


def test_family_with_matching_contexts_passes():
    """
    Proposed modern forms match their historical contexts,
    while alternative modern forms have different contexts.
    """

    pairs = pd.DataFrame({
        "hist_form": ["kala", "kata"],
        "cand_form": ["khala", "khata"],
        "family": ["ASP", "ASP"],
    })

    historical_vectors = {
        "kala": np.array([1.0, 0.0]),
        "kata": np.array([0.0, 1.0]),
    }

    modern_vectors = {
        "khala": np.array([1.0, 0.0]),
        "khata": np.array([0.0, 1.0]),
    }

    frequencies = {
        "khala": 10,
        "khata": 10,
    }

    results = family_validity(
        pairs,
        historical_vectors,
        modern_vectors,
        frequencies,
        n_boot=100,
        min_support=2,
    )

    row = results.iloc[0]

    assert row["n_validated_pairs"] == 2
    assert row["delta"] > 0
    assert bool(row["accepted"])


def test_missing_vectors_do_not_count_as_evidence():
    """Missing historical contexts cannot validate a family."""

    pairs = pd.DataFrame({
        "hist_form": ["kala"],
        "cand_form": ["khala"],
        "family": ["ASP"],
    })

    results = family_validity(
        pairs,
        hist_vecs={},
        mod_vecs={"khala": np.array([1.0, 0.0])},
        mod_freqs={"khala": 10},
        n_boot=100,
    )

    assert results.iloc[0]["n_validated_pairs"] == 0
    assert not bool(results.iloc[0]["accepted"])