"""
P9.5: Unit tests for PPMI-weighted context vectors.
"""
from collections import Counter

import pandas as pd
import numpy as np

from src.rules import context_vectors
from src.rules import select_context_anchors

def test_context_vectors_detect_associations():
    """
    Words occurring near different anchors should receive
    stronger weights for their associated anchors.
    """

    token_lists = [
        ["historical", "abantu", "historical", "abantu"],
        ["historical", "abantu"],
        ["modern", "ilizwe", "modern", "ilizwe"],
        ["modern", "ilizwe"],
    ]

    vectors = context_vectors(
        token_lists=token_lists,
        targets={"historical", "modern"},
        anchors=["abantu", "ilizwe"],
        window=1,
    )

    # Historical should be more associated with abantu.
    assert vectors["historical"][0] > vectors["historical"][1]

    # Modern should be more associated with ilizwe.
    assert vectors["modern"][1] > vectors["modern"][0]


def test_context_vectors_reproducible():
    """
    The same input should produce identical context vectors.
    """

    texts = [
        ["historical", "abantu"],
        ["modern", "ilizwe"],
    ]

    args = {
        "token_lists": texts,
        "targets": {"historical", "modern"},
        "anchors": ["abantu", "ilizwe"],
        "window": 1,
    }

    result_1 = context_vectors(**args)
    result_2 = context_vectors(**args)

    for word in result_1:
        np.testing.assert_array_equal(
            result_1[word],
            result_2[word]
        )


def test_context_vectors_missing_target():
    """
    Unobserved targets should not receive fabricated vectors.
    """

    vectors = context_vectors(
        token_lists=[
            ["historical", "abantu"],
            ["modern", "ilizwe"],
        ],
        targets={"historical", "modern", "missing"},
        anchors=["abantu", "ilizwe"],
    )

    assert "missing" not in vectors

def test_anchor_selection():
    """
    Shared frequent words should be selected,
    excluding candidate rewrite forms.
    """

    historical = [
        "abantu abantu ILIZWE, umthetho",
        "abantu ilizwe incwadi akuko ndi yazi",
    ]

    modern_lexicon = Counter({
        "abantu": 100,
        "ilizwe": 40,
        "umthetho": 30,
        "incwadi": 8,
        "akukho": 10,
        "ndiyazi": 5,
    })

    candidates = pd.DataFrame({
        "hist_form": ["akuko", "ndi yazi"],
        "cand_form": ["akukho", "ndiyazi"],
    })

    anchors = select_context_anchors(
        historical_texts=historical,
        modern_lexicon=modern_lexicon,
        candidate_pairs=candidates,
        n_anchors=3,
    )

    assert anchors == [
        "abantu",
        "ilizwe",
        "umthetho",
    ]


def test_anchor_selection_is_deterministic():
    """
    Reordering input documents must not change anchor selection.
    """

    historical = [
        "abantu ilizwe",
        "ilizwe abantu",
    ]

    modern_lexicon = Counter({
        "abantu": 20,
        "ilizwe": 10,
    })

    empty_candidates = pd.DataFrame({
        "hist_form": [],
        "cand_form": [],
    })

    first = select_context_anchors(
        historical,
        modern_lexicon,
        empty_candidates,
        n_anchors=2,
    )

    second = select_context_anchors(
        reversed(historical),
        modern_lexicon,
        empty_candidates,
        n_anchors=2,
    )

    assert first == second