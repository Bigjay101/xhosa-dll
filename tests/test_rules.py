"""
P9.3 tests: character-edit extraction and pattern counting.

Synthetic examples are used to verify correctness independently
of the historical corpus.
"""

import pandas as pd
import pytest

from src.rules import edit_patterns, count_patterns, classify_pattern, assign_families


def test_insert_pattern():
    """
    Verify insertion of h in a historical spelling.
    """

    edits = edit_patterns("kakulu", "kakhulu")

    assert edits == [
        ("insert", "", "h", "k", "u")
    ]


def test_multiple_insertions():
    """
    Verify that multiple edits are extracted from one pair.
    """

    edits = edit_patterns("teta", "thetha")

    assert edits == [
        ("insert", "", "h", "t", "e"),
        ("insert", "", "h", "t", "a"),
    ]


def test_hyphen_deletion():
    """
    Verify that removing a hyphen is identified as deletion.
    """

    edits = edit_patterns("ka-tixo", "katixo")

    assert edits == [
        ("delete", "-", "", "a", "t")
    ]


def test_identical_words_have_no_edits():
    """
    Identical words should not generate spelling patterns.
    """

    assert edit_patterns("thetha", "thetha") == []


def test_pattern_support_and_weight():
    """
    Verify support counts distinct pairs and weight sums
    their historical frequencies.

    A duplicated pair and repeated identical edits within
    a word pair must not inflate support.
    """

    pairs = pd.DataFrame({
        "hist_form": ["tata", "tata", "teta", "inteto"],
        "cand_form": ["thatha", "thatha", "thetha", "intetho"],
        "hist_freq": [20, 20, 7, 3],
    })

    result = count_patterns(
        pairs,
        min_support=2
    )

    # Find insertion of h between t and a.
    matching = result[
        (result["op"] == "insert")
        & (result["tgt"] == "h")
        & (result["left_char"] == "t")
        & (result["right_char"] == "a")
    ]

    assert len(matching) == 1

    pattern = matching.iloc[0]

    # tata -> thatha and teta -> thetha support this pattern.
    assert pattern["support"] == 2

    # Sum of the historical frequencies: 20 + 7.
    assert pattern["weight"] == 27


def test_minimum_support_filter():
    """
    Patterns below the minimum support must be excluded.
    """

    pairs = pd.DataFrame({
        "hist_form": ["kakulu"],
        "cand_form": ["kakhulu"],
        "hist_freq": [100],
    })

    result = count_patterns(
        pairs,
        min_support=2
    )

    assert result.empty

def test_asp_family():
    """Insertion of h after k should be classified as aspiration."""
    assert classify_pattern(
        "insert", "", "h", "k", "u"
    ) == "ASP"


def test_vel_family():
    """Insertion of h after r belongs to the VEL family."""
    assert classify_pattern(
        "insert", "", "h", "r", "a"
    ) == "VEL"


def test_bh_family():
    """Insertion of h after b belongs to the BH family."""
    assert classify_pattern(
        "insert", "", "h", "b", "a"
    ) == "BH"


def test_nas_family():
    """Insertion of g in nq -> ngq belongs to NAS."""
    assert classify_pattern(
        "insert", "", "g", "n", "q"
    ) == "NAS"


def test_sv_family():
    """Insertion of w between two vowels belongs to SV."""
    assert classify_pattern(
        "insert", "", "w", "a", "u"
    ) == "SV"


def test_hyphen_family():
    """Hyphen deletion should be classified separately."""
    assert classify_pattern(
        "delete", "-", "", "a", "T"
    ) == "HYP"


def test_unclassified_pattern():
    """Unrecognised substitutions must remain OTHER."""
    assert classify_pattern(
        "replace", "z", "b", "a", "i"
    ) == "OTHER"