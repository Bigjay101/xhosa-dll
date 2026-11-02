"""
Tests for P9.1 character auditing.
"""

import pytest

from src.orthography import char_audit


def test_char_audit_counts():
    """
    Check character counts and frequency rates.
    """

    texts = {
        "imvo": ["aabb"],
        "modern_M1": ["aaaa"],
    }

    result = char_audit(texts)

    row = result[
        (result["source"] == "imvo")
        & (result["character"] == "a")
    ].iloc[0]

    assert row["count"] == 2

    # Two a's out of four characters = 5,000 per 10,000.
    assert row["rate_per_10k"] == pytest.approx(5000)


def test_historical_diacritic_flag():
    """
    Verify that historically enriched non-ASCII letters
    are identified.
    """

    texts = {
        "isigidimi": ["á" * 12 + "a" * 4],
        "modern_M1": ["a" * 16],
    }

    result = char_audit(texts)

    row = result[
        (result["source"] == "isigidimi")
        & (result["character"] == "á")
    ].iloc[0]

    assert bool(row["is_non_ascii_letter"])
    assert bool(row["historically_enriched"])


def test_ascii_letters_not_flagged():
    """
    Ordinary ASCII letters should not be labelled
    as non-ASCII historical characters.
    """

    result = char_audit({
        "imvo": ["hello"],
        "modern_M1": ["hello"],
    })

    assert not result["historically_enriched"].any()