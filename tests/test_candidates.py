"""
Unit tests for P9.2 candidate spelling discovery.
"""

from collections import Counter

from src.candidates import mine_candidate_pairs


def test_candidate_discovery():
    """
    Confirm that edit-distance candidates can be discovered.
    """

    lexicon = Counter({
        "thetha": 30,
        "rhulumente": 20,
    })

    historical = [
        "teta rulumente"
    ] * 5

    pairs = mine_candidate_pairs(
        historical,
        lexicon,
        min_hist_freq=5,
    )

    discovered = set(
        zip(pairs["hist_form"], pairs["cand_form"])
    )

    assert ("teta", "thetha") in discovered
    assert ("rulumente", "rhulumente") in discovered


def test_word_division():
    """
    Confirm that adjacent historical words can form
    a modern lexicon word.
    """

    lexicon = Counter({
        "ndiyazi": 10,
    })

    historical = ["ndi yazi"] * 5

    pairs = mine_candidate_pairs(
        historical,
        lexicon,
        min_hist_freq=5,
    )

    matches = pairs[pairs["kind"] == "wd"]

    assert (
        (matches["hist_form"] == "ndi yazi")
        & (matches["cand_form"] == "ndiyazi")
    ).any()


def test_hyphen_candidate():
    """
    Confirm that removing a historical hyphen can
    produce a modern lexicon candidate.
    """

    lexicon = Counter({
        "katixo": 10,
    })

    historical = ["ka-tixo"] * 5

    pairs = mine_candidate_pairs(
        historical,
        lexicon,
        min_hist_freq=5,
    )

    matches = pairs[pairs["kind"] == "hyphen"]

    assert (
        (matches["hist_form"] == "ka-tixo")
        & (matches["cand_form"] == "katixo")
    ).any()