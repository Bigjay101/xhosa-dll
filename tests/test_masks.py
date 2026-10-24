
"""T8: Tests for masked-language-model masking (handbook D3)."""

import pytest

from src.mlm_scoring import build_masks


def test_l2r_masks():
    """Mask the current token and later pieces of the same word."""
    word_ids = [None, 0, 0, 1, 1, None]

    expected = [
        [1, 2],
        [2],
        [3, 4],
        [4],
    ]

    assert build_masks(word_ids, method="l2r") == expected


def test_original_pll_masks():
    """Original PLL masks only the current token."""
    word_ids = [None, 0, 0, 1, 1, None]

    expected = [
        [1],
        [2],
        [3],
        [4],
    ]

    assert build_masks(word_ids, method="orig") == expected


def test_special_tokens_are_skipped():
    """Special tokens must never become prediction targets."""
    word_ids = [None, 0, None, 1, None]

    assert build_masks(word_ids) == [[1], [3]]


def test_empty_sequence():
    """No regular tokens means no predictions."""
    assert build_masks([]) == []
    assert build_masks([None, None]) == []


def test_invalid_method():
    """Unsupported masking methods should fail explicitly."""
    with pytest.raises(ValueError):
        build_masks([None, 0, None], method="invalid")
