"""
P9.2: Candidate historical-to-modern spelling pairs.

Uses historical training text and the frozen modern lexicon.
No historical test data or model scores are used.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping

import pandas as pd
from symspellpy import SymSpell, Verbosity

from src.lexicon import lex_key
from src.segment import words


COLUMNS = [
    "hist_form",
    "cand_form",
    "edit_distance",
    "hist_freq",
    "cand_freq",
    "kind",
]


def mine_candidate_pairs(
    train_texts: Iterable[str],
    lexicon: Mapping[str, int],
    min_hist_freq: int = 5,
    max_edit_distance: int = 2,
    max_suggestions: int = 3,
) -> pd.DataFrame:
    """
    Identify possible historical spelling variants.

    Parameters
    ----------
    train_texts:
        Historical TRAINING text only.

    lexicon:
        Frozen modern isiXhosa word-frequency dictionary.

    min_hist_freq:
        Minimum historical word frequency required for edit candidates.

    max_edit_distance:
        Maximum Levenshtein distance allowed.

    max_suggestions:
        Maximum number of modern candidates retained per historical type.

    Returns
    -------
    pd.DataFrame
        Candidate spelling pairs and their frequencies.

    Notes
    -----
    Candidate generation is unsupervised.

    The function does not determine whether a candidate is a genuine
    historical spelling variant. That requires contextual validation.
    """

    # STEP 1: Count historical word types and adjacent word pairs.
    token_counts = Counter()
    bigram_counts = Counter()

    for text in train_texts:

        # Use the project's standard whitespace-word definition.
        tokens = [lex_key(w) for w in words(text)]

        # Ignore empty normalized tokens.
        token_counts.update(t for t in tokens if t)

        # Count adjacent words for possible word-division changes.
        bigram_counts.update(
            (a, b)
            for a, b in zip(tokens, tokens[1:])
            if a and b
        )

    # STEP 2: Build a SymSpell dictionary using the modern lexicon.
    spell = SymSpell(
        max_dictionary_edit_distance=max_edit_distance,
        prefix_length=7,
    )

    for modern_word, frequency in lexicon.items():
        spell.create_dictionary_entry(modern_word, int(frequency))

    rows = []

    # STEP 3: Find modern candidates for historical OOV types.
    for historical_word, historical_freq in token_counts.items():

        # Only investigate sufficiently frequent historical OOVs.
        if historical_freq < min_hist_freq:
            continue

        if historical_word in lexicon:
            continue

        # Ignore numbers and obviously corrupted character sequences.
        if not any(ch.isalpha() for ch in historical_word):
            continue

        if "\ufffd" in historical_word:
            continue

        # Find modern words within the permitted edit distance.
        suggestions = spell.lookup(
            historical_word,
            Verbosity.ALL,
            max_edit_distance=max_edit_distance,
            include_unknown=False,
        )

        # Prefer smaller edit distance, then higher modern frequency.
        suggestions = sorted(
            suggestions,
            key=lambda s: (s.distance, -s.count, s.term),
        )[:max_suggestions]

        for suggestion in suggestions:
            if suggestion.term == historical_word:
                continue

            rows.append({
                "hist_form": historical_word,
                "cand_form": suggestion.term,
                "edit_distance": suggestion.distance,
                "hist_freq": historical_freq,
                "cand_freq": suggestion.count,
                "kind": "edit",
            })

    # STEP 4: Identify possible historical word-division differences.
    #
    # Example:
    #   "ndi yazi" -> "ndiyazi"
    #
    # One of the two historical parts must be absent from the lexicon.
    for (a, b), frequency in bigram_counts.items():

        merged = a + b

        if (
            merged in lexicon
            and lexicon[merged] >= 5
            and (a not in lexicon or b not in lexicon)
        ):
            rows.append({
                "hist_form": f"{a} {b}",
                "cand_form": merged,
                "edit_distance": 1,
                "hist_freq": frequency,
                "cand_freq": lexicon[merged],
                "kind": "wd",
            })

    # STEP 5: Identify possible historical hyphen differences.
    #
    # Example:
    #   "ka-tixo" -> "katixo"
    #
    # Candidate matching uses lowercase lexicon keys.
    # Original casing must be restored during P10 normalisation.
    for historical_word, frequency in token_counts.items():

        if "-" not in historical_word:
            continue

        candidate = historical_word.replace("-", "")

        if candidate in lexicon:
            rows.append({
                "hist_form": historical_word,
                "cand_form": candidate,
                "edit_distance": historical_word.count("-"),
                "hist_freq": frequency,
                "cand_freq": lexicon[candidate],
                "kind": "hyphen",
            })

    # STEP 6: Build the candidate table.
    pairs = pd.DataFrame(rows, columns=COLUMNS)

    if pairs.empty:
        return pairs

    # Prevent duplicate pairs within the same candidate category.
    pairs = pairs.drop_duplicates(
        subset=["hist_form", "cand_form", "kind"]
    )

    # Sort deterministically for reproducibility.
    return pairs.sort_values(
        ["kind", "hist_freq", "hist_form", "edit_distance"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)