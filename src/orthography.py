"""P9.1 Character audit (Member B). STUB.
"""
from __future__ import annotations

import pandas as pd  # noqa: F401
from collections import Counter
import unicodedata


def char_audit(texts_by_source: dict) -> pd.DataFrame:
    """Count every character per source; rate per 10,000 characters; flag non-ASCII letters
    much more common in historical than modern text (handbook P9.1).
    Notes
    -----
    A non-ASCII letter is flagged as historically enriched when:

    1. It appears at least 10 times in historical text.
    2. Its historical frequency rate is at least three times
       its modern frequency rate.

    This is an exploratory threshold, not statistical validation.
    """

    # STEP 1: Count every character separately for each source.
    source_counts = {}

    for source, texts in texts_by_source.items():

        counts = Counter()

        for text in texts:

            # Reject invalid data instead of silently counting it.
            if not isinstance(text, str):
                raise ValueError(
                    f"{source} contains a non-string text"
                )

            # Counter(text) counts individual Unicode characters.
            counts.update(text)

        source_counts[source] = counts

    # STEP 2: Pool historical and modern counts separately.
    historical_counts = Counter()
    modern_counts = Counter()

    for source, counts in source_counts.items():

        if source.startswith("modern_"):
            modern_counts.update(counts)
        else:
            historical_counts.update(counts)

    historical_total = sum(historical_counts.values())
    modern_total = sum(modern_counts.values())

    # STEP 3: Calculate pooled rates for every observed character.
    all_characters = set(historical_counts) | set(modern_counts)

    enriched_characters = set()

    for char in all_characters:

        # A Unicode letter has a category beginning with "L".
        is_non_ascii_letter = (
            ord(char) > 127
            and unicodedata.category(char).startswith("L")
        )

        hist_rate = (
            historical_counts[char] / historical_total * 10000
            if historical_total else 0.0
        )

        modern_rate = (
            modern_counts[char] / modern_total * 10000
            if modern_total else 0.0
        )

        # STEP 4: Flag unusually common historical letters.
        if (
            is_non_ascii_letter
            and historical_counts[char] >= 10
            and hist_rate >= 3 * modern_rate
            and historical_total > 0
            and modern_total > 0
        ):
            enriched_characters.add(char)

    # STEP 5: Build a table with one row per source and character.
    rows = []

    for source, counts in source_counts.items():

        total = sum(counts.values())

        for char, count in counts.items():

            rows.append({
                "source": source,
                "character": char,
                "unicode_codepoint": f"U+{ord(char):04X}",
                "unicode_name": unicodedata.name(
                    char, "UNKNOWN"
                ),
                "count": count,
                "rate_per_10k": count / total * 10000,
                "is_non_ascii_letter": (
                    ord(char) > 127
                    and unicodedata.category(char).startswith("L")
                ),
                "historically_enriched": (
                    char in enriched_characters
                ),
            })

    # STEP 6: Convert the observations into a DataFrame.
    result = pd.DataFrame(rows)

    # Sort so the most frequent characters appear first.
    if not result.empty:
        result = result.sort_values(
            ["source", "count"],
            ascending=[True, False]
        ).reset_index(drop=True)

    return result
