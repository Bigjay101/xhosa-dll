"""P10.1 Normalisation (Member B). STUB. Tests: T11.
"""
from __future__ import annotations

import pandas as pd  # noqa: F401


def normalise_text(text: str, lexicon, rules: dict, mode: str = "lexicon", families: list | None = None) -> tuple[str, list[dict]]:
    """mode='lexicon': rewrite a word only if it is not a modern word and a candidate is;
    choose the most frequent candidate; restore casing; log {pos, old, new, family}.
    mode='blind': apply every accepted pattern without the lexicon check.
    families: restrict to these families (None = all accepted)."""
    raise NotImplementedError("P10.1 (Member B): implement per handbook P10.1")
