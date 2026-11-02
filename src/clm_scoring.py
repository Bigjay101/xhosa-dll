"""P8.2 Causal-LM likelihood for MzansiLM (Member C). STUB.

Implement per handbook P8.2. Tests: T10.
"""
from __future__ import annotations

import pandas as pd  # noqa: F401


def token_to_word(offsets: list[tuple[int, int]], text: str) -> list[int | None]:
    """Map each token to the index of the whitespace word containing its first non-space
    character (None for tokens with no characters). Pure function (T10)."""
    raise NotImplementedError("P8.2 (Member C): implement per handbook P8.2")


def score_clm(model, tokenizer, texts: list[str], batch_size: int = 32) -> list[dict]:
    """Exact NLL of every text token given its prefix; prepend BOS if the tokenizer does not
    (CHECK). Same return fields as mlm_scoring.score_mlm."""
    raise NotImplementedError("P8.2 (Member C): implement per handbook P8.2")
