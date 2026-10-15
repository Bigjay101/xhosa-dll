"""P8.1 Masked-LM pseudo-log-likelihood, PLL-word-l2r (Member C). STUB.

Implement per handbook P8.1 and D1. Keep these signatures; tests: T8, T9.
"""
from __future__ import annotations

import pandas as pd  # noqa: F401


def build_masks(word_ids: list[int | None], method: str = "l2r") -> list[list[int]]:
    """For each non-special position i, the positions to mask:
    l2r : {i} plus {j > i : word_ids[j] == word_ids[i]}
    orig: {i}
    Special positions (word_ids[i] is None) get no mask. Pure function (T8)."""
    raise NotImplementedError("P8.1 (Member C): implement per handbook P8.1")


def score_mlm(model, tokenizer, texts: list[str], method: str = "l2r", batch_size: int = 64, device: str = "cuda") -> list[dict]:
    """Per text: words = segment.words(text); enc = tokenizer(words, is_split_into_words=True).
    One masked copy per non-special position; batched; model.eval(); torch.no_grad().
    Fast path: model.roberta(...) then model.lm_head only at the masked position.
    Returns {total_nll, n_tokens, n_words, word_nll: list[float], word_pieces: list[int]}."""
    raise NotImplementedError("P8.1 (Member C): implement per handbook P8.1")
