"""P8.1 Masked-LM pseudo-log-likelihood, PLL-word-l2r (Member C). STUB.

Implement per handbook P8.1 and D1. Keep these signatures; tests: T8, T9.
"""
from __future__ import annotations

import pandas as pd  # noqa: F401



def build_masks(word_ids: list[int | None], method: str = "l2r") -> list[list[int]]:
    """Build masking positions for word-level pseudo-log-likelihood."""
    # When predicting A₁, we mask A₁, A₂ and A₃. When predicting A₂, we mask A₂ and A₃. When predicting A₃, we only mask A₃.

    if method not in {"l2r", "orig"}:
        raise ValueError(f"Unknown masking method: {method}")

    masks = []

    for i, word_id in enumerate(word_ids):
        if word_id is None:
            continue

        positions = [i]

        if method == "l2r":
            positions.extend(
                j
                for j in range(i + 1, len(word_ids))
                if word_ids[j] == word_id
            )

        masks.append(positions)

    return masks



def score_mlm(model, tokenizer, texts: list[str], method: str = "l2r", batch_size: int = 64, device: str = "cuda") -> list[dict]:
    """Per text: words = segment.words(text); enc = tokenizer(words, is_split_into_words=True).
    One masked copy per non-special position; batched; model.eval(); torch.no_grad().
    Fast path: model.roberta(...) then model.lm_head only at the masked position.
    Returns {total_nll, n_tokens, n_words, word_nll: list[float], word_pieces: list[int]}."""
    raise NotImplementedError("P8.1 (Member C): implement per handbook P8.1")
