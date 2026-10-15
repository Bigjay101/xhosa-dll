"""P11.3-P11.4 Analysis helpers (Member C). STUB.

Implement per handbook D1 (depth effects). Tests: T15.
"""
from __future__ import annotations

import pandas as pd  # noqa: F401


def assign_category(word: str, rewritten: bool, lexicon, english: set[str]) -> str:
    """First match of NUMPUNCT -> ENG -> CAP -> REWRITE -> OOV -> INLEX (handbook P11.3)."""
    raise NotImplementedError("P11.3 (Member C): implement per handbook P11.3")


def category_contributions(hist_words: pd.DataFrame, mod_words: pd.DataFrame) -> pd.DataFrame:
    """G_k = p_k^H * mean_k^H - p_k^M * mean_k^M per category; contributions sum to G (T15).
    Inputs have columns: category, nll."""
    raise NotImplementedError("P11.3 (Member C): implement per handbook P11.3")


def explained_share(words: pd.DataFrame) -> dict:
    """OLS of word NLL on hist (raw) and hist + log2(pieces) + OOV + CAP + NUMPUNCT + ENG
    (adjusted), cluster-robust SEs; returns b_raw, b_adj, explained = 1 - b_adj/b_raw with CIs."""
    raise NotImplementedError("P11.3 (Member C): implement per handbook P11.3")


def context_gap(hist_words: pd.DataFrame, mod_words: pd.DataFrame, min_count: int = 5, n_boot: int = 10000, seed: int = 13) -> dict:
    """Mean over shared identically spelled types of [mean NLL in H - mean NLL in M];
    bootstrap over types (handbook D1)."""
    raise NotImplementedError("P11.3 (Member C): implement per handbook P11.3")


def dose_slope(share_changed, delta_nll) -> float:
    """Least-squares slope through the origin: sum(s*d)/sum(s^2) (handbook D1)."""
    raise NotImplementedError("P11.4 (Member C): implement per handbook P11.4")
