"""P10.2 Historicisation (Member B). STUB. Tests: T12.
"""
from __future__ import annotations

import pandas as pd  # noqa: F401


def estimate_rates(h_dev_changes: list, h_dev_norm_texts: list, rules: dict) -> dict:
    """r_F = (H-dev words changed by F) / (positions in normalised H-dev text where inverse F applies)."""
    raise NotImplementedError("P10.2 (Member B): implement per handbook P10.2")


def historicise_text(text: str, rules: dict, rates: dict | None = None, dose: float | None = None, families: list | None = None, seed: int = 13) -> tuple[str, list[dict]]:
    """Exactly one of rates (matched mode) or dose (share of eligible positions).
    families restricts the families applied."""
    raise NotImplementedError("P10.2 (Member B): implement per handbook P10.2")
