"""P4.2-P4.3 Edition-level splits and evaluation samples (Member A).

Splits are by newspaper edition, never by chunk, so text from one edition cannot
leak between train, dev and test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _pattern(fractions: tuple[float, float, float], length: int = 20) -> list[str]:
    """Interleaved split pattern, e.g. 60/20/20 -> 12 train, 4 dev, 4 test spread evenly."""
    names = ("train", "dev", "test")
    counts = [round(f * length) for f in fractions]
    counts[0] += length - sum(counts)
    items = [((j + 0.5) / c, k, name) for k, (name, c) in enumerate(zip(names, counts)) for j in range(c)]
    return [name for _, _, name in sorted(items)]


def split_editions(meta: pd.DataFrame, fractions=(0.6, 0.2, 0.2), seed: int = 13) -> pd.DataFrame:
    """Group = edition_id, stratified by year within each source.

    Editions are sorted by year (random order within a year) and labelled with an
    interleaved train/dev/test pattern (60/20/20 -> 12/4/4 per 20) from a seeded
    random starting point. Every run of 5 consecutive editions contains all three
    splits, so each year contributes to train, dev and test in proportion, and the
    overall shares match the fractions almost exactly.
    Input columns: edition_id, source, year (year may be missing).
    Returns edition_id, source, year, split."""
    rng = np.random.default_rng(seed)
    pattern = _pattern(tuple(fractions))
    eds = (meta[["edition_id", "source", "year"]].drop_duplicates("edition_id")
           .assign(stratum_year=lambda d: d["year"].fillna(-1).astype(int))
           .sort_values(["source", "stratum_year", "edition_id"]).reset_index(drop=True))
    eds["_key"] = rng.random(len(eds))
    eds = eds.sort_values(["source", "stratum_year", "_key"]).reset_index(drop=True)
    splits = []
    for _, g in eds.groupby("source", sort=True):
        offset = int(rng.integers(len(pattern)))
        splits.extend(pattern[(offset + i) % len(pattern)] for i in range(len(g)))
    eds["split"] = splits
    return eds[["edition_id", "source", "year", "split"]]


def sample_eval(chunks: pd.DataFrame, per_source: int, max_per_group: int,
                group_col: str, seed: int = 13) -> pd.Series:
    """Seeded sample of up to `per_source` chunks per source with at most
    `max_per_group` per group (edition or document). Returns a Series aligned with
    `chunks.index`: the 0-based sample rank (random order of selection) or -1 if
    not sampled. The depth subset is the first n by this rank (handbook P4.3)."""
    rank = pd.Series(-1, index=chunks.index, dtype=int)
    rng = np.random.default_rng(seed)
    for _, g in chunks.groupby("source", sort=True):
        order = rng.permutation(g.index.to_numpy())
        taken, per_group = 0, {}
        for idx in order:
            grp = g.at[idx, group_col]
            if per_group.get(grp, 0) >= max_per_group:
                continue
            per_group[grp] = per_group.get(grp, 0) + 1
            rank.at[idx] = taken
            taken += 1
            if taken >= per_source:
                break
    return rank
