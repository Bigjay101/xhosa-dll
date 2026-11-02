"""P2.2 Deduplication (Member A).

Runs BEFORE splitting so repeated adverts and notices cannot appear in both
training and test data. Keeps the earliest-dated copy of each duplicate.
"""
from __future__ import annotations

import re

import pandas as pd
from datasketch import MinHash, MinHashLSH


def exact_key(text: str) -> str:
    """Lowercase letters, digits and single spaces only."""
    t = re.sub(r"[^\w\s]", " ", text.lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", t).strip()


def _minhash(words: list[str], shingle: int, num_perm: int) -> MinHash:
    mh = MinHash(num_perm=num_perm, seed=13)
    grams = [" ".join(words[i:i + shingle]) for i in range(max(1, len(words) - shingle + 1))]
    for g in grams:
        mh.update(g.encode("utf-8"))
    return mh


def dedup_blocks(blocks: list[dict], threshold: float = 0.8, shingle: int = 5,
                 num_perm: int = 128, exact_only_below: int = 10
                 ) -> tuple[list[dict], pd.DataFrame]:
    """Exact duplicates: key = lowercase letters/digits/spaces only; keep the
    earliest-dated copy. Near duplicates: MinHash LSH on word 5-shingles with
    Jaccard >= threshold (confirmed with the MinHash estimate), keep earliest.
    Blocks under `exact_only_below` words: exact matching only.
    Returns (kept blocks, report DataFrame with removals per source/year/kind)."""
    order = sorted(range(len(blocks)), key=lambda i: (
        str(blocks[i].get("date") or "9999"), blocks[i].get("source", ""),
        blocks[i].get("edition_id", ""), blocks[i].get("block_idx", 0)))
    seen_exact: set[str] = set()
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    hashes: dict[str, MinHash] = {}
    kept, removed = [], []
    for i in order:
        b = blocks[i]
        key = exact_key(b["text"])
        if key in seen_exact:
            removed.append((b, "exact"))
            continue
        seen_exact.add(key)
        words = key.split()
        if len(words) >= exact_only_below:
            mh = _minhash(words, shingle, num_perm)
            hits = [h for h in lsh.query(mh) if hashes[h].jaccard(mh) >= threshold]
            if hits:
                removed.append((b, "near"))
                continue
            name = f"b{i}"
            lsh.insert(name, mh)
            hashes[name] = mh
        kept.append(b)
    rows = [{"source": b.get("source"), "year": b.get("year"), "kind": kind,
             "n_words": b.get("n_words", len(b["text"].split()))} for b, kind in removed]
    if rows:
        report = (pd.DataFrame(rows).groupby(["source", "year", "kind"], dropna=False)
                  .agg(n_removed=("n_words", "size"), words_removed=("n_words", "sum"))
                  .reset_index())
    else:
        report = pd.DataFrame(columns=["source", "year", "kind", "n_removed", "words_removed"])
    kept.sort(key=lambda b: (b.get("source", ""), b.get("edition_id", ""), b.get("block_idx", 0)))
    return kept, report
