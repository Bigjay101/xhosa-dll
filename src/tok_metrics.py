"""P6 Tokenizer metrics, experiment X1 (Member A; B reruns on new conditions).

Per chunk: fertility, PCW, P4, CPT, UNK and OOV (handbook D1). Special tokens are
never counted. Works with Hugging Face fast tokenizers (word_ids or offset
mapping) and with any object offering .tokenize() (slow tokenizers, test doubles).
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from src.lexicon import oov_rate
from src.segment import words as split_words


def _pieces_fast(tokenizer, words: list[str]) -> tuple[list[int], list[int]]:
    """Pieces per word and token ids, via word_ids() (needs a fast tokenizer)."""
    try:
        enc = tokenizer(words, is_split_into_words=True, add_special_tokens=False)
        ids = list(enc["input_ids"])
        wids = enc.word_ids()
    except Exception:  # noqa: BLE001 - e.g. byte-level BPE needing add_prefix_space
        text = " ".join(words)
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = list(enc["input_ids"])
        char_to_word = []
        for wi, w in enumerate(words):
            char_to_word.extend([wi] * len(w))
            char_to_word.append(None)  # the space
        wids = []
        for start, end in enc["offset_mapping"]:
            wid = None
            for c in range(start, end):
                if c < len(char_to_word) and char_to_word[c] is not None:
                    wid = char_to_word[c]
                    break
            wids.append(wid)
    counts = [0] * len(words)
    for wid in wids:
        if wid is not None:
            counts[wid] += 1
    return counts, ids


def word_pieces(tokenizer, words: list[str]) -> list[int]:
    """Number of subword pieces per word, special tokens excluded."""
    return _tokenize_words(tokenizer, words)[0]


def _tokenize_words(tokenizer, words: list[str]) -> tuple[list[int], list[int]]:
    if getattr(tokenizer, "is_fast", False):
        return _pieces_fast(tokenizer, words)
    counts, ids = [], []
    for w in words:  # slow tokenizers / test doubles: tokenize word by word
        toks = tokenizer.tokenize(w)
        counts.append(len(toks))
        ids.extend(tokenizer.convert_tokens_to_ids(toks))
    return counts, ids


def chunk_tok_metrics(tokenizer, text: str, lexicon: Counter) -> dict:
    """{fertility, pcw, p4, cpt, unk, oov} for one chunk (formulas in handbook D1)."""
    ws = split_words(text)
    counts, ids = _tokenize_words(tokenizer, ws)
    n_words, n_tok = len(ws), max(1, sum(counts))
    unk_id = getattr(tokenizer, "unk_token_id", None)
    return {
        "fertility": sum(counts) / n_words,
        "pcw": sum(c >= 2 for c in counts) / n_words,
        "p4": sum(c >= 4 for c in counts) / n_words,
        "cpt": sum(len(w) for w in ws) / n_tok,
        "unk": (sum(i == unk_id for i in ids) / n_tok) if unk_id is not None else 0.0,
        "oov": oov_rate(text, lexicon),
    }


def cluster_bootstrap_mean_simple(x: np.ndarray, clusters: np.ndarray, n_boot: int = 2000,
                                  ci: float = 0.95, seed: int = 13) -> tuple[float, float, float]:
    """Mean with a percentile CI from resampling whole clusters. A lightweight
    version for X1; C's src/stats.py provides the full protocol (D2)."""
    x, clusters = np.asarray(x, float), np.asarray(clusters)
    uniq, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=x)
    ns = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    reps = sums[idx].sum(1) / ns[idx].sum(1)
    lo, hi = np.percentile(reps, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(x.mean()), float(lo), float(hi)


def summarise(chunk_table: pd.DataFrame, n_boot: int = 2000, seed: int = 13) -> pd.DataFrame:
    """Mean and cluster-bootstrap CI of every metric per tokenizer x condition."""
    rows = []
    for (tok, cond), g in chunk_table.groupby(["tokenizer", "condition"], sort=True):
        for metric in ("fertility", "pcw", "p4", "cpt", "unk", "oov"):
            mean, lo, hi = cluster_bootstrap_mean_simple(
                g[metric].to_numpy(), g["cluster_id"].to_numpy(), n_boot=n_boot, seed=seed)
            rows.append({"tokenizer": tok, "condition": cond, "metric": metric,
                         "n_chunks": len(g), "mean": mean, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)
