"""P2.1 Cleaning (Member A).

Applied identically to historical and modern text. This is NOT the spelling
normalisation of P10: letters, diacritics and in-line hyphens are never changed.
"""
from __future__ import annotations

import re
import statistics
import unicodedata
from collections import Counter

_QUOTES = {"’": "'", "‘": "'", "ʼ": "'", "`": "'", "´": "'",
           "“": '"', "”": '"', "„": '"'}
_REMOVE = dict.fromkeys(map(ord, "­​‌‍⁠﻿"), None)
_WORD_CHARS = re.compile(r"[^\W\d_]", re.UNICODE)


def normalise_unicode(text: str) -> str:
    """NFC; curly quotes/apostrophes -> ' and "; remove soft hyphens and zero-width
    characters; NBSP -> space; collapse whitespace within lines (newlines kept)."""
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_REMOVE)
    for src, tgt in _QUOTES.items():
        text = text.replace(src, tgt)
    text = text.replace(" ", " ").replace(" ", " ")
    lines = [re.sub(r"[ \t\f\v]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def word_counts(texts) -> Counter:
    """Lowercased whitespace-token counts, used to decide line-break hyphen joins."""
    counts: Counter = Counter()
    for t in texts:
        counts.update(w.lower().strip(".,;:!?\"'()[]") for w in t.split())
    return counts


def join_linebreak_hyphens(lines: list[str], counts: Counter) -> list[str]:
    """Join lines; at a line ending 'abc-' followed by a line starting 'def', write
    'abcdef' only if it is more frequent in the corpus than 'abc-def', otherwise
    'abc-def'. In-line hyphens are never touched. Returns a single-element list
    holding the joined text (a list keeps the signature from the handbook)."""
    out = ""
    for line in (l for l in lines if l.strip()):
        if out.endswith("-") and len(out) > 1 and _WORD_CHARS.match(out[-2]) \
                and _WORD_CHARS.match(line[0]):
            left = out.rsplit(" ", 1)[-1][:-1]
            right = line.split(" ", 1)[0]
            joined = (left + right).lower().strip(".,;:!?\"'()")
            hyphenated = (left + "-" + right).lower().strip(".,;:!?\"'()")
            if counts.get(joined, 0) > counts.get(hyphenated, 0):
                out = out[:-1] + line
            else:
                out = out + line
        else:
            out = (out + " " + line) if out else line
    return [out]


def _is_heading(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return len(text.split()) < 4 or (len(letters) >= 4 and all(c.isupper() for c in letters))


def build_blocks(paragraphs: list[dict], counts: Counter | None = None,
                 merge_if_median_below: int = 15, min_block_words: int = 3) -> list[dict]:
    """Group one edition's paragraphs into text blocks.

    If the median non-empty paragraph has fewer than `merge_if_median_below` words
    (i.e. paragraphs are really layout lines), consecutive non-empty paragraphs are
    merged, breaking at empty paragraphs, table/text-box boundaries and heading-like
    lines (all caps, or under 4 words). Otherwise each paragraph is a block.
    Blocks under `min_block_words` words are dropped.
    Returns [{block_idx, text, n_words}] in document order.
    """
    counts = counts or Counter()
    paras = [dict(p, text=normalise_unicode(p.get("text", ""))) for p in paragraphs]
    lengths = [len(p["text"].split()) for p in paras if p["text"]]
    if not lengths:
        return []
    merge = statistics.median(lengths) < merge_if_median_below

    groups: list[list[str]] = []
    if merge:
        current: list[str] = []
        context = None
        for p in paras:
            ctx = (p.get("in_table", False), p.get("in_textbox", False))
            if not p["text"] or ctx != context or _is_heading(p["text"]):
                if current:
                    groups.append(current)
                current = []
                context = ctx
                if p["text"] and _is_heading(p["text"]):
                    groups.append([p["text"]])
                    continue
            if p["text"]:
                current.extend(p["text"].split("\n"))
        if current:
            groups.append(current)
    else:
        groups = [p["text"].split("\n") for p in paras if p["text"]]

    blocks = []
    for lines in groups:
        text = join_linebreak_hyphens(lines, counts)[0]
        n = len(text.split())
        if n >= min_block_words:
            blocks.append({"block_idx": len(blocks), "text": text, "n_words": n})
    return blocks
