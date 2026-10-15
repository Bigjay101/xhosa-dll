"""P5.4 Modern isiXhosa lexicon (Member A).

Built from modern LEXICON-role text only (MzansiText validation + Vuk'uzenzele
train/eval), never from modern test text.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable


def lex_key(word: str) -> str:
    """Lowercase; strip leading/trailing punctuation; keep internal ' and -."""
    w = word.lower()
    start, end = 0, len(w)
    while start < end and not w[start].isalnum():
        start += 1
    while end > start and not w[end - 1].isalnum():
        end -= 1
    return w[start:end]


def build_lexicon(texts: Iterable[str], min_freq: int = 2) -> Counter:
    """Counter of lex_key forms with frequency >= min_freq (empty keys and pure
    numbers dropped)."""
    counts: Counter = Counter()
    for t in texts:
        counts.update(k for k in (lex_key(w) for w in t.split()) if k and not k.isdigit())
    return Counter({w: c for w, c in counts.items() if c >= min_freq})


def save_lexicon(lex: Counter, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("word\tfreq\n")
        for w, c in lex.most_common():
            f.write(f"{w}\t{c}\n")


def load_lexicon_file(path: str | Path) -> Counter:
    lex: Counter = Counter()
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            w, c = line.rstrip("\n").split("\t")
            lex[w] = int(c)
    return lex


def oov_rate(text: str, lexicon: Counter) -> float:
    """Share of whitespace words whose lex_key is not in the lexicon. Pure
    punctuation (empty key) and pure numbers count as in-vocabulary."""
    ws = text.split()
    if not ws:
        return 0.0
    return sum(1 for w in ws if is_oov(w, lexicon)) / len(ws)


def is_oov(word: str, lexicon: Counter) -> bool:
    k = lex_key(word)
    return bool(k) and not k.isdigit() and k not in lexicon
