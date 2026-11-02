"""P4.1 Words and fixed-length chunks (Member A).

`words()` is the single definition of a word used everywhere in the project:
chunking, fertility, scoring and analysis. Do not tokenise words any other way.
"""
from __future__ import annotations


def words(text: str) -> list[str]:
    """Whitespace words after cleaning. Punctuation stays attached."""
    return text.split()


def make_chunks(blocks: list[dict], n_words: int = 32, carry: tuple[str, ...] = ()) -> list[dict]:
    """Non-overlapping n_words windows inside each block; the remainder is dropped;
    chunks never cross blocks. Each block needs: block_id, source, edition_id, text.
    Fields named in `carry` are copied from the block to its chunks.
    chunk_id = f"{source}:{edition_id}:{block_id}:{k}"."""
    chunks = []
    for b in blocks:
        ws = words(b["text"])
        for k in range(len(ws) // n_words):
            piece = ws[k * n_words:(k + 1) * n_words]
            row = {"chunk_id": f"{b['source']}:{b['edition_id']}:{b['block_id']}:{k}",
                   "source": b["source"], "edition_id": b["edition_id"],
                   "block_id": b["block_id"], "k": k,
                   "text": " ".join(piece), "n_words": len(piece)}
            for key in carry:
                row[key] = b.get(key)
            chunks.append(row)
    return chunks
