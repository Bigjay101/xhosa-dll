"""T4 (Member A): words and fixed-length chunks."""
from src.segment import make_chunks, words


def test_words_keeps_punctuation():
    assert words("  Abantu, bonke!  ngoku ") == ["Abantu,", "bonke!", "ngoku"]


def test_chunks_exact_length_no_crossing():
    blocks = [{"block_id": "b1", "source": "imvo", "edition_id": "e1", "text": " ".join(f"w{i}" for i in range(70)), "year": 1885},
              {"block_id": "b2", "source": "imvo", "edition_id": "e1", "text": " ".join(f"v{i}" for i in range(31))}]
    chunks = make_chunks(blocks, 32, carry=("year",))
    assert len(chunks) == 2                       # 70 -> 2 chunks, remainder 6 dropped; 31 -> 0
    assert all(c["n_words"] == 32 for c in chunks)
    assert all(c["block_id"] == "b1" for c in chunks)
    assert chunks[1]["text"].split()[0] == "w32"
    assert chunks[0]["chunk_id"] == "imvo:e1:b1:0" and chunks[0]["year"] == 1885
