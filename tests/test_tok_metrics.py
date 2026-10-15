"""T6 (Member A): tokenizer metrics with a toy tokenizer."""
from collections import Counter

import pytest

from src.tok_metrics import chunk_tok_metrics, word_pieces


class ToyTokenizer:
    """Splits words into 3-letter pieces; the word 'zzz' maps to <unk>."""
    unk_token_id = 0
    is_fast = False

    def tokenize(self, word):
        return [word[i:i + 3] for i in range(0, len(word), 3)]

    def convert_tokens_to_ids(self, toks):
        return [0 if t == "zzz" else 1 for t in toks]


def test_pieces_and_metrics():
    tok = ToyTokenizer()
    assert word_pieces(tok, ["abc", "abcdef", "abcdefghijkl"]) == [1, 2, 4]
    m = chunk_tok_metrics(tok, "abc abcdef abcdefghijkl zzz", Counter({"abc": 3}))
    assert m["fertility"] == pytest.approx(8 / 4)
    assert m["pcw"] == pytest.approx(2 / 4) and m["p4"] == pytest.approx(1 / 4)
    assert 0 <= m["p4"] <= m["pcw"] <= 1
    assert m["unk"] == pytest.approx(1 / 8)
    assert m["oov"] == pytest.approx(3 / 4)
    assert m["cpt"] == pytest.approx(24 / 8)


def test_repetition_invariant():
    tok = ToyTokenizer()
    lex = Counter()
    once = chunk_tok_metrics(tok, "abcdefg", lex)["fertility"]
    many = chunk_tok_metrics(tok, " ".join(["abcdefg"] * 5), lex)["fertility"]
    assert once == many
