"""T1 (Member A): cleaning and block building."""
from collections import Counter

from src.clean import build_blocks, join_linebreak_hyphens, normalise_unicode


def test_normalise_unicode_quotes_and_invisibles():
    s = "intomb’omlungu  “abantu”­​ngoku kakhulu"
    assert normalise_unicode(s) == "intomb'omlungu \"abantu\"ngoku kakhulu"


def test_nfc_applied():
    decomposed = "é"  # e + combining acute
    assert normalise_unicode(decomposed) == "é"


def test_inline_hyphens_kept():
    assert normalise_unicode("sama-Xosa ka-Tixo") == "sama-Xosa ka-Tixo"


def test_linebreak_hyphen_joined_only_when_more_frequent():
    counts = Counter({"abantu": 5, "aba-ntu": 0, "sama-xosa": 3, "samaxosa": 0})
    assert join_linebreak_hyphens(["aba-", "ntu bonke"], counts) == ["abantu bonke"]
    assert join_linebreak_hyphens(["Isigidimi sama-", "Xosa"], counts) == ["Isigidimi sama-Xosa"]


def test_build_blocks_merges_layout_lines_and_breaks_on_empty():
    paras = [{"para_idx": i, "text": t, "in_table": False, "in_textbox": False} for i, t in enumerate([
        "abantu bonke bathi ngoku kakhulu", "umntu uthetha khona phantsi phezulu", "",
        "ukuba kwaye kodwa ngoku kakhulu", "abafundisi bafunda incwadi ekhaya"])]
    blocks = build_blocks(paras, Counter(), merge_if_median_below=15, min_block_words=3)
    assert [b["n_words"] for b in blocks] == [10, 9]


def test_build_blocks_keeps_long_paragraphs_separate_and_drops_short():
    long = " ".join(["abantu"] * 20)
    paras = [{"para_idx": 0, "text": long}, {"para_idx": 1, "text": long}, {"para_idx": 2, "text": "ewe"}]
    blocks = build_blocks(paras, Counter(), merge_if_median_below=15, min_block_words=3)
    assert len(blocks) == 2 and all(b["n_words"] == 20 for b in blocks)
