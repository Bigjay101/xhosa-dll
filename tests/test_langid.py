"""P3 (Member A): GlotLID label mapping works through the low-level fastText call,
which avoids the fastText/NumPy 2 crash."""
from src.langid import eng_token_rate, glotlid_labels


class _FakeLowLevel:
    def predict(self, text, k, threshold, on_unicode_error):
        assert "\n" not in text[:-1]  # newlines inside the text are removed
        return [(0.9, "__label__eng_Latn" if "the" in text else "__label__xho_Latn")]


class _FakeModel:
    f = _FakeLowLevel()


def test_glotlid_labels_low_level():
    assert glotlid_labels(_FakeModel(), ["the meeting\nwas held", "abantu bonke"]) == ["eng", "xho"]


def test_eng_token_rate():
    assert eng_token_rate("the meeting abantu bonke", {"the", "meeting"}) == 0.5
    assert eng_token_rate("", {"the"}) == 0.0


def test_english_wordlist_keeps_function_words_despite_noisy_lexicon():
    """A web-crawled isiXhosa lexicon may contain a few 'the'/'and'; they must stay on
    the English list. Names common in both languages are dropped."""
    from collections import Counter

    from src.langid import build_english_wordlist
    eng = ["the minister and the people of Mandela Bay"] * 50
    xho = Counter({"abantu": 5000, "the": 3, "and": 2, "mandela": 400, "bay": 300})
    words = build_english_wordlist(eng, xho, size=50)
    assert {"the", "and", "of", "minister", "people"} <= words
    assert "mandela" not in words
