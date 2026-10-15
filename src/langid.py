"""P3 Fully automatic language filtering (Member A).

A character n-gram classifier trained on Vuk'uzenzele (config name = label),
an optional independent second opinion from GlotLID, and helpers for English
words inside isiXhosa blocks. Nobody reads text to check any of this.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.pipeline import Pipeline

from src.lexicon import lex_key


def text_to_blocks(text: str, block_words: int = 40) -> list[str]:
    """Split a long modern document into ~block_words-word pieces, so training
    examples resemble historical blocks in length."""
    ws = text.split()
    return [" ".join(ws[i:i + block_words]) for i in range(0, len(ws), block_words)
            if len(ws[i:i + block_words]) >= 5]


def fit_langid(texts: list[str], labels: list[str], seed: int = 13) -> Pipeline:
    """Character 1-4-gram TF-IDF + LogisticRegression, classes {eng, xho}."""
    model = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), lowercase=True,
                                  min_df=2, sublinear_tf=True, max_features=200_000)),
        ("clf", LogisticRegression(max_iter=2000, C=4.0, random_state=seed)),
    ])
    model.fit(texts, labels)
    return model


def predict_eng_prob(model: Pipeline, texts: list[str]) -> np.ndarray:
    """Probability that each text is English."""
    classes = list(model.classes_)
    return model.predict_proba(texts)[:, classes.index("eng")]


def load_glotlid(repo: str = "cis-lmu/glotlid", filename: str = "model.bin"):
    """Return (fastText GlotLID model, None) or (None, reason) if fasttext or the
    download is unavailable (then P3.3 and X6.4 are skipped and the reason logged).
    Install with `pip install fasttext-wheel` (or `fasttext`)."""
    try:
        import fasttext  # type: ignore
    except ImportError:
        return None, "fasttext not installed (pip install fasttext-wheel)"
    try:
        from huggingface_hub import hf_hub_download
        return fasttext.load_model(hf_hub_download(repo_id=repo, filename=filename)), None
    except Exception as exc:  # noqa: BLE001
        return None, f"download/load failed: {exc}"


def _top_label(model, text: str) -> str:
    """Top GlotLID label. Uses the low-level predictor because fastText's Python
    wrapper crashes under NumPy 2 ('Unable to avoid copy while creating an array')."""
    text = text.replace("\n", " ")
    try:
        pairs = model.f.predict(text + "\n", 1, 0.0, "strict")  # [(prob, label)]
        return pairs[0][1] if pairs else ""
    except AttributeError:
        labels, _ = model.predict(text, k=1)
        return labels[0] if labels else ""


def glotlid_labels(model, texts: list[str]) -> list[str]:
    """Map GlotLID predictions to 'eng', 'xho' or 'other'. GlotLID labels look like
    '__label__xho_Latn'."""
    out = []
    for t in texts:
        lab = _top_label(model, t).replace("__label__", "")
        out.append("eng" if lab.startswith("eng") else "xho" if lab.startswith("xho") else "other")
    return out


def kappa(a: list[str], b: list[str]) -> float:
    return float(cohen_kappa_score(a, b))


def build_english_wordlist(eng_texts: list[str], xho_lexicon: Counter, size: int = 2000,
                           ratio: float = 5.0) -> set[str]:
    """Top `size` English lex_keys from Vuk'uzenzele eng. A word is dropped only if it is
    NOT clearly more English than isiXhosa: kept when its relative frequency in English
    is at least `ratio` times its relative frequency in the isiXhosa lexicon. This keeps
    'the', 'and', 'of' even if web-crawled isiXhosa text contains a few of them, and
    drops names and loan words that are common in both."""
    counts: Counter = Counter()
    for t in eng_texts:
        counts.update(k for k in (lex_key(w) for w in t.split()) if k and not k.isdigit())
    n_eng = max(1, sum(counts.values()))
    n_xho = max(1, sum(xho_lexicon.values()))
    return {w for w, c in counts.most_common(size)
            if c / n_eng >= ratio * xho_lexicon.get(w, 0) / n_xho}


def eng_token_rate(text: str, eng_words: set[str]) -> float:
    ws = text.split()
    return sum(1 for w in ws if lex_key(w) in eng_words) / len(ws) if ws else 0.0
