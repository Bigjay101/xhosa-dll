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
    """Return a fastText GlotLID model, or None if fasttext or the download is
    unavailable (then P3.3 and X6.4 are skipped and logged). CHECK the file name
    and label format on the model card."""
    try:
        import fasttext  # type: ignore
        from huggingface_hub import hf_hub_download
        return fasttext.load_model(hf_hub_download(repo_id=repo, filename=filename))
    except Exception:  # noqa: BLE001 - any failure means "not available"
        return None


def glotlid_labels(model, texts: list[str]) -> list[str]:
    """Map GlotLID predictions to 'eng', 'xho' or 'other'. GlotLID labels look like
    '__label__xho_Latn' (CHECK on the model card)."""
    out = []
    for t in texts:
        labels, _ = model.predict(t.replace("\n", " "), k=1)
        lab = labels[0].replace("__label__", "") if labels else ""
        out.append("eng" if lab.startswith("eng") else "xho" if lab.startswith("xho") else "other")
    return out


def kappa(a: list[str], b: list[str]) -> float:
    return float(cohen_kappa_score(a, b))


def build_english_wordlist(eng_texts: list[str], xho_lexicon: Counter, size: int = 2000) -> set[str]:
    """Top `size` English lex_keys from Vuk'uzenzele eng, minus any word that is also
    in the modern isiXhosa lexicon (e.g. names and loan words)."""
    counts: Counter = Counter()
    for t in eng_texts:
        counts.update(k for k in (lex_key(w) for w in t.split()) if k and not k.isdigit())
    return {w for w, _ in counts.most_common(size) if w not in xho_lexicon}


def eng_token_rate(text: str, eng_words: set[str]) -> float:
    ws = text.split()
    return sum(1 for w in ws if lex_key(w) in eng_words) / len(ws) if ws else 0.0
