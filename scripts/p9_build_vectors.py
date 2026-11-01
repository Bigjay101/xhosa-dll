"""
P9.5: Build historical and modern PPMI context vectors.

Historical data:
    Training split only.

Modern data:
    Recovered lexicon-role text only.

Outputs:
    data/interim/p9_historical_vectors.npz
    data/interim/p9_modern_vectors.npz

No historical or modern test text is used.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import P, load_config
from src.data_api import load_hist_chunks, load_lexicon
from src.lexicon import lex_key
from src.rules import context_vectors
from src.segment import words


def save_vectors(
    vectors: dict[str, np.ndarray],
    path: Path,
    n_anchors: int,
) -> None:
    """
    Save context vectors in a compressed NumPy file.

    Each row corresponds to one target word.
    Each column corresponds to a shared context anchor.
    """

    # STEP 1: Ensure we actually generated vectors.
    if not vectors:
        raise ValueError("No context vectors were generated")

    # STEP 2: Sort words for deterministic row ordering.
    terms = sorted(vectors)

    # STEP 3: Construct a matrix containing every word vector.
    matrix = np.stack(
        [vectors[word] for word in terms]
    ).astype(np.float32)

    # STEP 4: Verify vector dimensionality.
    assert matrix.shape == (len(terms), n_anchors)

    # STEP 5: Save words and their corresponding vectors.
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        path,
        terms=np.asarray(terms),
        vectors=matrix,
    )

    print(f"Saved: {path}")
    print(f"Target words with usable contexts: {len(terms):,}")


def main() -> None:
    """
    Build comparable PPMI context vectors for historical
    and modern isiXhosa.
    """

    # STEP 1: Load configuration.
    cfg = load_config()

    # STEP 2: Load the selected context anchors.
    anchors_path = P(
        cfg,
        "results",
        "p9_anchors.txt",
    )

    anchors = anchors_path.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(anchors) == cfg["rules"]["anchors"]
    assert len(set(anchors)) == len(anchors)

    print(f"Loaded {len(anchors)} shared anchors")

    # STEP 3: Load historical TRAINING chunks only.
    historical = load_hist_chunks(cfg=cfg)

    train = historical.loc[
        historical["split"] == "train"
    ]

    print(f"Historical training chunks: {len(train):,}")

    # STEP 4: Load our 45,818 candidate pairs.
    pairs = pd.read_parquet(
        P(cfg, "processed", "candidate_pairs.parquet")
    )

    # Use single-word historical forms.
    # Word-division candidates need separate handling.
    historical_targets = {
        lex_key(form)
        for form in pairs["hist_form"]
        if len(words(form)) == 1
    }

    historical_targets.discard("")

    print(
        "Historical target types:",
        len(historical_targets)
    )

    # STEP 5: Construct historical PPMI context vectors.
    print("\nBuilding historical context vectors...", flush=True)

    historical_vectors = context_vectors(
        token_lists=(
            words(text)
            for text in train["text"]
        ),
        targets=historical_targets,
        anchors=anchors,
        window=cfg["rules"]["context_window"],
    )

    save_vectors(
        historical_vectors,
        P(
            cfg,
            "interim",
            "p9_historical_vectors.npz",
            mkdir=True,
        ),
        len(anchors),
    )

    # STEP 6: Load the recovered modern reference corpus.
    modern = pd.read_parquet(
        P(cfg, "interim", "p9_modern_context.parquet")
    )

    # Explicitly verify that modern test text is absent.
    assert set(modern["split"]).issubset(
        {"train", "eval", "validation"}
    )

    # STEP 7: Use the frozen modern lexicon for target words.
    #
    # We include the lexicon vocabulary so that P9.5 can
    # construct comparison vectors for matched control words.
    lexicon = load_lexicon(cfg)

    modern_targets = set(lexicon)

    print("\nModern target types:", len(modern_targets))

    # STEP 8: Construct modern PPMI context vectors.
    print("\nBuilding modern context vectors...", flush=True)

    modern_vectors = context_vectors(
        token_lists=(
            words(text)
            for text in modern["text"]
        ),
        targets=modern_targets,
        anchors=anchors,
        window=cfg["rules"]["context_window"],
    )

    save_vectors(
        modern_vectors,
        P(
            cfg,
            "interim",
            "p9_modern_vectors.npz",
            mkdir=True,
        ),
        len(anchors),
    )

    print("\nP9.5 context-vector construction complete.")


if __name__ == "__main__":
    main()