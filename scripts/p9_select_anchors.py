"""
P9.5: Select shared historical-modern context anchors.
"""

import sys
from pathlib import Path

import pandas as pd

# Make the repository root available for imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import P, load_config
from src.data_api import (
    load_hist_chunks,
    load_lexicon,
    load_english_wordlist,
)
from src.rules import select_context_anchors

# STEP 1: Load configuration.
cfg = load_config()

english_words = load_english_wordlist(cfg)

# STEP 2: Load historical training chunks only.
historical = load_hist_chunks(cfg=cfg)

train = historical.loc[
    historical["split"] == "train"
]

print("Historical training chunks:", len(train))

# STEP 3: Load the frozen modern lexicon.
lexicon = load_lexicon(cfg)

print("Modern lexicon types:", len(lexicon))

# STEP 4: Load the previously mined candidate pairs.
pairs = pd.read_parquet(
    P(cfg, "processed", "candidate_pairs.parquet")
)

print("Candidate pairs:", len(pairs))

# STEP 5: Select our 2,000 context anchors.
anchors = select_context_anchors(
    historical_texts=train["text"],
    modern_lexicon=lexicon,
    candidate_pairs=pairs,
    n_anchors=cfg["rules"]["anchors"],
    excluded_words=english_words,
)

# STEP 6: Verify basic properties.
assert len(anchors) == 2000
assert len(set(anchors)) == 2000
assert all(word in lexicon for word in anchors)

# STEP 7: Save the anchor vocabulary.
output = P(
    cfg,
    "results",
    "p9_anchors.txt",
    mkdir=True,
)

output.write_text(
    "\n".join(anchors) + "\n",
    encoding="utf-8",
)

print("\nP9.5 — CONTEXT ANCHORS")
print("=" * 55)
print("Selected anchors:", len(anchors))
print("Unique anchors:", len(set(anchors)))

print("\nFirst 20 anchors:")
for word in anchors[:20]:
    print(word)

print("\nSaved to:", output)