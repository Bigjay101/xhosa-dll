"""
P9.5: Validate historical spelling families using PPMI contexts.

Inputs:
    P9.2 candidate pairs
    P9.4 classified edit patterns
    Historical and modern PPMI vectors
    Frozen modern lexicon

Outputs:
    data/processed/p9_labeled_pairs.parquet
    results/rule_validation.csv

Historical test data is never used.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import P, load_config
from src.data_api import load_lexicon
from src.rules import edit_patterns, family_validity


def load_vectors(path: Path) -> dict:
    """Load target words and their saved PPMI vectors."""

    with np.load(path, allow_pickle=False) as data:
        terms = data["terms"]
        matrix = data["vectors"]

        return {
            str(word): vector
            for word, vector in zip(terms, matrix)
        }


def main() -> None:
    """Label candidate pairs and evaluate contextual validity."""

    cfg = load_config()

    # STEP 1: Load candidate pairs and classified patterns.
    pairs = pd.read_parquet(
        P(cfg, "processed", "candidate_pairs.parquet")
    )

    patterns = pd.read_csv(
        P(cfg, "results", "p9_family_patterns.csv"),
        keep_default_na=False,
    )

    # STEP 2: Map each retained edit pattern to its family.
    pattern_lookup = {
        (
            row.op,
            row.src,
            row.tgt,
            row.left_char,
            row.right_char,
        ): row.family
        for row in patterns.itertuples(index=False)
    }

    # STEP 3: Assign candidate pairs to families.
    #
    # Require all edits in a pair to belong to the same
    # retained non-OTHER family. This conservatively
    # rejects pairs with additional unexplained edits.
    labeled_rows = []

    for row in pairs.itertuples(index=False):

        # Multiword WD candidates require separate validation.
        if " " in row.hist_form:
            continue

        edits = edit_patterns(row.hist_form, row.cand_form)

        if not edits:
            continue

        families = [pattern_lookup.get(edit) for edit in edits]

        if (
            len(set(families)) == 1
            and families[0] not in {None, "OTHER", "WD"}
        ):
            labeled_rows.append({
                "hist_form": row.hist_form,
                "cand_form": row.cand_form,
                "hist_freq": row.hist_freq,
                "cand_freq": row.cand_freq,
                "family": families[0],
            })

    labeled = pd.DataFrame(
        labeled_rows,
        columns=[
            "hist_form",
            "cand_form",
            "hist_freq",
            "cand_freq",
            "family",
        ],
    ).drop_duplicates(
        ["hist_form", "cand_form", "family"]
    )

    if labeled.empty:
        raise RuntimeError(
            "No candidate pairs matched the retained rule patterns."
        )

    print("Candidate pairs:", len(pairs))
    print("Family-labeled pairs:", len(labeled))
    print("\nLabeled pairs by family:")
    print(labeled["family"].value_counts())

    labeled.to_parquet(
        P(cfg, "processed", "p9_labeled_pairs.parquet", mkdir=True),
        index=False,
    )

    # STEP 4: Load the two saved vector collections.
    historical_vectors = load_vectors(
        P(cfg, "interim", "p9_historical_vectors.npz")
    )

    modern_vectors = load_vectors(
        P(cfg, "interim", "p9_modern_vectors.npz")
    )

    # STEP 5: Load frozen modern word frequencies.
    lexicon = load_lexicon(cfg)

    # STEP 6: Compare candidate contexts against controls.
    print("\nValidating spelling families...", flush=True)

    validation = family_validity(
        pairs=labeled,
        hist_vecs=historical_vectors,
        mod_vecs=modern_vectors,
        mod_freqs=lexicon,
        n_boot=cfg["rules"]["validity_boot"],
        seed=cfg["seed"],
        min_support=cfg["rules"]["min_support"],
    )

    # STEP 7: Save and report the validation results.
    output = P(
        cfg,
        "results",
        "rule_validation.csv",
        mkdir=True,
    )

    validation.to_csv(output, index=False)

    print("\nP9.5 — FAMILY VALIDATION")
    print("=" * 70)
    print(validation.to_string(index=False))

    print("\nAccepted families:")
    print(
        validation.loc[
            validation["accepted"], "family"
        ].tolist()
    )

    print("\nSaved:", output)


if __name__ == "__main__":
    main()