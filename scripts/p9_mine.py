"""
P9.2: Mine candidate historical-to-modern isiXhosa spelling pairs.

Inputs
------
1. Historical training chunks from hist_chunks.parquet.
2. Frozen modern isiXhosa lexicon from modern_lexicon.tsv.

Outputs
-------
1. data/processed/candidate_pairs.parquet
2. results/p9_candidate_summary.csv

Important
---------
Historical development and test chunks are NEVER used
during candidate discovery.

Candidate pairs are not yet validated spelling variants.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# STEP 1: Make project modules importable when running
# this script directly from the scripts/ directory.
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from src.config import P, base_parser, load_config
from src.data_api import load_hist_chunks, load_lexicon
from src.candidates import mine_candidate_pairs


def main() -> None:
    """
    Mine historical spelling candidates from the frozen training split.

    The function:
        1. Loads the historical corpus.
        2. Selects the training split.
        3. Loads the modern lexicon.
        4. Mines candidate spelling pairs.
        5. Saves candidates and summary statistics.
    """

    # STEP 2: Load project configuration.
    parser = base_parser(__doc__)
    args = parser.parse_args()

    cfg = load_config(
        args.config,
        smoke=args.smoke
    )

    print("\nP9.2 — HISTORICAL SPELLING CANDIDATE MINING")
    print("=" * 55)

    # STEP 3: Load the historical dataset through our data API.
    historical = load_hist_chunks(cfg=cfg)

    print(f"Total historical chunks loaded: {len(historical):,}")

    # STEP 4: Select ONLY historical training chunks.
    #
    # The 'split' column contains train/dev/test.
    # Do not use 'eval_set' here because unsampled
    # training chunks have eval_set='none'.
    train = historical.loc[
        historical["split"] == "train"
    ].copy()

    if train.empty:
        raise ValueError(
            "No historical training chunks were found."
        )

    if train["text"].isna().any():
        raise ValueError(
            "Historical training data contains missing text."
        )

    print(f"Training chunks selected: {len(train):,}")

    # Display the number of training chunks per newspaper.
    print("\nTraining chunks by historical source:")
    print(train["source"].value_counts().to_string())

    # STEP 5: Load the frozen modern word-frequency lexicon.
    #
    # This lexicon was built using the designated
    # modern lexicon-role data during P5.
    lexicon = load_lexicon(cfg)

    if not lexicon:
        raise ValueError("Modern lexicon is empty.")

    print(f"\nModern lexicon size: {len(lexicon):,} word types")

    # STEP 6: Retrieve the candidate-mining parameters.
    #
    # We use the predefined configuration rather than
    # choosing parameters based on test-set results.
    min_freq = cfg["rules"]["min_oov_freq"]
    max_distance = cfg["rules"]["max_edit_distance"]

    print("\nCandidate mining parameters:")
    print(f"Minimum historical frequency: {min_freq}")
    print(f"Maximum edit distance: {max_distance}")
    print("Maximum suggestions per historical type: 3")

    # STEP 7: Run our previously implemented candidate miner.
    
    print("\nMining candidate spelling pairs...", flush=True)

    pairs = mine_candidate_pairs(
        train_texts=train["text"],
        lexicon=lexicon,
        min_hist_freq=min_freq,
        max_edit_distance=max_distance,
        max_suggestions=3,
    )

    print(f"Candidate pairs discovered: {len(pairs):,}")

    # STEP 8: Validate the output structure.
    required = {
        "hist_form",
        "cand_form",
        "edit_distance",
        "hist_freq",
        "cand_freq",
        "kind",
    }

    if not required.issubset(pairs.columns):
        raise ValueError(
            f"Missing candidate columns: {required - set(pairs.columns)}"
        )

    # Every modern candidate must exist in the modern lexicon.
    if not pairs.empty:
        invalid = ~pairs["cand_form"].isin(lexicon)

        if invalid.any():
            raise ValueError(
                f"{invalid.sum()} candidates are absent "
                "from the modern lexicon."
            )

    # STEP 9: Save the candidate table.
    output = P(
        cfg,
        "processed",
        "candidate_pairs.parquet",
        mkdir=True,
    )

    pairs.to_parquet(output, index=False)

    print(f"\nCandidate pairs saved to:\n{output}")

    # STEP 10: Generate summary statistics.
    if not pairs.empty:

        summary = (
            pairs.groupby("kind")
            .agg(
                candidate_pairs=("hist_form", "size"),
                unique_hist_forms=("hist_form", "nunique"),
                unique_modern_forms=("cand_form", "nunique"),
            )
            .reset_index()
        )

    else:
        summary = pd.DataFrame(
            columns=[
                "kind",
                "candidate_pairs",
                "unique_hist_forms",
                "unique_modern_forms",
            ]
        )

    summary_path = P(
        cfg,
        "results",
        "p9_candidate_summary.csv",
        mkdir=True,
    )

    summary.to_csv(summary_path, index=False)

    print("\nCANDIDATE SUMMARY")
    print("=" * 55)
    print(summary.to_string(index=False))

    # STEP 11: Display high-frequency candidates.
    #
    # This is descriptive inspection only.
    # It must not be treated as validation of the rules.
    if not pairs.empty:

        print("\nTOP 20 CANDIDATE PAIRS")
        print("=" * 55)

        top = pairs.sort_values(
            ["hist_freq", "edit_distance"],
            ascending=[False, True]
        ).head(20)

        print(top.to_string(index=False))

    print("\nP9.2 candidate mining completed successfully.")


if __name__ == "__main__":
    main()