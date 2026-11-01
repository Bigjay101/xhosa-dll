"""
P9.1: Audit character frequencies in historical and modern isiXhosa.

Historical input:
    Training split only.

Modern input:
    Development chunks from lexicon-role corpus splits.

Output:
    results/char_audit.csv
"""

import sys
from pathlib import Path

# Allow execution from the repository's scripts directory.
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from src.config import P, load_config
from src.data_api import load_hist_chunks, load_modern_chunks
from src.orthography import char_audit


def main() -> None:
    """
    Load permitted training/reference text, perform the
    character audit, and save the resulting table.
    """

    # STEP 1: Load configuration and historical chunks.
    cfg = load_config()

    historical = load_hist_chunks(cfg=cfg)

    # STEP 2: Use historical TRAIN only.
    historical = historical[
        historical["split"] == "train"
    ]

    # STEP 3: Load modern development chunks.
    # These originate from the lexicon-role splits.
    modern = load_modern_chunks(
        role="dev",
        cfg=cfg
    )

    # STEP 4: Organise text by source.
    texts_by_source = {}

    for source, group in historical.groupby("source"):
        texts_by_source[source] = group["text"]

    for source, group in modern.groupby("source"):
        texts_by_source[f"modern_{source}"] = group["text"]

    # STEP 5: Run the audit.
    result = char_audit(texts_by_source)

    # STEP 6: Save the output.
    output = P(
        cfg,
        "results",
        "char_audit.csv",
        mkdir=True
    )

    result.to_csv(output, index=False)

    print(f"Character audit saved: {output}")
    print(f"Characters analysed: {len(result)}")

    # STEP 7: Display candidate historically enriched letters.
    enriched = result[
        result["historically_enriched"]
    ]

    print("\nHistorically enriched non-ASCII letters:")
    print(
        enriched[
            ["source", "character", "count", "rate_per_10k"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()