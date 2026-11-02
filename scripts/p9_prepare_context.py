"""
P9.5: Prepare modern reference text for contextual rule validation.

Use only the modern splits that contributed to the P5 lexicon:
    M1: Vuk'uzenzele train and eval
    M2: MzansiText validation

"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Allow direct execution from the scripts directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.clean import normalise_unicode
from src.config import P, load_config
from src.modern_data import load_mzansi, load_vuk


def main() -> None:
    """
    Recover and cache modern lexicon-role text for P9.5.

    Prefer existing local data and retrieve missing text
    through the project's established P5 loaders.
    """

    # STEP 1: Load project configuration.
    cfg = load_config()

    output = P(
        cfg,
        "interim",
        "p9_modern_context.parquet",
        mkdir=True,
    )

    # Reuse the complete P9 context cache on subsequent runs.
    if output.exists():
        print(f"Existing P9 context dataset: {output}")

        existing = pd.read_parquet(output)

        print(existing.groupby(["source", "split"]).size())
        return

    # STEP 2: Recover Vuk'uzenzele isiXhosa text.
    vuk_cache = P(cfg, "interim", "vuk_langid.parquet")

    if vuk_cache.exists():

        print("Loading cached Vuk'uzenzele text...")

        vuk = pd.read_parquet(vuk_cache)

        required = {"text", "label", "split"}

        if not required.issubset(vuk.columns):
            raise ValueError(
                f"Vuk cache is missing {required - set(vuk.columns)}"
            )

        # Retain only isiXhosa lexicon-role documents.
        m1 = vuk.loc[
            (vuk["label"] == "xho")
            & (vuk["split"].isin(["train", "eval"])),
            ["split", "text"],
        ].copy()

    else:

        print("Downloading Vuk'uzenzele isiXhosa...")

        settings = cfg["modern"]["vuk"]

        vuk = load_vuk(
            settings["hf_id"],
            settings["config"],
            settings["text_field"],
        )

        m1 = vuk.loc[
            vuk["split"].isin(["train", "eval"]),
            ["split", "text"],
        ].copy()

    if m1.empty:
        raise ValueError("No M1 lexicon-role documents found.")

    m1["source"] = "M1"

    # STEP 3: Recover MzansiText validation documents.
    #
    # This uses the same split, language code and row limit
    # as the original P5 data-processing pipeline.
    m2_cache = P(
        cfg,
        "interim",
        "p9_mzansi_validation.parquet",
    )

    if m2_cache.exists():

        print("Loading cached MzansiText validation text...")

        m2 = pd.read_parquet(m2_cache)

    else:

        print("Streaming MzansiText validation documents...")

        settings = cfg["modern"]["mzansi"]

        m2 = load_mzansi(
            hf_id=settings["hf_id"],
            split="validation",
            lang_value=settings["lang_value"],
            max_rows=settings["max_rows"],
        )

        if m2.empty:
            raise ValueError(
                "No isiXhosa MzansiText validation documents found."
            )

        m2.to_parquet(m2_cache, index=False)

    m2 = m2[["split", "text"]].copy()

    m2["source"] = "M2"

    # STEP 4: Combine the two permitted reference corpora.
    modern = pd.concat(
        [m1, m2],
        ignore_index=True,
    )

    # Apply the same Unicode normalisation used in P5.
    modern["text"] = modern["text"].map(
        lambda text: normalise_unicode(str(text)).replace("\n", " ")
    )

    # Remove empty documents.
    modern = modern.loc[
        modern["text"].str.strip().ne("")
    ].copy()

    # STEP 5: Verify that no modern test data entered.
    assert set(modern["split"]).issubset(
        {"train", "eval", "validation"}
    )

    assert not modern.empty

    # STEP 6: Save without changing any frozen P5 files.
    modern.to_parquet(output, index=False)

    # STEP 7: Print data coverage.
    modern["n_words"] = modern["text"].str.split().str.len()

    summary = (
        modern.groupby(["source", "split"])
        .agg(
            documents=("text", "size"),
            words=("n_words", "sum"),
        )
    )

    print("\nP9.5 — MODERN CONTEXT CORPUS")
    print("=" * 55)
    print(summary)

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()

# First, it checks whether vuk_langid.parquet exists. We previously generated this file during historical language identification, so we may be able to reuse it.

# Second, it retrieves the MzansiText validation documents using the same configuration as our original P5 pipeline. This may require an internet connection.

# Third, it applies the same text normalisation used in P5 and saves the combined reference corpus to a new file.